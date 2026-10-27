"""Чанки корпуса и чанкеры Chonkie.

Чанкер описан неизменяемой спецификацией `ChunkerSpec` — она же ключ кэша и строка итоговой таблицы.
Документы режутся независимо: чанк никогда не пересекает границу документа.
"""

from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd

from gigarag.corpus import Document

SHORT_CHUNK, LONG_CHUNK = 50, 500  # пороги ТЗ: доля чанков короче 50 и длиннее 500 токенов

ChunkTexts = Callable[[list[str]], list[list[str]]]  # тексты документов → тексты чанков каждого


@dataclass(frozen=True)
class Chunk:
    chunk_id: str   # <doc_id>@<позиция>
    doc_id: str
    position: int   # номер чанка в документе, с нуля
    text: str


def chunk_documents(docs: Sequence[Document], chunk_texts: ChunkTexts) -> list[Chunk]:
    """Чанки всех документов в порядке корпуса; пустые и пробельные чанки отброшены."""
    pieces = chunk_texts([d.text for d in docs])
    return [Chunk(f"{doc.doc_id}@{i}", doc.doc_id, i, text)
            for doc, texts in zip(docs, pieces, strict=True)
            for i, text in enumerate(t for t in texts if t.strip())]


def child_chunks(parents: Sequence[Chunk], chunk_texts: ChunkTexts) -> tuple[list[Chunk], list[int]]:
    """Дочерние чанки для схемы parent/child: каждый родитель режется отдельно, ребёнок помнит родителя.
    Возвращает детей и индекс родителя каждого ребёнка."""
    pieces = chunk_texts([p.text for p in parents])
    pairs = [(Chunk(f"{parent.chunk_id}/{j}", parent.doc_id, j, text), i)
             for i, (parent, texts) in enumerate(zip(parents, pieces, strict=True))
             for j, text in enumerate(t for t in texts if t.strip())]
    return [child for child, _ in pairs], [owner for _, owner in pairs]


@dataclass(frozen=True)
class ChunkerSpec:
    """Чанкер и его параметры. `embedder` — модель, по эмбеддингам которой ищутся границы (семантические)."""
    kind: str                      # token | sentence | recursive | semantic | neural
    chunk_size: int = 512
    overlap: int = 0
    threshold: float | None = None
    skip_window: int = 0           # > 0 — режим SDPM: слияние похожих групп через `skip_window` соседей
    similarity_window: int = 3
    embedder: str | None = None

    def params(self) -> dict:
        return asdict(self)

    @property
    def label(self) -> str:
        name = "sdpm" if self.kind == "semantic" and self.skip_window else self.kind
        extra = [f"{self.chunk_size}"] if self.kind != "neural" else []
        extra += [f"overlap={self.overlap}"] if self.overlap else []
        extra += [f"t={self.threshold}"] if self.threshold is not None else []
        extra += [f"skip={self.skip_window}"] if self.skip_window else []
        extra += [self.embedder.split("/")[-1]] if self.embedder else []
        return f"{name}({', '.join(extra)})"


def chonkie_texts(chunker: Any) -> ChunkTexts:
    """Адаптер чанкера Chonkie: документы по одному, последовательно — без мультипроцессинга `chunk_batch`
    результат не зависит от числа ядер."""
    return lambda texts: [[c.text for c in chunker.chunk(text)] for text in texts]


@dataclass(frozen=True)
class ChunkerResources:
    """Модели, которые нужны чанкерам: токенизатор длин и фабрика эмбеддингов для семантических.
    Передаются снаружи, поэтому модуль не знает, какие именно модели и где они лежат."""
    tokenizer: Any
    embeddings: Callable[[str], Any] | None = None   # имя модели → BaseEmbeddings Chonkie
    neural_model: Callable[[], Any] | None = None    # → NeuralChunker


def build_chunker(spec: ChunkerSpec, resources: ChunkerResources) -> ChunkTexts:
    import chonkie
    builders: dict[str, Callable[[], Any]] = {
        "token": lambda: chonkie.TokenChunker(resources.tokenizer, chunk_size=spec.chunk_size,
                                              chunk_overlap=spec.overlap),
        "sentence": lambda: chonkie.SentenceChunker(resources.tokenizer, chunk_size=spec.chunk_size,
                                                    chunk_overlap=spec.overlap),
        "recursive": lambda: chonkie.RecursiveChunker(resources.tokenizer, chunk_size=spec.chunk_size),
        "semantic": lambda: chonkie.SemanticChunker(
            resources.embeddings(spec.embedder), threshold=spec.threshold, chunk_size=spec.chunk_size,
            similarity_window=spec.similarity_window, skip_window=spec.skip_window),
        "neural": lambda: resources.neural_model(),
    }
    return chonkie_texts(builders[spec.kind]())


def chunk_length_stats(token_counts: Sequence[int], short: int = SHORT_CHUNK, long: int = LONG_CHUNK) -> dict:
    """Распределение длин чанков в токенах: среднее, медиана, std и доли коротких (< short) и длинных (> long)."""
    values = np.asarray(token_counts)
    return {"chunks": len(values), "mean": values.mean(), "median": float(np.median(values)), "std": values.std(),
            f"<{short}_%": (values < short).mean() * 100, f">{long}_%": (values > long).mean() * 100}


def chunks_frame(chunks: Sequence[Chunk], token_counts: Sequence[int]) -> pd.DataFrame:
    return pd.DataFrame({"chunk_id": [c.chunk_id for c in chunks], "doc_id": [c.doc_id for c in chunks],
                         "position": [c.position for c in chunks], "text": [c.text for c in chunks],
                         "tokens": token_counts})


def frame_chunks(frame: pd.DataFrame) -> list[Chunk]:
    return [Chunk(r.chunk_id, r.doc_id, int(r.position), r.text) for r in frame.itertuples(index=False)]


def neural_chunker(model_name: str, revision: str, *, stride: int = 256, device: str | None = None) -> Any:
    """NeuralChunker Chonkie на модели с закреплённой ревизией (по имени Chonkie грузит последнюю версию
    и принимает только англоязычные модели из своего списка)."""
    from chonkie import NeuralChunker
    from transformers import AutoModelForTokenClassification, AutoTokenizer

    from gigarag.embeddings import pick_device
    model = AutoModelForTokenClassification.from_pretrained(model_name, revision=revision)
    tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)
    device = device or pick_device()
    return NeuralChunker(model=model.to(device), tokenizer=tokenizer, device_map=device, stride=stride)


def chunk_boundaries(texts: Sequence[str]) -> frozenset[int]:
    """Внутренние границы разбиения — символьные смещения концов чанков (кроме последнего).
    Верно для чанкеров, у которых чанки подряд дают исходный текст (Sentence, Semantic)."""
    return frozenset(np.cumsum([len(t) for t in texts[:-1]]).tolist())


def boundary_agreement(chunks_a: Sequence[Chunk], chunks_b: Sequence[Chunk]) -> dict:
    """Насколько совпадают два разбиения одних документов: средний Jaccard множеств границ по документам,
    которые хоть одно разбиение разрезало, и среднее число чанков на такой документ у каждого."""
    def by_doc(chunks: Sequence[Chunk]) -> dict[str, list[str]]:
        grouped: dict[str, list[str]] = {}
        for c in chunks:
            grouped.setdefault(c.doc_id, []).append(c.text)
        return grouped

    a, b = by_doc(chunks_a), by_doc(chunks_b)
    split = [d for d in a if len(a[d]) > 1 or len(b.get(d, [])) > 1]
    jaccards = [_jaccard(chunk_boundaries(a[d]), chunk_boundaries(b.get(d, [])) ) for d in split]
    return {"разрезанных документов": len(split), "Jaccard границ": float(np.mean(jaccards)) if jaccards else 1.0,
            "чанков A": float(np.mean([len(a[d]) for d in split])) if split else 0.0,
            "чанков B": float(np.mean([len(b.get(d, [])) for d in split])) if split else 0.0}


def _jaccard(x: frozenset[int], y: frozenset[int]) -> float:
    return len(x & y) / len(x | y) if x | y else 1.0


def sentence_chunk_labels(text: str, pieces: Sequence[str]) -> tuple[list[int], list[str]]:
    """Предложения (и строки) чанков подряд и номер чанка каждого — для визуализации разбиения."""
    from gigarag.patterns import sentences
    pairs = [(i, s) for i, piece in enumerate(pieces) for line in piece.split("\n") for s in sentences(line)]
    return [i for i, _ in pairs], [s for _, s in pairs]
