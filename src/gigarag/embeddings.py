"""Эмбеддинги текстов моделями sentence-transformers с закреплённой ревизией."""

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np


def pick_device() -> str:
    """`cuda`, если доступен GPU, иначе `cpu` (результат тот же, только медленнее)."""
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


def encode_passages(texts: Sequence[str], model_name: str, revision: str, *, max_seq_length: int,
                    prefix: str = "", batch_size: int = 128, device: str | None = None) -> np.ndarray:
    """Нормированные эмбеддинги; `prefix` — служебный префикс модели (у e5 — `"passage: "`)."""
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name, revision=revision, device=device or pick_device())
    model.max_seq_length = max_seq_length
    return model.encode([prefix + t for t in texts], batch_size=batch_size,
                        normalize_embeddings=True, show_progress_bar=False)


def load_sentence_model(model_name: str, revision: str, *, max_seq_length: int, half: bool = False,
                        device: str | None = None):
    """Модель sentence-transformers; `half` — fp16 на GPU (на CPU игнорируется: там fp16 медленнее)."""
    from sentence_transformers import SentenceTransformer
    device = device or pick_device()
    model = SentenceTransformer(model_name, revision=revision, device=device)
    model.max_seq_length = max_seq_length
    return model.half() if half and device == "cuda" else model


def encode(model, texts: Sequence[str], *, prefix: str = "", batch_size: int = 64) -> np.ndarray:
    """Нормированные эмбеддинги уже загруженной моделью (float32)."""
    return model.encode([prefix + t for t in texts], batch_size=batch_size, normalize_embeddings=True,
                        show_progress_bar=False, convert_to_numpy=True).astype(np.float32)


def memo_embeddings(model, *, prefix: str = "", batch_size: int = 64):
    """Эмбеддинги для чанкеров Chonkie с памятью «текст → вектор».

    Семантический чанкер на каждом пороге заново эмбеддит те же предложения и окна; память делает
    перебор порогов почти бесплатным, а время эмбеддинга границ считается один раз на модель.
    """
    from chonkie import SentenceTransformerEmbeddings

    class MemoEmbeddings(SentenceTransformerEmbeddings):
        def __init__(self) -> None:
            super().__init__(model)
            self.memo: dict[str, np.ndarray] = {}

        def embed(self, text: str) -> np.ndarray:
            return self.embed_batch([text])[0]

        def embed_batch(self, texts: list[str]) -> list[np.ndarray]:
            missing = list(dict.fromkeys(t for t in texts if t not in self.memo))
            if missing:
                self.memo |= zip(missing, encode(model, missing, prefix=prefix, batch_size=batch_size), strict=True)
            return [self.memo[t] for t in texts]

    return MemoEmbeddings()


@dataclass(frozen=True)
class EmbedderSpec:
    """Модель эмбеддингов с закреплённой ревизией и служебными префиксами запроса и документа
    (e5 обучены с «query: »/«passage: », у instruct-моделей запрос начинается с инструкции)."""
    name: str
    revision: str
    params_millions: float
    query_prefix: str = ""
    passage_prefix: str = ""
    max_seq_length: int = 512

    @property
    def short_name(self) -> str:
        return self.name.split("/")[-1]

    def cache_params(self) -> dict:
        return {"model": self.name, "revision": self.revision, "max_seq": self.max_seq_length,
                "passage_prefix": self.passage_prefix}


def model_size_mb(model) -> float:
    """Размер весов в fp32, МБ: число параметров × 4 байта (как модель лежит на диске)."""
    return sum(p.numel() for p in model.parameters()) * 4 / 2 ** 20


def query_latencies(model, queries: Sequence[str], *, prefix: str = "",
                    search: Callable[[np.ndarray], object] | None = None) -> np.ndarray:
    """Время каждого запроса по одному, с: так запрос приходит в проде, без батча. С `search` в таймер входит
    и поиск по вектору запроса (end-to-end), без него — только эмбеддинг."""
    def one(query: str) -> float:
        start = time.perf_counter()
        vector = encode(model, [query], prefix=prefix, batch_size=1)[0]
        if search is not None:
            search(vector)
        return time.perf_counter() - start
    one(queries[0])  # прогрев: первая инициализация CUDA не в счёт
    return np.array([one(q) for q in queries])
