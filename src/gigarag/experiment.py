"""Верстак экспериментов КТ 2: общие ресурсы и один способ прогнать конфигурацию.

Конфигурация = чанки + записи индекса (что эмбеддится и ищется) + необязательный фильтр.
Каждая прогоняется двумя поисковиками — BM25 и плотным BGE-M3, — и оценивается на gold одними метриками.
Чанки и эмбеддинги кэшируются вместе со временем их расчёта: ключ — отпечаток текстов, параметры и код.
"""

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from gigarag import embeddings, metrics, search
from gigarag.cache import CacheStore, load_or_compute_timed, read_npy, read_parquet, timed, write_npy, write_parquet
from gigarag.chunking import Chunk, ChunkTexts, chunk_documents, chunks_frame, frame_chunks
from gigarag.corpus import Document, sha256_hex
from gigarag.gold import GoldItem
from gigarag.relevance import build_qrels
from gigarag.tokens import token_counts

RETRIEVERS = ("bm25", "dense")


def texts_fingerprint(texts: Sequence[str]) -> str:
    return sha256_hex(json.dumps(list(texts), ensure_ascii=False).encode("utf-8"))[:16]


def docs_fingerprint(docs: Sequence[Document]) -> str:
    return texts_fingerprint([f"{d.doc_id}\n{d.text}" for d in docs])


@dataclass(frozen=True)
class Entries:
    """Записи индекса: тексты и чанк-владелец каждой записи."""
    texts: list[str]
    owners: list[int]

    @classmethod
    def of_chunks(cls, chunks: Sequence[Chunk]) -> "Entries":
        return cls([c.text for c in chunks], list(range(len(chunks))))


@dataclass(frozen=True)
class RunResult:
    per_question: pd.DataFrame           # метрики каждого вопроса × поисковик
    seconds: dict[str, float]            # время построения индексов: bm25, dense (эмбеддинг записей)


@dataclass
class Workbench:
    cache: CacheStore
    items: list[GoldItem]
    tokenizer: Any
    encoder: Any                       # плотный поисковик (BGE-M3)
    encoder_params: dict               # модель, ревизия, длина, точность — входят в ключ кэша эмбеддингов
    code: str                          # отпечаток кода, от которого зависят чанки
    queries: dict[str, str] = field(init=False)
    query_vectors: dict[str, np.ndarray] = field(init=False)

    def __post_init__(self) -> None:
        self.queries = {i.qid: i.question for i in self.items}
        self.query_vectors = dict(zip(self.queries, embeddings.encode(self.encoder, list(self.queries.values())),
                                      strict=True))

    def chunks(self, docs: Sequence[Document], params: dict, chunker: Callable[[], ChunkTexts]
               ) -> tuple[list[Chunk], np.ndarray, float]:
        """Чанки корпуса, их длины в токенах и время чанкинга (с подсчётом токенов)."""
        def compute() -> pd.DataFrame:
            chunks = chunk_documents(docs, chunker())
            return chunks_frame(chunks, token_counts(self.tokenizer, [c.text for c in chunks]))

        path = self.cache.path("chunks", "parquet", docs=docs_fingerprint(docs), code=self.code, **params)
        frame, seconds = load_or_compute_timed(path, compute, load=read_parquet, save=write_parquet)
        return frame_chunks(frame), frame["tokens"].to_numpy(), seconds

    def vectors(self, texts: Sequence[str]) -> tuple[np.ndarray, float]:
        path = self.cache.path("vectors", "npy", texts=texts_fingerprint(texts), **self.encoder_params)
        return load_or_compute_timed(path, lambda: embeddings.encode(self.encoder, texts),
                                     load=read_npy, save=write_npy)

    def run(self, config: str, chunks: Sequence[Chunk], aliases: Mapping[str, str] | None = None,
            entries: Entries | None = None, masks: Mapping[str, np.ndarray] | None = None,
            retrievers: Sequence[str] = RETRIEVERS, chunk_tokens: Sequence[int] | None = None) -> RunResult:
        """Метрики конфигурации на всех вопросах gold; разбивка по split и типам — в `per_question`."""
        entries = entries or Entries.of_chunks(chunks)
        qrels = build_qrels(self.items, chunks, aliases)
        frames, seconds = [], {}
        for retriever in retrievers:
            scorer, seconds[retriever] = self._scorer(retriever, entries)
            rankings = search.search_all(scorer, self.queries, entries.owners, len(chunks), allowed=masks)
            per_q = metrics.per_question(rankings, qrels, self.items, chunk_tokens=chunk_tokens)
            frames.append(per_q.assign(config=config, retriever=retriever))
        return RunResult(pd.concat(frames, ignore_index=True), seconds)

    def _scorer(self, retriever: str, entries: Entries) -> tuple[search.Scorer, float]:
        if retriever == "bm25":
            return timed(lambda: search.bm25_scorer(entries.texts))
        vectors, seconds = self.vectors(entries.texts)
        return search.dense_scorer(vectors, self.query_vectors, self.queries), seconds
