"""Поисковая конфигурация, выбранная в КТ 3 (`reports/kt3.md`): baseline КТ 4 и основа КТ 5.

Корпус — все шаги чистки КТ 2, кроме удаления подписей Telegram; чанки — RecursiveChunker(512) по токенам
BGE-M3; dense — BGE-M3 по чанку с префиксом метаданных; BM25 (k1 = 0,3, b = 1,0) — по чистому тексту;
слияние — взвешенная сумма с alpha = 0,6; фильтр по патчу из запроса — до отбора кандидатов.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import pandas as pd

import gigarag.chunking
import gigarag.cleaning
import gigarag.metadata
import gigarag.noise
import gigarag.patterns
from gigarag import cleaning, metadata, tokens
from gigarag.cache import CacheStore, code_fingerprint, load_or_compute, read_parquet, write_parquet
from gigarag.chunking import (
    Chunk,
    ChunkerResources,
    ChunkerSpec,
    build_chunker,
    chunk_documents,
    chunks_frame,
    frame_chunks,
)
from gigarag.corpus import Document
from gigarag.embeddings import EmbedderSpec
from gigarag.experiment import docs_fingerprint, texts_fingerprint

EMBEDDER = EmbedderSpec("BAAI/bge-m3", "5617a9f61b028005a4858fdac845db406aefb181", 567)
CHUNKER = ChunkerSpec("recursive", chunk_size=512)
SKIPPED_CLEANING_STAGE = "signatures"  # без удаления подписей validation КТ 2 был лучше у обоих поисковиков
BM25_K1, BM25_B = 0.3, 1.0
ALPHA = 0.6  # вес dense во взвешенной сумме
POOL = 100   # кандидаты каждого поисковика для слияния


@dataclass(frozen=True)
class BaselineCorpus:
    clean: cleaning.CleanCorpus
    chunks: list[Chunk]
    meta: pd.DataFrame

    @property
    def texts(self) -> list[str]:
        """Чистый текст чанков — его индексирует BM25."""
        return [c.text for c in self.chunks]

    @property
    def dense_texts(self) -> list[str]:
        """Чанк с префиксом «заголовок | тип | патч | дата» — его эмбеддит BGE-M3 и читают реранкеры."""
        return [metadata.with_metadata_prefix(c.text, row)
                for c, row in zip(self.chunks, self.meta.to_dict("records"), strict=True)]


def build_corpus(raw_docs: Sequence[Document], tokenizer, cache: CacheStore) -> BaselineCorpus:
    """Чистка, чанки и метаданные чанков baseline; чанки и метаданные кэшируются по данным и коду."""
    stages = [s for s in cleaning.PIPELINE if s.name != SKIPPED_CLEANING_STAGE]
    clean = cleaning.run_stages(cleaning.CleanCorpus(raw_docs), stages)[-1][1]

    def make_chunks() -> pd.DataFrame:
        chunks = chunk_documents(clean.docs, build_chunker(CHUNKER, ChunkerResources(tokenizer)))
        return chunks_frame(chunks, tokens.token_counts(tokenizer, [c.text for c in chunks]))

    chunk_frame = load_or_compute(
        cache.path("chunks", "parquet", docs=docs_fingerprint(clean.docs), **CHUNKER.params(),
                   code=code_fingerprint(gigarag.patterns, gigarag.cleaning, gigarag.chunking)),
        make_chunks, load=read_parquet, save=write_parquet)
    chunks = frame_chunks(chunk_frame)

    def make_meta() -> pd.DataFrame:
        doc_tokens = dict(zip([d.doc_id for d in clean.docs],
                              tokens.token_counts(tokenizer, [d.text for d in clean.docs]), strict=True))
        return metadata.chunk_metadata(chunks, clean.docs, doc_tokens, tokens.offsets_fn(tokenizer))

    meta = load_or_compute(
        cache.path("chunk_meta", "parquet", chunks=texts_fingerprint([c.text for c in chunks]),
                   code=code_fingerprint(gigarag.metadata, gigarag.noise, gigarag.patterns),
                   tokenizer=EMBEDDER.revision),
        make_meta, load=read_parquet, save=write_parquet)
    return BaselineCorpus(clean, chunks, meta)
