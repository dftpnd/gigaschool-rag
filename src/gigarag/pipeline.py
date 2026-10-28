"""Поисковый пайплайн с замером каждого этапа: трансформация запроса → эмбеддинг → поиск → слияние → реранкинг.

Пайплайн зависит только от протоколов этапов (`Expander`, `Reranker`, `Encoder`), а не от конкретных моделей:
multi-query, cross-encoder и LLM-реранкер подключаются без правки пайплайна. Каждый запрос возвращает
`Trace` — выдачу, время каждого этапа и токены LLM, — из которого строится профиль задержки.

Время LLM-этапов — сохранённое время ответа API при первом запросе (`ChatResponse.seconds`): повторное
исполнение из кэша сообщает реальную задержку, а не время чтения файла. Остальные этапы меряются заново.
"""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from gigarag import fusion
from gigarag.cache import timed
from gigarag.retrieval import Bm25Index, Ranking
from gigarag.search import masked_top_k

STAGES = ("transform", "embed", "dense", "bm25", "fusion", "rerank")
HYBRID, DENSE = "hybrid", "dense"
P95 = 95


@dataclass(frozen=True)
class HybridRetriever:
    """Dense по нормированным эмбеддингам чанков + BM25 и их взвешенная сумма; маска фильтра метаданных
    применяется к полному массиву score каждого поисковика до отбора кандидатов (pre-filter)."""
    vectors: np.ndarray
    bm25: Bm25Index
    alpha: float
    pool: int

    def dense(self, query_vector: np.ndarray, allowed: np.ndarray | None) -> Ranking:
        return masked_top_k(self.vectors @ query_vector, allowed, self.pool)

    def sparse(self, query: str, allowed: np.ndarray | None) -> Ranking:
        return masked_top_k(self.bm25.scores(query), allowed, self.pool)

    def fuse(self, dense: Ranking, sparse: Ranking) -> Ranking:
        return fusion.weighted_fusion(dense, sparse, self.alpha, self.pool)


@dataclass(frozen=True)
class LlmCall:
    """Время и токены одного этапа с LLM (на этапе может быть несколько параллельных запросов)."""
    seconds: float
    prompt_tokens: int
    completion_tokens: int


@dataclass(frozen=True)
class Expansion:
    """Запросы для поиска: исходный первым, затем сгенерированные."""
    queries: tuple[str, ...]
    call: LlmCall


@dataclass(frozen=True)
class Reranked:
    ranking: Ranking
    seconds: float
    call: LlmCall | None = None  # только у LLM-реранкера


Encoder = Callable[[Sequence[str]], np.ndarray]  # тексты запросов → нормированные эмбеддинги
Expander = Callable[[str], Expansion]
Reranker = Callable[[str, Ranking], Reranked]   # (исходный запрос, кандидаты) → новая выдача


@dataclass(frozen=True)
class Trace:
    ranking: Ranking
    seconds: Mapping[str, float]          # этап → с
    prompt_tokens: int = 0
    completion_tokens: int = 0
    queries: tuple[str, ...] = ()

    @property
    def total_seconds(self) -> float:
        """End-to-end: этапы идут последовательно, поэтому задержка запроса — сумма этапов."""
        return sum(self.seconds.values())

    @property
    def tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


@dataclass(frozen=True)
class Pipeline:
    retriever: HybridRetriever
    encode: Encoder
    mode: str = HYBRID                    # HYBRID или DENSE (без BM25)
    expand: Expander | None = None
    rerank: Reranker | None = None

    def run(self, query: str, allowed: np.ndarray | None) -> Trace:
        """Выдача одного запроса и время каждого этапа. Фильтр (`allowed`) задаётся исходным запросом и общий
        для всех его переформулировок: LLM может опустить или выдумать номер патча."""
        expansion = self.expand(query) if self.expand else Expansion((query,), LlmCall(0.0, 0, 0))
        queries = expansion.queries
        vectors, embed_seconds = timed(lambda: self.encode(queries))
        dense, dense_seconds = timed(lambda: [self.retriever.dense(v, allowed) for v in vectors])
        seconds = {"embed": embed_seconds, "dense": dense_seconds}
        if self.expand:
            seconds["transform"] = expansion.call.seconds
        rankings = dense
        if self.mode == HYBRID:
            sparse, seconds["bm25"] = timed(lambda: [self.retriever.sparse(q, allowed) for q in queries])
            rankings, seconds["fusion"] = timed(
                lambda: [self.retriever.fuse(d, s) for d, s in zip(dense, sparse, strict=True)])
        ranking, rrf_seconds = timed(lambda: rankings[0] if len(rankings) == 1
                                     else fusion.rrf_fusion(rankings, self.retriever.pool))
        seconds["fusion"] = seconds.get("fusion", 0.0) + rrf_seconds
        calls = [expansion.call]
        if self.rerank:
            reranked = self.rerank(query, ranking)
            ranking, seconds["rerank"] = reranked.ranking, reranked.seconds
            calls += [reranked.call] if reranked.call else []
        return Trace(ranking, seconds, sum(c.prompt_tokens for c in calls), sum(c.completion_tokens for c in calls),
                     queries)


def run_all(pipeline: Pipeline, queries: Mapping[str, str], masks: Mapping[str, np.ndarray]) -> dict[str, Trace]:
    """Трассы всех запросов по одному, последовательно: так запрос приходит в проде, и этапы на GPU
    не делят устройство с соседними запросами."""
    return {key: pipeline.run(query, masks.get(key)) for key, query in queries.items()}


def percentile_95(values: Sequence[float]) -> float:
    """p95 — значение, которое не превышают 95% запросов (линейная интерполяция numpy)."""
    return float(np.percentile(values, P95))


def stage_profile(traces: Sequence[Trace]) -> pd.DataFrame:
    """Медиана и p95 каждого этапа и всего запроса, мс; строки — этапы в порядке пайплайна и «итого»."""
    present = [s for s in STAGES if any(s in t.seconds for t in traces)]
    columns = {stage: [t.seconds.get(stage, 0.0) * 1000 for t in traces] for stage in present}
    columns["итого"] = [t.total_seconds * 1000 for t in traces]
    return pd.DataFrame({"медиана, мс": {s: float(np.median(v)) for s, v in columns.items()},
                         "p95, мс": {s: percentile_95(v) for s, v in columns.items()}})


def bottleneck(profile: pd.DataFrame) -> str:
    """Этап с наибольшей медианной задержкой."""
    return str(profile.drop(index="итого")["медиана, мс"].idxmax())
