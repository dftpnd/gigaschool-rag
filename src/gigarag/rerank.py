"""Реранкинг кандидатов первичного поиска: общие операции над выдачей и cross-encoder.

Реранкер переупорядочивает первые `pool` кандидатов, остальные идут за ними в прежнем порядке со score −∞:
выдача остаётся полной длины, а порог по score (`metrics.threshold_curve`) отсекает и хвост, который
реранкер не видел.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from gigarag.cache import timed
from gigarag.pipeline import Reranked
from gigarag.retrieval import Ranking

UNSEEN = float("-inf")  # score кандидата за пределами пула реранкера


def rerank_by_scores(ranking: Ranking, scores: Sequence[float], pool: int) -> Ranking:
    """Первые `pool` кандидатов по убыванию `scores` (при равенстве — прежний порядок), затем остальные."""
    head = sorted(zip(ranking[:pool], scores, strict=True), key=lambda pair: -pair[1])
    return [(doc, float(score)) for (doc, _), score in head] + [(doc, UNSEEN) for doc, _ in ranking[pool:]]


def reorder(ranking: Ranking, order: Sequence[int], scores: Sequence[float]) -> Ranking:
    """Первые len(order) кандидатов в порядке `order` (позиции в `ranking`) со score `scores`, затем остальные."""
    return [(ranking[i][0], float(s)) for i, s in zip(order, scores, strict=True)] + \
        [(doc, UNSEEN) for doc, _ in ranking[len(order):]]


@dataclass(frozen=True)
class CrossEncoderSpec:
    name: str
    revision: str
    params_millions: float
    max_length: int = 512

    @property
    def short_name(self) -> str:
        return self.name.split("/")[-1]


def load_cross_encoder(spec: CrossEncoderSpec, *, half: bool = False, device: str | None = None):
    """Cross-encoder с закреплённой ревизией; `half` — fp16 на GPU (на CPU игнорируется)."""
    import torch
    from sentence_transformers import CrossEncoder

    from gigarag.embeddings import pick_device
    device = device or pick_device()
    dtype = torch.float16 if half and device == "cuda" else torch.float32
    return CrossEncoder(spec.name, revision=spec.revision, device=device, max_length=spec.max_length,
                        model_kwargs={"torch_dtype": dtype})


@dataclass(frozen=True)
class CrossEncoderReranker:
    """Score пары «запрос — чанк» одним проходом модели; все `pool` пар — одним батчем, как в проде."""
    model: object
    passages: Sequence[str]
    pool: int

    def scores(self, query: str, ranking: Ranking) -> np.ndarray:
        pairs = [(query, self.passages[doc]) for doc, _ in ranking[:self.pool]]
        return self.model.predict(pairs, batch_size=max(len(pairs), 1), show_progress_bar=False)

    def __call__(self, query: str, ranking: Ranking) -> Reranked:
        scores, seconds = timed(lambda: self.scores(query, ranking))
        return Reranked(rerank_by_scores(ranking, scores, self.pool), seconds)
