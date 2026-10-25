"""Простые поисковики для генерации кандидатов: BM25 и плотный поиск по эмбеддингам."""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from rank_bm25 import BM25Okapi

from gigarag.patterns import word_tokens

Ranking = list[tuple[int, float]]  # (индекс отрывка, score) по убыванию score


def bm25_tokens(text: str) -> list[str]:
    return word_tokens(text.lower())


def top_k(scores: np.ndarray, k: int) -> Ranking:
    """k индексов с наибольшим score, по убыванию."""
    k = min(k, len(scores))
    idx = np.argpartition(-scores, k - 1)[:k]
    return [(int(i), float(scores[i])) for i in idx[np.argsort(-scores[idx])]]


@dataclass(frozen=True)
class Bm25Index:
    bm25: BM25Okapi

    @classmethod
    def build(cls, texts: Sequence[str], k1: float = 1.5, b: float = 0.75) -> "Bm25Index":
        return cls(BM25Okapi([bm25_tokens(t) for t in texts], k1=k1, b=b))

    def search(self, query: str, k: int) -> Ranking:
        return top_k(self.bm25.get_scores(bm25_tokens(query)), k)


def dense_search(passage_vectors: np.ndarray, query_vector: np.ndarray, k: int) -> Ranking:
    """Плотный поиск по нормированным эмбеддингам: score — косинусная близость."""
    return top_k(passage_vectors @ query_vector, k)


def union_ids(rankings: Sequence[Ranking]) -> list[int]:
    """Объединение выдач без повторов, в порядке «лучшие места всех выдач сначала» (round-robin)."""
    seen: dict[int, None] = {}
    for rank in range(max(map(len, rankings), default=0)):
        for ranking in rankings:
            if rank < len(ranking):
                seen.setdefault(ranking[rank][0], None)
    return list(seen)
