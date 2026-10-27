"""Слияние выдач разных поисковиков (hybrid): Reciprocal Rank Fusion и взвешенная сумма нормированных score.

Выдача — `Ranking`: список (индекс чанка, score) по убыванию score, обычно топ-N кандидатов поисковика.
"""

from collections import defaultdict
from collections.abc import Sequence

from gigarag.retrieval import Ranking

RRF_K = 60  # константа из статьи Cormack et al. (2009): сглаживает вклад первых мест, подбора не требует


def rrf_fusion(rankings: Sequence[Ranking], depth: int, k: int = RRF_K) -> Ranking:
    """RRF: score(d) = Σ_i 1 / (k + rank_i(d)), rank с единицы; документ вне выдачи поисковика вклада не даёт.
    Использует только места, поэтому шкалы score поисковиков (BM25 и косинус) сравнивать не нужно."""
    fused: dict[int, float] = defaultdict(float)
    for ranking in rankings:
        for rank, (doc, _) in enumerate(ranking, 1):
            fused[doc] += 1 / (k + rank)
    return _top(fused, depth)


def min_max(ranking: Ranking) -> dict[int, float]:
    """Score выдачи в [0, 1]: (s − min) / (max − min); при равных score — все 1."""
    if not ranking:
        return {}
    scores = [s for _, s in ranking]
    low, high = min(scores), max(scores)
    return {doc: (s - low) / (high - low) if high > low else 1.0 for doc, s in ranking}


def weighted_fusion(dense: Ranking, sparse: Ranking, alpha: float, depth: int) -> Ranking:
    """Взвешенная сумма: alpha · dense' + (1 − alpha) · sparse', где ' — min-max внутри своей выдачи.
    Кандидат, которого один поисковик не вернул, получает от него 0 — как худший из его кандидатов."""
    dense_norm, sparse_norm = min_max(dense), min_max(sparse)
    fused = {doc: alpha * dense_norm.get(doc, 0.0) + (1 - alpha) * sparse_norm.get(doc, 0.0)
             for doc in dense_norm.keys() | sparse_norm.keys()}
    return _top(fused, depth)


def _top(scores: dict[int, float], depth: int) -> Ranking:
    """Лучшие `depth` по score; при равенстве — меньший индекс (детерминированно)."""
    return sorted(scores.items(), key=lambda pair: (-pair[1], pair[0]))[:depth]
