"""Поиск по чанкам: индексируются «записи» (текст чанка, его резюме, вопросы к нему, дочерние чанки),
а выдаются чанки-владельцы записей.

Score чанка — максимум по его записям: чанк с тремя сгенерированными вопросами находится по лучшему
из них, дочерние чанки поднимают своего родителя. Фильтр по метаданным — маска допустимых чанков,
применяется до выбора топ-k (pre-filtering), поэтому выдача остаётся полной длины.
"""

from collections.abc import Callable, Mapping, Sequence

import numpy as np

from gigarag.retrieval import Bm25Index, Ranking, top_k

Scorer = Callable[[str], np.ndarray]  # запрос → score каждой записи индекса
DEPTH = 10                            # глубина выдачи: максимальное k в метриках


def chunk_scores(entry_scores: np.ndarray, owners: np.ndarray, n_chunks: int) -> np.ndarray:
    """Score чанка = max score его записей; у чанка без записей — −∞."""
    scores = np.full(n_chunks, -np.inf)
    np.maximum.at(scores, owners, entry_scores)
    return scores


def rank_chunks(entry_scores: np.ndarray, owners: np.ndarray, n_chunks: int, depth: int = DEPTH,
                allowed: np.ndarray | None = None) -> list[int]:
    """Индексы `depth` лучших чанков; `allowed` — булева маска чанков, прошедших фильтр."""
    scores = chunk_scores(entry_scores, owners, n_chunks)
    if allowed is not None:
        scores = np.where(allowed, scores, -np.inf)
    return [i for i, s in top_k(scores, depth) if np.isfinite(s)]


def search_all(scorer: Scorer, queries: Mapping[str, str], owners: Sequence[int], n_chunks: int,
               depth: int = DEPTH, allowed: Mapping[str, np.ndarray] | None = None) -> dict[str, list[int]]:
    """Выдача для каждого запроса (qid → индексы чанков); `allowed` — маска фильтра для каждого qid."""
    owner_array = np.asarray(owners)
    return {qid: rank_chunks(scorer(query), owner_array, n_chunks, depth, (allowed or {}).get(qid))
            for qid, query in queries.items()}


def bm25_scorer(texts: Sequence[str]) -> Scorer:
    index = Bm25Index.build(texts)
    return index.scores


def dense_scorer(entry_vectors: np.ndarray, query_vectors: Mapping[str, np.ndarray],
                 queries: Mapping[str, str]) -> Scorer:
    """Косинусная близость нормированных эмбеддингов; векторы запросов посчитаны заранее."""
    by_text = {queries[qid]: vector for qid, vector in query_vectors.items()}
    return lambda query: entry_vectors @ by_text[query]


def one_entry_per_chunk(texts: Sequence[str]) -> tuple[list[str], list[int]]:
    """Простейший индекс: запись = текст чанка."""
    return list(texts), list(range(len(texts)))


def masked_top_k(scores: np.ndarray, allowed: np.ndarray | None, k: int) -> Ranking:
    """Pre-filter: топ-k только среди чанков, прошедших фильтр, по полному массиву score — подходящий чанк
    не теряется, даже если без фильтра он был бы ниже k-го места."""
    if allowed is None:
        return top_k(scores, k)
    candidates = np.flatnonzero(allowed)
    return [(int(candidates[i]), s) for i, s in top_k(scores[candidates], k)] if len(candidates) else []


def filter_ranking(ranking: Ranking, allowed: np.ndarray | None) -> Ranking:
    """Выдача без чанков, не прошедших фильтр метаданных (маска по индексам чанков)."""
    return ranking if allowed is None else [(i, s) for i, s in ranking if allowed[i]]


def ranked_ids(ranking: Ranking) -> list[int]:
    return [i for i, _ in ranking]
