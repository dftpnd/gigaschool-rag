import math

import numpy as np
import pytest

from gigarag import fusion, metrics
from gigarag.cache import array_fingerprint
from gigarag.gold import Evidence, GoldItem
from gigarag.profiling import traced_peak_mb
from gigarag.relevance import EvidenceChunks, chunk_grades
from gigarag.search import filter_ranking, ranked_ids
from gigarag.vectordb import IndexConfig

# ---------- слияние выдач ----------


def test_rrf_uses_only_ranks():
    dense = [(1, 0.9), (2, 0.8), (3, 0.1)]
    sparse = [(3, 40.0), (1, 30.0)]
    fused = fusion.rrf_fusion([dense, sparse], depth=3, k=60)
    assert ranked_ids(fused) == [1, 3, 2]
    assert fused[0][1] == pytest.approx(1 / 61 + 1 / 62)


def test_min_max_and_weighted_fusion():
    assert fusion.min_max([(1, 5.0), (2, 3.0), (3, 1.0)]) == {1: 1.0, 2: 0.5, 3: 0.0}
    assert fusion.min_max([(1, 2.0), (2, 2.0)]) == {1: 1.0, 2: 1.0}
    dense = [(1, 0.9), (2, 0.5)]
    sparse = [(2, 10.0), (3, 2.0)]
    assert ranked_ids(fusion.weighted_fusion(dense, sparse, alpha=1.0, depth=3))[:1] == [1]
    assert ranked_ids(fusion.weighted_fusion(dense, sparse, alpha=0.0, depth=3))[:1] == [2]
    # 2 — лучший в BM25 и худший кандидат dense: при alpha = 0,5 набирает 0,5, как 1 (лучший dense)
    assert dict(fusion.weighted_fusion(dense, sparse, alpha=0.5, depth=3)) == {1: 0.5, 2: 0.5, 3: 0.0}


def test_fusion_ties_are_deterministic():
    assert ranked_ids(fusion.rrf_fusion([[(5, 1.0)], [(2, 1.0)]], depth=2)) == [2, 5]


# ---------- метрики ----------


def evidence() -> list[EvidenceChunks]:
    return [EvidenceChunks("p1", 2, frozenset({3})), EvidenceChunks("p2", 1, frozenset({3, 7}))]


def test_chunk_grades_take_max():
    assert chunk_grades(evidence()) == {3: 2, 7: 1}


def test_ndcg_graded():
    # идеальная выдача [3, 7]: выигрыши 3 и 1
    ideal = 3 + 1 / math.log2(3)
    assert metrics.ndcg_at_k([3, 7], evidence(), 5) == pytest.approx(1.0)
    assert metrics.ndcg_at_k([7, 3], evidence(), 5) == pytest.approx((1 + 3 / math.log2(3)) / ideal)
    assert metrics.ndcg_at_k([9], evidence(), 5) == 0.0


def test_average_precision():
    ev = [EvidenceChunks("p1", 2, frozenset({1})), EvidenceChunks("p2", 2, frozenset({4}))]
    assert metrics.average_precision([1, 2, 4], ev, 10) == pytest.approx((1 / 1 + 2 / 3) / 2)
    assert metrics.average_precision([2], ev, 10) == 0.0


def test_overlap_at_k():
    assert metrics.overlap_at_k([1, 2, 3, 9], [3, 2, 1, 4], 3) == 1.0
    assert metrics.overlap_at_k([1, 9], [1, 2], 2) == 0.5


def test_bm25_empty_text_scores_zero_not_nan():
    from gigarag.retrieval import Bm25Index
    index = Bm25Index.build(["урон увеличен", ".....", "броня уменьшена"], k1=1.2, b=1.0)
    scores = index.scores("урон")
    assert not np.isnan(scores).any() and scores[1] == 0.0
    assert ranked_ids(index.search("урон", 1)) == [0]


def test_negative_near_share():
    items = [GoldItem("n1", "negative", "?", "", (Evidence("p", "d", 1, "q"),)),
             GoldItem("f1", "factual", "?", "", (Evidence("p", "d", 2, "q"),))]
    qrels = {"n1": [EvidenceChunks("p", 1, frozenset({4}))], "f1": [EvidenceChunks("p", 2, frozenset({4}))]}
    frame = metrics.negative_per_question({"n1": [4, 5, 6, 7, 8], "f1": [4]}, qrels, items)
    assert frame["qid"].tolist() == ["n1"] and frame["near@5"].tolist() == [0.2]


def test_per_question_has_ndcg_and_map():
    items = [GoldItem("q", "factual", "?", "", (Evidence("p", "d", 2, "x"),))]
    frame = metrics.per_question({"q": [3]}, {"q": evidence()}, items)
    assert {"NDCG@5", "MAP", "P@1"} <= set(frame.columns)
    assert frame.loc[0, "MAP"] == 1.0


# ---------- поиск, индекс, служебное ----------


def test_filter_ranking():
    ranking = [(0, 0.9), (1, 0.8), (2, 0.7)]
    assert filter_ranking(ranking, None) == ranking
    assert filter_ranking(ranking, np.array([True, False, True])) == [(0, 0.9), (2, 0.7)]


def test_index_config():
    assert IndexConfig("flat", hnsw_m=0).exact
    assert not IndexConfig("hnsw").exact
    assert IndexConfig("q", int8=True).params()["int8"] is True


def test_array_fingerprint_depends_on_content_and_shape():
    a = np.arange(6, dtype=np.float32)
    assert array_fingerprint(a) == array_fingerprint(a.copy())
    assert array_fingerprint(a) != array_fingerprint(a.reshape(2, 3))
    assert array_fingerprint(a) != array_fingerprint(a + 1)


def test_traced_peak_mb_sees_allocation():
    value, peak = traced_peak_mb(lambda: np.ones(2 ** 20, dtype=np.float64))
    assert value.size == 2 ** 20 and peak >= 7.5


def test_masked_top_k_is_true_prefilter():
    from gigarag.search import masked_top_k
    scores = np.array([0.9, 0.8, 0.7, 0.1])
    allowed = np.array([False, False, False, True])
    # без фильтра чанк 3 — последний, но pre-filter его находит
    assert masked_top_k(scores, allowed, 2) == [(3, pytest.approx(0.1))]
    assert ranked_ids(masked_top_k(scores, None, 2)) == [0, 1]
    assert masked_top_k(scores, np.zeros(4, dtype=bool), 2) == []


def test_query_latencies_includes_search():
    from gigarag.embeddings import query_latencies

    class FakeModel:
        def encode(self, texts, **_):
            return np.ones((len(texts), 2), dtype=np.float32)

    calls = []
    latencies = query_latencies(FakeModel(), ["a", "b"], search=calls.append)
    assert len(latencies) == 2 and len(calls) == 3  # прогрев + два запроса
