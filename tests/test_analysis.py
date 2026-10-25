import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix

from gigarag import dedup, stats, topics
from gigarag.cache import CacheStore, cache_key, load_or_compute
from gigarag.clustering import deterministic_kmeans

# ---------- dedup ----------

def test_normalize_and_hash():
    assert dedup.normalize_for_dedup("  Патч  ВЫШЕЛ ") == "патч вышел"
    assert dedup.content_hash("a") == dedup.content_hash("a") != dedup.content_hash("b")


def test_shingles_and_jaccard():
    assert dedup.word_shingles(["a", "b", "c", "d"]) == {"a b c", "b c d"}
    sets = dedup.shingle_sets({"x": "a b c d e", "y": "a b c d f", "short": "a b"}, n=3, min_words=5)
    assert set(sets) == {"x", "y"}
    assert dedup.jaccard(sets["x"], sets["y"]) == pytest.approx(2 / 4)
    assert dedup.jaccard(frozenset(), frozenset()) == 0.0


def test_lsh_candidate_probability():
    assert dedup.lsh_candidate_probability(1.0, bands=9, rows=13) == 1.0
    assert dedup.lsh_candidate_probability(0.8, bands=25, rows=5) > 0.999
    assert dedup.lsh_candidate_probability(0.8, bands=9, rows=13) == pytest.approx(1 - (1 - 0.8 ** 13) ** 9)


def test_lsh_finds_near_duplicate_and_confirmation_filters():
    base = "valve выпустили патч и понерфили пуджа снова как всегда в пятницу вечером"
    texts = {"a": base, "b": base + " ура", "c": "совсем другой текст про турнир и команды из европы"}
    sh = dedup.shingle_sets(texts)
    run = dedup.lsh_candidates(dedup.minhash_signatures(sh, num_perm=128), threshold=0.5, num_perm=128)
    assert ("a", "b") in run.pairs
    assert dedup.confirmed_pairs(run.pairs, sh, 0.8) == [("a", "b", pytest.approx(10 / 11))]
    assert dedup.confirmed_pairs([("a", "c")], sh, 0.8) == []


# ---------- topics ----------

def test_class_tfidf_prefers_cluster_specific_terms():
    doc_term = csr_matrix(np.array([[2, 1, 0], [3, 1, 0], [0, 1, 4]]))
    counts = topics.cluster_term_counts(doc_term, np.array([0, 0, 1]), n_clusters=2)
    assert counts.tolist() == [[5, 2, 0], [0, 1, 4]]
    weights = topics.class_tfidf(counts)
    assert topics.top_terms(weights, np.array(["патч", "дота", "турнир"]), k=1) == [["патч"], ["турнир"]]


# ---------- stats ----------

def test_length_summary():
    df = pd.DataFrame({"src": ["a", "a", "b"], "len": [10, 100, 3000]})
    table = stats.length_summary(df, "len", by="src", short=50, long=2000)
    assert table.loc["a", "<50_%"] == 50.0
    assert table.loc["b", ">2000_%"] == 100.0
    assert table.loc["ВСЕГО", "count"] == 3
    assert stats.share_percent(pd.Series([1, 3])).tolist() == [25.0, 75.0]


# ---------- cache ----------

def test_cache_key_depends_on_data_and_params(tmp_path):
    assert cache_key("sha1", {"m": 1}) == cache_key("sha1", {"m": 1})
    assert cache_key("sha1", {"m": 1}) != cache_key("sha2", {"m": 1}) != cache_key("sha1", {"m": 2})
    store = CacheStore(tmp_path, "sha1")
    assert store.path("emb", "npy", m=1) != store.path("emb", "npy", m=2)


def test_load_or_compute_computes_once(tmp_path):
    calls = []

    def compute():
        calls.append(1)
        return "value"

    path = tmp_path / "x.txt"
    load, save = (lambda p: p.read_text()), (lambda v, p: p.write_text(v))
    assert load_or_compute(path, compute, load=load, save=save) == "value"
    assert load_or_compute(path, compute, load=load, save=save) == "value"
    assert calls == [1]


# ---------- clustering ----------

def test_deterministic_kmeans_is_repeatable():
    vectors = np.random.default_rng(0).normal(size=(300, 8))
    a = deterministic_kmeans(vectors, n_clusters=5, seed=1)
    b = deterministic_kmeans(vectors, n_clusters=5, seed=1)
    assert (a == b).all()
