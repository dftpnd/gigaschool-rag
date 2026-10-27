import json
from dataclasses import dataclass

import numpy as np
import pytest

from gigarag import metrics
from gigarag.gold import GoldItem
from gigarag.llm import CachedChat, ChatResponse, chat_request
from gigarag.llm_rerank import (
    ListwiseLlmReranker,
    PointwiseLlmReranker,
    call_of,
    order_by_grades,
    parse_grade,
    parse_order,
)
from gigarag.pipeline import (
    DENSE,
    Expansion,
    HybridRetriever,
    LlmCall,
    Pipeline,
    Reranked,
    Trace,
    bottleneck,
    stage_profile,
)
from gigarag.query_expansion import parse_queries, search_queries
from gigarag.relevance import EvidenceChunks
from gigarag.rerank import UNSEEN, reorder, rerank_by_scores
from gigarag.retrieval import Bm25Index
from gigarag.search import ranked_ids
from gigarag.stats import agreement_rate

# ---------- операции над выдачей ----------


def test_rerank_by_scores_reorders_pool_and_keeps_tail():
    ranking = [(10, 0.9), (11, 0.8), (12, 0.7), (13, 0.6)]
    reranked = rerank_by_scores(ranking, [0.1, 0.5, 0.3], pool=3)
    assert ranked_ids(reranked) == [11, 12, 10, 13]
    assert reranked[0][1] == 0.5 and reranked[-1][1] == UNSEEN


def test_rerank_by_scores_ties_keep_original_order():
    assert ranked_ids(rerank_by_scores([(1, 0.0), (2, 0.0)], [0.5, 0.5], pool=2)) == [1, 2]


def test_reorder_keeps_unseen_tail():
    ranking = [(10, 0.9), (11, 0.8), (12, 0.7)]
    assert reorder(ranking, [1, 0], [2, 1]) == [(11, 2.0), (10, 1.0), (12, UNSEEN)]


# ---------- LLM-реранкер: разбор ответов ----------


def test_parse_order_valid_permutation():
    parsed = parse_order(json.dumps({"ranking": [3, 1, 2]}), 3)
    assert parsed.order == [2, 0, 1] and parsed.valid


def test_parse_order_repairs_model_mistakes():
    # повтор, номер вне диапазона, пропуск: пропущенный дописывается в исходном порядке
    parsed = parse_order(json.dumps({"ranking": [2, 2, 9, 1]}), 4)
    assert parsed.order == [1, 0, 2, 3] and not parsed.valid
    assert parse_order("не JSON", 3) == parse_order('{"ranking": "1,2,3"}', 3)
    assert parse_order("не JSON", 3).order == [0, 1, 2]
    assert not parse_order(json.dumps({"ranking": [True, 1, 2]}), 2).valid


def test_parse_grade_and_order_by_grades():
    assert parse_grade('{"grade": 2}') == 2
    assert parse_grade('{"grade": 3}') is None
    assert parse_grade('{"grade": true}') is None
    assert parse_grade("оценка 2") is None
    assert order_by_grades([1, None, 2, 1]) == [2, 0, 3, 1]


def test_call_of_parallel_requests_waits_for_slowest():
    responses = [ChatResponse("", 10, 2, seconds=0.5), ChatResponse("", 20, 3, seconds=1.5)]
    assert call_of(responses) == LlmCall(1.5, 30, 5)


@dataclass
class FakeChat:
    """Подмена CachedChat: отвечает функцией от запроса и считает вызовы."""
    answer: object
    calls: int = 0

    def complete(self, request):
        self.calls += 1
        return ChatResponse(self.answer(request), prompt_tokens=100, completion_tokens=5, seconds=0.2)


def test_listwise_reranker_reorders_pool_with_llm_tokens():
    chat = FakeChat(lambda _: json.dumps({"ranking": [2, 1]}))
    reranked = ListwiseLlmReranker(chat, ["a", "b", "c"], pool=2)("вопрос", [(0, 0.9), (1, 0.8), (2, 0.7)])
    assert ranked_ids(reranked.ranking) == [1, 0, 2]
    assert reranked.call == LlmCall(0.2, 100, 5) and chat.calls == 1


def test_pointwise_reranker_one_request_per_candidate():
    chat = FakeChat(lambda request: json.dumps({"grade": 2 if "текст b" in request.messages[1][1] else 0}))
    reranked = PointwiseLlmReranker(chat, ["текст a", "текст b"], pool=2)("вопрос", [(0, 0.9), (1, 0.8)])
    assert ranked_ids(reranked.ranking) == [1, 0]
    assert reranked.call == LlmCall(0.2, 200, 10) and chat.calls == 2


def test_cached_chat_returns_saved_api_seconds(tmp_path):
    class Client:
        def complete(self, request):
            return ChatResponse("ответ", 3, 1, seconds=2.5)

    request = chat_request("model", "system", "user", 0.0, 42)
    chat = CachedChat(Client(), tmp_path)
    assert chat.complete(request).seconds == 2.5
    again = chat.complete(request)
    assert again.cached and again.seconds == 2.5


# ---------- multi-query ----------


def test_parse_queries_and_search_queries():
    assert parse_queries('{"queries": [" a ", "", 3, "b"]}') == ["a", "b"]
    assert parse_queries("битый {") == []
    assert search_queries("вопрос", ["вопрос", "x", "y", "z", "w"], limit=3) == ("вопрос", "x", "y")


# ---------- пайплайн ----------


def tiny_retriever() -> HybridRetriever:
    vectors = np.eye(3, dtype=np.float32)
    bm25 = Bm25Index.build(["огонь лёд", "лёд вода", "камень"], k1=0.3, b=1.0)
    return HybridRetriever(vectors, bm25, alpha=0.5, pool=3)


def encode(texts):
    return np.array([[1.0, 0.0, 0.0] if "огонь" in t else [0.0, 1.0, 0.0] for t in texts], dtype=np.float32)


def test_pipeline_stages_and_filter():
    trace = Pipeline(tiny_retriever(), encode).run("огонь", None)
    assert ranked_ids(trace.ranking)[0] == 0
    assert set(trace.seconds) == {"embed", "dense", "bm25", "fusion"} and trace.tokens == 0
    assert trace.total_seconds == pytest.approx(sum(trace.seconds.values()))
    masked = Pipeline(tiny_retriever(), encode).run("огонь", np.array([False, True, True]))
    assert 0 not in ranked_ids(masked.ranking)


def test_pipeline_multi_query_and_rerank():
    expand = lambda q: Expansion((q, "вода"), LlmCall(1.0, 50, 10))  # noqa: E731
    rerank = lambda q, r: Reranked(r[::-1], 0.3, LlmCall(2.0, 70, 5))  # noqa: E731
    trace = Pipeline(tiny_retriever(), encode, mode=DENSE, expand=expand, rerank=rerank).run("огонь", None)
    assert trace.queries == ("огонь", "вода")
    assert "bm25" not in trace.seconds
    assert trace.seconds["transform"] == 1.0 and trace.seconds["rerank"] == 0.3
    assert (trace.prompt_tokens, trace.completion_tokens) == (120, 15)


def test_stage_profile_median_p95_and_bottleneck():
    traces = [Trace([], {"embed": 0.004, "rerank": r}) for r in (0.1, 0.2, 0.3)]
    profile = stage_profile(traces)
    assert list(profile.index) == ["embed", "rerank", "итого"]
    assert profile.loc["rerank", "медиана, мс"] == pytest.approx(200)
    assert profile.loc["итого", "p95, мс"] == pytest.approx(np.percentile([104, 204, 304], 95))
    assert bottleneck(profile) == "rerank"


# ---------- порог и согласие ----------


def test_threshold_curve():
    items = [GoldItem("q1", "factual", "?", "!", split="validation"),
             GoldItem("n1", "negative", "?", "", split="validation")]
    qrels = {"q1": [EvidenceChunks("p", 2, frozenset({1}))], "n1": []}
    rankings = {"q1": [(1, 0.9), (2, 0.4)], "n1": [(5, 0.3)]}
    curve = metrics.threshold_curve(rankings, qrels, items, [0.0, 0.5, 1.0]).set_index("порог")
    assert curve.loc[0.0, "precision"] == 0.5 and curve.loc[0.0, "recall"] == 1.0
    assert curve.loc[0.5, "precision"] == 1.0 and curve.loc[0.5, "негативные без контекста"] == 1.0
    assert curve.loc[1.0, "recall"] == 0.0 and curve.loc[1.0, "F1"] == 0.0


def test_agreement_rate():
    assert agreement_rate([1, 2, 2], [1, 2, 0]) == pytest.approx(2 / 3)
    assert agreement_rate([], []) == 1.0
