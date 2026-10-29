import json
import math

import pytest
from sklearn.metrics import cohen_kappa_score

from gigarag import generation, judge
from gigarag.patterns import is_abstention
from gigarag.stats import weighted_kappa

SCALE = (1, 2, 3, 4, 5)


# ---------- weighted kappa ----------

@pytest.mark.parametrize("first, second", [
    ([1, 2, 3, 4, 5, 5, 4], [1, 2, 3, 4, 5, 4, 5]),
    ([5, 5, 4, 3, 1, 2], [4, 5, 4, 2, 2, 1]),
    ([1, 5, 1, 5], [5, 1, 5, 1]),
])
def test_weighted_kappa_matches_sklearn(first, second):
    expected = cohen_kappa_score(first, second, weights="quadratic", labels=list(SCALE))
    assert weighted_kappa(first, second, SCALE) == pytest.approx(expected)


def test_weighted_kappa_edge_cases():
    assert weighted_kappa([1, 2, 5], [1, 2, 5], SCALE) == pytest.approx(1.0)
    assert math.isnan(weighted_kappa([5, 5], [5, 5], SCALE))  # оба ставят одно и то же — κ не определена
    # далёкое расхождение штрафуется сильнее близкого
    assert weighted_kappa([1, 2, 4, 5], [1, 2, 5, 5], SCALE) > weighted_kappa([1, 2, 4, 5], [1, 2, 1, 5], SCALE)


# ---------- генерация ----------

def test_answer_request_numbers_context_and_disables_thinking():
    request = generation.answer_request("вопрос?", ["первый", "второй"])
    user = request.messages[1][1]
    assert "[1]\nпервый" in user and "[2]\nвторой" in user and user.startswith("Вопрос: вопрос?")
    assert request.payload()["thinking"] == {"type": "disabled"} and request.temperature == 0


def test_prompt_abstention_phrase_is_recognized():
    # фраза отказа из промпта генерации должна распознаваться без LLM
    phrase = generation.SYSTEM.split("ответь ровно «")[1].split("»")[0]
    assert is_abstention(phrase)


# ---------- судья ----------

def test_judge_prompts_cover_all_criteria_and_scale():
    for version, prompts in judge.PROMPTS.items():
        assert set(prompts) == set(judge.CRITERIA), version
        for text in prompts.values():
            assert all(f"{score} —" in text for score in SCALE)


def test_judge_request_is_one_criterion_per_call():
    requests = [judge.judge_request(c, "в?", ["ф"], "о", 1) for c in judge.CRITERIA]
    assert len({r.key() for r in requests}) == 3
    assert all(r.json_mode and r.payload()["thinking"] == {"type": "disabled"} for r in requests)


@pytest.mark.parametrize("content, score", [
    (json.dumps({"rationale": "ок", "score": 4}), 4),
    (json.dumps({"score": 6}), None),
    (json.dumps({"score": True}), None),
    (json.dumps({"score": "5"}), None),
    ("[]", None),
    ("не JSON", None),
])
def test_parse_verdict(content, score):
    assert judge.parse_verdict(content).score == score


def test_judge_answers_one_request_per_criterion_and_case():
    from gigarag.llm import ChatResponse

    class FakeChat:
        def __init__(self):
            self.systems = []

        def complete(self, request):
            self.systems.append(request.messages[0][1])
            return ChatResponse(json.dumps({"score": 4, "rationale": "ок"}), 1, 1)

    chat = FakeChat()
    verdicts = judge.judge_answers(chat, {"a": ("в?", ["ф"], "о"), "b": ("в2?", ["ф"], "о")}, 3)
    assert len(chat.systems) == 6 and len(set(chat.systems)) == 3
    assert verdicts[("relevance", "b")] == judge.Verdict(4, "ок")


def test_v3_verdict_keeps_structured_fields_as_rationale():
    verdict = judge.parse_verdict(json.dumps({"unsupported": [], "minor": ["ссылка"], "score": 4}, ensure_ascii=False))
    assert verdict.score == 4 and "ссылка" in verdict.rationale


# ---------- искажения для контроля судьи достоверности ----------

def test_value_spans_skip_patches_and_citations():
    from gigarag.patterns import value_spans
    text = "В патче 7.36b урон уменьшен с 6% до 4,5% [1], стоимость 4000 [2][3]. Урон 75/95 до 250."
    assert [text[a:b] for a, b in value_spans(text)] == ["6", "4,5", "4000", "75", "95", "250"]


def test_swap_value_puts_value_absent_from_context():
    import random

    from gigarag import perturbation
    passages = ["урон с 75 до 85", "стоимость 151"]
    answer = "Урон увеличен с 75 до 85 [1]."
    swapped = perturbation.swap_value(answer, passages, random.Random(0))
    assert swapped != answer and swapped.count("[1]") == 1
    new_values = set(perturbation.context_values([swapped])) - set(perturbation.context_values([answer]))
    assert len(new_values) == 1 and not new_values & perturbation.context_values(passages)
    assert perturbation.swap_value("Не знаю.", passages, random.Random(0)) is None


def test_add_invented_claim_is_deterministic_and_unsupported():
    import random

    from gigarag import perturbation
    passages = [" ".join(str(n) for n in range(13, 90))]  # почти все значения заняты
    answer = "Первый абзац [1].\n\nВторой абзац."
    claimed = perturbation.add_invented_claim(answer, passages, random.Random(1))
    assert claimed == perturbation.add_invented_claim(answer, passages, random.Random(1))
    assert claimed.endswith("\n\nВторой абзац.") and claimed.startswith("Первый абзац [1]. ")
    added = set(perturbation.context_values([claimed])) - set(perturbation.context_values([answer]))
    assert added and all(int(v) >= 90 for v in added)


def test_openai_chat_keeps_model_identity():
    import httpx

    from gigarag.llm import OpenAIChat
    body = {"id": "resp-1", "model": "deepseek-flash", "system_fingerprint": "fp-1",
            "choices": [{"message": {"content": "ок"}}], "usage": {"prompt_tokens": 3, "completion_tokens": 1}}
    client = OpenAIChat("key", "https://example.test")
    client._client = httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(200, json=body)))
    response = client.complete(generation.answer_request("в?", ["ф"]))
    assert (response.model, response.fingerprint, response.response_id) == ("deepseek-flash", "fp-1", "resp-1")


def test_within_tolerance_share():
    from gigarag.stats import within_tolerance
    assert within_tolerance([5, 4, 1, 3], [5, 5, 3, 3], 0) == 0.5
    assert within_tolerance([5, 4, 1, 3], [5, 5, 3, 3], 1) == 0.75
    assert within_tolerance([], [], 0) == 1.0


def test_detection_rates_and_kappa():
    from gigarag import perturbation
    scored = [("исходный", 5), ("исходный", 2), ("замена", 1), ("замена", 4), ("факт", 3), ("факт", None)]
    result = perturbation.detection(scored, original="исходный")
    assert result.rates == {"исходный": 0.5, "замена": 0.5, "факт": 0.5}
    perfect = perturbation.detection([("исходный", 5), ("замена", 3), ("факт", 1)], original="исходный")
    assert perfect.kappa == pytest.approx(1.0) and perfect.rates["исходный"] == 0


# ---------- разметка: оценка привязана к содержимому ответа ----------

RECORD = {"id": "best retrieval|q01", "question": "в?", "passages": ["ф1", "ф2"], "answer": "о"}


def test_content_digest_changes_with_any_visible_part():
    from gigarag import annotation
    base = annotation.content_digest(RECORD)
    for field, value in (("question", "в2?"), ("passages", ["ф2", "ф1"]), ("answer", "о2")):
        assert annotation.content_digest({**RECORD, field: value}) != base
    assert annotation.content_digest({**RECORD, "id": "другой"}) == base  # id не входит: только то, что видит человек


def test_label_for_changed_answer_is_not_valid():
    from gigarag import annotation
    labels = annotation.with_label({}, RECORD, {"faithfulness": 5, "relevance": 4, "utilization": 3})
    assert annotation.is_complete(labels, RECORD)
    changed = {**RECORD, "answer": "перегенерирован"}
    assert annotation.valid_labels(labels, [changed]) == {} and not annotation.is_complete(labels, changed)
    # новая оценка для изменённого текста не смешивается со старыми баллами
    relabeled = annotation.with_label(labels, changed, {"relevance": 1})
    assert annotation.valid_labels(relabeled, [changed]) == {RECORD["id"]: {"relevance": 1}}


def test_reuse_labels_only_for_unchanged_content():
    from gigarag import annotation
    old = annotation.with_label({}, RECORD, {"faithfulness": 5, "relevance": 5, "utilization": 5})
    same, changed = RECORD, {**RECORD, "id": "baseline retrieval|q02", "answer": "x"}
    other = annotation.with_label({}, {**changed, "answer": "y"}, {"relevance": 2})
    assert set(annotation.reuse_labels([old, other], [same, changed])) == {RECORD["id"]}


def test_loss_stage_splits_retrieval_and_generation():
    from gigarag.answer_eval import ANSWERED, GENERATION_LOSS, RETRIEVAL_LOSS, loss_stage
    assert loss_stage(0.0, abstained=False, faithful=True) == RETRIEVAL_LOSS
    assert loss_stage(0.0, abstained=True, faithful=True) == RETRIEVAL_LOSS  # отказ без контекста — вина поиска
    assert loss_stage(0.5, abstained=True, faithful=True) == GENERATION_LOSS
    assert loss_stage(1.0, abstained=False, faithful=False) == GENERATION_LOSS
    assert loss_stage(0.5, abstained=False, faithful=True) == ANSWERED
