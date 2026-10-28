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
