import json

import pytest

from gigarag import gold_coverage as coverage
from gigarag import patterns
from gigarag.answer_eval import AnswerVerdict, abstention_verdict, load_verdicts, summary_by_type
from gigarag.gold import Evidence, GoldItem, review_digest
from gigarag.text import unsupported_words

SOURCES = {"patch:7.38:a": "patchnotes", "patch:7.41:b": "patchnotes", "article:1": "dota2ru_article",
           "tg:1": "tg_Dota2"}
PATCHES = {"patch:7.38:a": "7.38", "patch:7.41:b": "7.41", "article:1": "7.41", "tg:1": None}


@pytest.fixture
def items() -> list[GoldItem]:
    return [
        GoldItem("q01", "factual", "Урон Axe в 7.38?", "+5", split="validation", styles=("short",), evidence=(
            Evidence("patch:7.38:a#0", "patch:7.38:a", 2), Evidence("article:1#3", "article:1", 1))),
        GoldItem("q02", "composite", "Как менялся бкб в 7.38 и 7.41?", "...", split="test", evidence=(
            Evidence("patch:7.38:a#0", "patch:7.38:a", 2), Evidence("patch:7.41:b#0", "patch:7.41:b", 2),
            Evidence("tg:1#0", "tg:1", 2))),
        GoldItem("q03", "negative", "Что в 7.42?", "В корпусе нет информации для ответа.", split="validation",
                 evidence=(Evidence("patch:7.41:b#0", "patch:7.41:b", 1),)),
    ]


def test_question_frame_counts_relevant_and_related(items):
    q = coverage.question_frame(items).set_index("qid")
    assert q.loc["q01", ["relevant_docs", "related_docs", "style"]].tolist() == [1, 1, "short"]
    assert q.loc["q02", "relevant_docs"] == 3
    assert q.loc["q03", ["relevant_docs", "related_docs", "style"]].tolist() == [0, 1, coverage.NO_STYLE]


def test_split_balance_totals_and_test_share(items):
    table = coverage.split_balance(coverage.question_frame(items))
    assert table.loc["ВСЕГО", ["test", "validation"]].tolist() == [1, 2]
    assert table.loc["composite", "test_%"] == 100.0
    assert table.loc["abstract"].isna().all()  # типа нет в наборе — строка есть, но пустая


def test_source_coverage_counts_questions_not_passages(items):
    questions = coverage.question_frame(items)
    evidence = coverage.evidence_frame(items, SOURCES, PATCHES)
    table = coverage.source_coverage(evidence, questions)
    assert table.loc["composite", "patchnotes"] == 100.0  # два патчноута у одного вопроса — один вопрос
    assert "dota2ru_article" not in table.columns  # у статьи только оценка 1 — источник не даёт ответов
    assert (table.loc["negative"] == 0).all()
    per_question = coverage.sources_per_question(evidence, questions)
    assert per_question.to_dict() == {"q01": 1, "q02": 2, "q03": 0}


def test_patch_coverage_is_chronological_and_keeps_gaps(items):
    evidence = coverage.evidence_frame(items, SOURCES, PATCHES)
    table = coverage.patch_coverage(evidence, ["7.41", "7.9", "7.38"])
    assert table.index.tolist() == ["7.9", "7.38", "7.41"]
    assert table["questions"].tolist() == [0, 2, 1]


def test_review_frame_sorted_by_candidate():
    records = {c: {"qtype": "factual", "meta": {"kind": "patch"}, "question": c} for c in ("c001", "c000")}
    decisions = {"c001": {"decision": "reject", "reason": "дубль"}, "c000": {"decision": "accept", "reason": ""}}
    frame = coverage.review_frame(decisions, records)
    assert frame["cid"].tolist() == ["c000", "c001"]
    assert frame.loc[1, ["decision", "seed_kind"]].tolist() == ["reject", "patch"]


def test_verdict_score_and_hallucination():
    assert AnswerVerdict("q", "factual", "m", "", "partial").score == 0.5
    assert AnswerVerdict("q", "factual", "m", "", "incorrect").hallucinated
    assert not AnswerVerdict("q", "factual", "m", "Не знаю", "abstained").hallucinated


def test_summary_by_type_shares_and_total(tmp_path):
    verdicts = [AnswerVerdict("q1", "factual", "m", "", "correct"),
                AnswerVerdict("q2", "factual", "m", "", "incorrect"),
                AnswerVerdict("q3", "negative", "m", "", "partial")]
    path = tmp_path / "v.jsonl"
    path.write_text("".join(json.dumps(v.__dict__, ensure_ascii=False) + "\n" for v in verdicts), encoding="utf-8")
    summary = summary_by_type(load_verdicts(path))
    assert summary.loc[("m", "factual"), ["correct", "incorrect", "abstained"]].tolist() == [50.0, 50.0, 0.0]
    assert summary.loc[("m", "ВСЕГО"), "n"] == 3
    assert summary.loc[("m", "ВСЕГО"), "score_%"] == 50.0


def test_abstention_and_error_metrics_are_separate():
    assert patterns.is_abstention("Не знаю.") and patterns.is_abstention("«не знаю», такого патча нет")
    assert not patterns.is_abstention("Я не знаю точно, но урон 50")
    assert abstention_verdict("negative") == "correct" and abstention_verdict("factual") == "abstained"
    wrong_negative = AnswerVerdict("q", "negative", "m", "Урон +5", "incorrect")
    assert wrong_negative.missed_abstention and not wrong_negative.hallucinated


def test_unsupported_words_flags_wrong_slang_expansion():
    evidence = ["Саппорты докупают нулик к арканам ради маны"]
    assert unsupported_words("Саппорты докупают нулифаер ради маны", evidence) == ["нулифаер"]
    assert unsupported_words("Саппорты докупили нулик", evidence) == []


def test_review_digest_tracks_reviewed_content():
    record = {"qtype": "factual", "question": "Q", "answer": "A", "seed_docs": ["d"], "labels": [], "cid": "c1"}
    assert review_digest(record) == review_digest({**record, "cid": "c2"})
    assert review_digest(record) != review_digest({**record, "answer": "B"})


def test_sentences_split_after_terminal_punctuation():
    assert patterns.sentences("Урон +5. Броня −1! Что дальше? Ок") == ["Урон +5.", "Броня −1!", "Что дальше?", "Ок"]


def test_patch_mentions():
    assert patterns.patch_mentions("в патчах 741 и 7.41F, потом 7.41f и 7.38") == ["7.41f", "7.38"]
    assert patterns.patch_mentions("версия 17.41 и 7.410") == []
