"""Baseline без RAG: модель отвечает на вопросы validation-части золотого набора без контекста,
LLM-судья сравнивает ответ с эталоном. Отвечает на вопрос «нужен ли нам RAG?».

    python baseline_no_rag.py

Результаты: gold/baseline_no_rag.jsonl (ответы и вердикты), сводка печатается и строится в ноутбуке gold.
Ответы LLM кэшируются (data/llm_cache).
"""

import json
from dataclasses import asdict, replace

from build_gold import GOLD_DIR, llm, report_usage

from gigarag.answer_eval import AnswerVerdict, abstention_verdict, summary_by_type
from gigarag.atomic import atomic_write_text
from gigarag.gold import GoldItem, load_gold
from gigarag.llm import chat_request, parse_json_object
from gigarag.parallel import parallel_map
from gigarag.patterns import is_abstention

ANSWER_MODELS = ("gpt-4.1-mini-2025-04-14", "gpt-4.1-2025-04-14")  # первая — кандидат на генератор в КТ 5
ANSWER_TEMPERATURE = 0.0
JUDGE_MODEL = "gpt-4.1-2025-04-14"
JUDGE_TEMPERATURE = 0.0
LLM_SEED = 42
SPLIT = "validation"
RESULTS = GOLD_DIR / "baseline_no_rag.jsonl"
# Ручная валидация вердиктов судьи: "qid|model" → исправленный вердикт и причина. Судья систематически
# ставил partial ответам, где выдуманы конкретные правки, но общий вывод совпал с эталоном.
VERDICT_REVIEW = GOLD_DIR / "baseline_review.json"

# Модель получает те же границы, что и RAG-система в КТ 5: вне окна корпуса ответ должен быть отказом.
# Иначе верный по памяти ответ про патч 7.29 считался бы выдумкой, хотя модель не знала, что он вне задачи.
ANSWER_SYSTEM = """Ты — помощник для игроков Dota 2. Ты отвечаешь только на вопросы о балансе Dota 2
в патчах 7.34–7.41f (август 2023 — октябрь 2026). Отвечай по-русски кратко и по существу.
Если вопрос выходит за эти границы (другие патчи, другие игры, будущие события), если такого героя, предмета
или изменения не было или ты не знаешь ответа точно — ответь, начав со слов «Не знаю», без догадок."""

JUDGE_SYSTEM = """Ты проверяешь ответ системы на вопрос по Dota 2, сравнивая его с эталонным ответом.
Вердикты:
- "correct" — ответ по существу совпадает с эталоном (числа, названия, вывод); для негативного вопроса
  (эталон: «в корпусе нет информации») — система отказалась отвечать или только указала, что такого
  не было, не сообщая никаких конкретных фактов;
- "partial" — верна только часть ответа или ответ слишком общий, но не противоречит эталону;
- "incorrect" — ответ противоречит эталону или выдуман; для негативного вопроса — любой содержательный ответ,
  в том числе конкретные факты о патче вне границ или описание несуществующего героя, предмета, юнита;
- "abstained" — система отказалась отвечать («не знаю»), хотя у вопроса есть эталонный ответ.
Для абстрактных вопросов оценивай совпадение основных тезисов, а не формулировок.
Ответ — строго JSON: {"verdict": "correct"|"partial"|"incorrect"|"abstained", "rationale": str}"""


def judge_user(item: GoldItem, answer: str) -> str:
    return (f"Тип вопроса: {item.qtype}\nВопрос: {item.question}\nЭталонный ответ: {item.answer}\n\n"
            f"Ответ системы: {answer}")


def answer_request(model: str, item: GoldItem):
    return chat_request(model, ANSWER_SYSTEM, item.question, ANSWER_TEMPERATURE, LLM_SEED)


def judge_request(item: GoldItem, answer: str):
    return chat_request(JUDGE_MODEL, JUDGE_SYSTEM, judge_user(item, answer), JUDGE_TEMPERATURE, LLM_SEED, True)


def evaluate(model: str, items: list[GoldItem]) -> list[AnswerVerdict]:
    """Ответы модели и вердикты. Отказ («Не знаю…») распознаётся детерминированно и судье не отправляется:
    судья путал его с верным ответом на позитивных вопросах."""
    chat = llm()
    responses = parallel_map(lambda i: chat.complete(answer_request(model, i)), items)
    report_usage(f"{model}: ответы", responses)
    answers = [r.content for r in responses]
    to_judge = [(i, a) for i, a in zip(items, answers, strict=True) if not is_abstention(a)]
    judged = parallel_map(lambda pair: chat.complete(judge_request(*pair)), to_judge)
    report_usage(f"{model}: судья", judged)
    by_qid = {i.qid: parse_json_object(v.content) for (i, _), v in zip(to_judge, judged, strict=True)}
    return [AnswerVerdict(i.qid, i.qtype, model, a,
                          **by_qid.get(i.qid, {"verdict": abstention_verdict(i.qtype), "rationale": "отказ"}))
            for i, a in zip(items, answers, strict=True)]


def apply_review(verdict: AnswerVerdict, review: dict[str, dict]) -> AnswerVerdict:
    """Вердикт после ручной валидации: исправление ревьюера заменяет вердикт судьи, причина — в rationale."""
    fix = review.get(f"{verdict.qid}|{verdict.model}")
    if fix is None:
        return verdict
    return replace(verdict, verdict=fix["verdict"],
                   rationale=f"ревью (судья: {verdict.verdict}): {fix['reason']}")


def main() -> None:
    items = [i for i in load_gold(GOLD_DIR / "gold.jsonl") if i.split == SPLIT]
    review = json.loads(VERDICT_REVIEW.read_text(encoding="utf-8"))
    results = [apply_review(v, review) for model in ANSWER_MODELS for v in evaluate(model, items)]
    unknown = set(review) - {f"{v.qid}|{v.model}" for v in results}
    if unknown:
        raise ValueError(f"исправления ревью к несуществующим ответам: {sorted(unknown)}")
    atomic_write_text(RESULTS, "".join(json.dumps(asdict(v), ensure_ascii=False) + "\n" for v in results))
    print(summary_by_type(results).to_string())


if __name__ == "__main__":
    main()
