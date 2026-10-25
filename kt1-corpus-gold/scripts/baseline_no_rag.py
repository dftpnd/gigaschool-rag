"""Baseline без RAG: модель отвечает на вопросы validation-части золотого набора без контекста,
LLM-судья сравнивает ответ с эталоном. Отвечает на вопрос «нужен ли нам RAG?».

    python baseline_no_rag.py

Результаты: gold/baseline_no_rag.jsonl (ответы и вердикты), сводка печатается и строится в ноутбуке gold.
Ответы LLM кэшируются (data/llm_cache).
"""

import json
from dataclasses import asdict

from build_gold import GOLD_DIR, llm, report_usage

from gigarag.answer_eval import AnswerVerdict, summary_by_type
from gigarag.atomic import atomic_write_text
from gigarag.gold import GoldItem, load_gold
from gigarag.llm import chat_request, parse_json_object
from gigarag.parallel import parallel_map

ANSWER_MODELS = ("gpt-4.1-mini-2025-04-14", "gpt-4.1-2025-04-14")  # первая — кандидат на генератор в КТ 5
ANSWER_TEMPERATURE = 0.0
JUDGE_MODEL = "gpt-4.1-2025-04-14"
JUDGE_TEMPERATURE = 0.0
LLM_SEED = 42
SPLIT = "validation"
RESULTS = GOLD_DIR / "baseline_no_rag.jsonl"

ANSWER_SYSTEM = """Ты — помощник для игроков Dota 2. Отвечай на вопрос по-русски кратко и по существу.
Если ты не знаешь ответа точно или вопрос касается событий, о которых у тебя нет надёжных сведений,
прямо скажи «Не знаю» вместо догадки."""

JUDGE_SYSTEM = """Ты проверяешь ответ системы на вопрос по Dota 2, сравнивая его с эталонным ответом.
Вердикты:
- "correct" — ответ по существу совпадает с эталоном (числа, названия, вывод); для негативного вопроса
  (эталон: «в корпусе нет информации») — система признала, что не знает, или указала, что такого нет;
- "partial" — верна только часть ответа или ответ слишком общий, но не противоречит эталону;
- "incorrect" — ответ противоречит эталону или выдуман; для негативного вопроса — любой содержательный ответ;
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
    chat = llm()
    answers = parallel_map(lambda i: chat.complete(answer_request(model, i)), items)
    verdicts = parallel_map(lambda pair: chat.complete(judge_request(*pair)),
                            [(i, a.content) for i, a in zip(items, answers, strict=True)])
    report_usage(f"{model}: ответы", answers)
    report_usage(f"{model}: судья", verdicts)
    return [AnswerVerdict(i.qid, i.qtype, model, a.content, **parse_json_object(v.content))
            for i, a, v in zip(items, answers, verdicts, strict=True)]


def main() -> None:
    items = [i for i in load_gold(GOLD_DIR / "gold.jsonl") if i.split == SPLIT]
    results = [v for model in ANSWER_MODELS for v in evaluate(model, items)]
    atomic_write_text(RESULTS, "".join(json.dumps(asdict(v), ensure_ascii=False) + "\n" for v in results))
    print(summary_by_type(results).to_string())


if __name__ == "__main__":
    main()
