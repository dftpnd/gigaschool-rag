"""LLM как реранкер: listwise (вся пачка кандидатов одним запросом) и pointwise (оценка 0–2 каждого кандидата).

Кандидаты подаются с префиксом метаданных (заголовок, тип, патч, дата): номер патча часто есть только в нём.
Ответ — JSON. Ошибки модели не роняют выдачу: в listwise пропущенные и выдуманные номера чинятся (недостающие
кандидаты дописываются в исходном порядке), в pointwise нераспознанная оценка считается нулём.
"""

import json
from collections.abc import Sequence
from dataclasses import dataclass

from gigarag.enrich import MODEL, SEED, TEMPERATURE
from gigarag.llm import CachedChat, ChatRequest, ChatResponse, chat_request, parse_json_object
from gigarag.parallel import parallel_map
from gigarag.pipeline import LlmCall, Reranked
from gigarag.rerank import reorder
from gigarag.retrieval import Ranking

PROMPT_VERSION = 1
GRADES = (0, 1, 2)

_TASK = """Ты оцениваешь фрагменты из базы материалов о балансе Dota 2 (патчноуты, статьи и комментарии dota2.ru,
посты Telegram) для вопроса игрока. Фрагмент полезен, если содержит ответ на вопрос или его необходимую часть:
конкретное изменение нужной сущности в нужном патче, мнение или вывод, о котором спрашивают. Правка той же
сущности в другом патче или общий текст на ту же тему ответом не являются."""

LISTWISE_SYSTEM = f"""{_TASK}

Тебе дадут вопрос и пронумерованные фрагменты. Упорядочь все фрагменты от самого полезного к наименее полезному.
Ответ — JSON: {{"ranking": [номера фрагментов]}} — каждый номер ровно один раз. (версия промпта {PROMPT_VERSION})"""

POINTWISE_SYSTEM = f"""{_TASK}

Тебе дадут вопрос и один фрагмент. Оцени его:
2 — фрагмент содержит ответ на вопрос или его необходимую часть;
1 — фрагмент по теме вопроса (та же сущность, тот же патч или обсуждение), но ответа не содержит;
0 — фрагмент не по теме.
Ответ — JSON: {{"grade": 0, 1 или 2}} (версия промпта {PROMPT_VERSION})"""


def listwise_user(question: str, passages: Sequence[str]) -> str:
    numbered = "\n\n".join(f"[{n}]\n{text}" for n, text in enumerate(passages, 1))
    return f"Вопрос: {question}\n\nФрагменты:\n\n{numbered}"


def listwise_request(question: str, passages: Sequence[str]) -> ChatRequest:
    return chat_request(MODEL, LISTWISE_SYSTEM, listwise_user(question, passages), TEMPERATURE, SEED, json_mode=True)


def pointwise_request(question: str, passage: str) -> ChatRequest:
    return chat_request(MODEL, POINTWISE_SYSTEM, f"Вопрос: {question}\n\nФрагмент:\n{passage}", TEMPERATURE, SEED,
                        json_mode=True)


@dataclass(frozen=True)
class ParsedOrder:
    order: list[int]  # позиции кандидатов (с нуля) от лучшего к худшему, все n
    valid: bool       # модель вернула перестановку всех номеров без ошибок


def parse_order(content: str, n: int) -> ParsedOrder:
    """Порядок из ответа listwise. Номера вне 1…n и повторы отбрасываются, пропущенные кандидаты
    дописываются в исходном порядке — первичный поиск для них остаётся лучшей оценкой."""
    try:
        raw = parse_json_object(content).get("ranking", [])
    except json.JSONDecodeError:
        raw = []
    raw = raw if isinstance(raw, list) else []
    named = list(dict.fromkeys(x - 1 for x in raw if isinstance(x, int) and not isinstance(x, bool) and 1 <= x <= n))
    order = named + [i for i in range(n) if i not in set(named)]
    return ParsedOrder(order, valid=len(raw) == len(named) == n)


def parse_grade(content: str) -> int | None:
    """Оценка 0–2 из ответа pointwise; None, если ответ не распознан."""
    try:
        grade = parse_json_object(content).get("grade")
    except json.JSONDecodeError:
        return None
    return grade if grade in GRADES and not isinstance(grade, bool) else None


def order_by_grades(grades: Sequence[int | None]) -> list[int]:
    """Позиции по убыванию оценки; при равной оценке — исходный порядок первичного поиска.
    Нераспознанная оценка — 0: кандидат уходит вниз, но не пропадает."""
    return sorted(range(len(grades)), key=lambda i: (-(grades[i] or 0), i))


def call_of(responses: Sequence[ChatResponse]) -> LlmCall:
    """Этап из параллельных запросов: ждём самый медленный, токены складываются."""
    return LlmCall(max((r.seconds for r in responses), default=0.0), sum(r.prompt_tokens for r in responses),
                   sum(r.completion_tokens for r in responses))


@dataclass(frozen=True)
class ListwiseLlmReranker:
    chat: CachedChat
    passages: Sequence[str]
    pool: int

    def __call__(self, query: str, ranking: Ranking) -> Reranked:
        head = ranking[:self.pool]
        response = self.chat.complete(listwise_request(query, [self.passages[doc] for doc, _ in head]))
        order = parse_order(response.content, len(head)).order
        scores = [len(order) - place for place in range(len(order))]  # чем выше место, тем больше score
        call = call_of([response])
        return Reranked(reorder(ranking, order, scores), call.seconds, call)


@dataclass(frozen=True)
class PointwiseLlmReranker:
    """N независимых запросов отправляются параллельно: задержка этапа — самый медленный из них."""
    chat: CachedChat
    passages: Sequence[str]
    pool: int

    def __call__(self, query: str, ranking: Ranking) -> Reranked:
        head = ranking[:self.pool]
        responses = parallel_map(self.chat.complete, [pointwise_request(query, self.passages[doc]) for doc, _ in head],
                                 workers=max(len(head), 1))
        grades = [parse_grade(r.content) for r in responses]
        order = order_by_grades(grades)
        call = call_of(responses)
        return Reranked(reorder(ranking, order, [grades[i] or 0 for i in order]), call.seconds, call)
