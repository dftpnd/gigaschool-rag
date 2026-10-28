"""LLM как реранкер: listwise (вся пачка кандидатов одним запросом) и pointwise (оценка 0–2 каждого кандидата).

Кандидаты подаются с префиксом метаданных (заголовок, тип, патч, дата): номер патча часто есть только в нём.
Ответ — JSON. Ошибки модели не роняют выдачу: пропущенные и выдуманные номера чинятся (недостающие
кандидаты дописываются в исходном порядке), нераспознанная оценка считается нулём.

Listwise-промпт прошёл три итерации (`LISTWISE_PROMPTS`): полная сортировка пула при t = 0 повторялась лишь
в 60% случаев — модель переставляет почти равные кандидаты. Оценки 0–2 всей пачке одним запросом с сортировкой
по оценке и исходным порядком при равенстве убирают эти перестановки.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from gigarag.enrich import MODEL, SEED, TEMPERATURE
from gigarag.llm import CachedChat, ChatRequest, ChatResponse, chat_request, json_list_field, json_object_or_empty
from gigarag.parallel import parallel_map
from gigarag.pipeline import LlmCall, Reranked
from gigarag.rerank import reorder
from gigarag.retrieval import Ranking

PROMPT_VERSION = 1
THINKING = False  # с рассуждением deepseek-flash тратит 3–4 тыс. скрытых токенов на ответ из десятка номеров
GRADES = (0, 1, 2)
TOP = 5  # столько чанков уходит в контекст генератора (КТ 5)

_TASK = """Ты оцениваешь фрагменты из базы материалов о балансе Dota 2 (патчноуты, статьи и комментарии dota2.ru,
посты Telegram) для вопроса игрока. Фрагмент полезен, если содержит ответ на вопрос или его необходимую часть:
конкретное изменение нужной сущности в нужном патче, мнение или вывод, о котором спрашивают. Правка той же
сущности в другом патче или общий текст на ту же тему ответом не являются."""

_GRADE_SCALE = """2 — фрагмент содержит ответ на вопрос или его необходимую часть;
1 — фрагмент по теме вопроса (та же сущность, тот же патч или обсуждение), но ответа не содержит;
0 — фрагмент не по теме."""

RANKING_SYSTEM = f"""{_TASK}

Тебе дадут вопрос и пронумерованные фрагменты. Упорядочь все фрагменты от самого полезного к наименее полезному.
Ответ — JSON: {{"ranking": [номера фрагментов]}} — каждый номер ровно один раз. (версия промпта {PROMPT_VERSION})"""

GRADES_SYSTEM = f"""{_TASK}

Тебе дадут вопрос и пронумерованные фрагменты. Оцени каждый фрагмент:
{_GRADE_SCALE}
Ответ — JSON: {{"grades": [оценка фрагмента 1, оценка фрагмента 2, …]}} — ровно по одной оценке на фрагмент,
в порядке номеров. (версия промпта 2)"""

TOP_SYSTEM = f"""{_TASK}

Тебе дадут вопрос и пронумерованные фрагменты. Выбери не больше {TOP} самых полезных фрагментов и упорядочь их
от самого полезного. Бесполезные фрагменты не включай: лучше меньше {TOP}, чем фрагмент без ответа.
Ответ — JSON: {{"top": [номера фрагментов]}} (версия промпта 3)"""

POINTWISE_SYSTEM = f"""{_TASK}

Тебе дадут вопрос и один фрагмент. Оцени его:
{_GRADE_SCALE}
Ответ — JSON: {{"grade": 0, 1 или 2}} (версия промпта {PROMPT_VERSION})"""


def listwise_user(question: str, passages: Sequence[str]) -> str:
    numbered = "\n\n".join(f"[{n}]\n{text}" for n, text in enumerate(passages, 1))
    return f"Вопрос: {question}\n\nФрагменты:\n\n{numbered}"


@dataclass(frozen=True)
class ParsedOrder:
    order: list[int]  # позиции кандидатов (с нуля) от лучшего к худшему, все n
    valid: bool       # модель ответила по формату без ошибок


def _positions(numbers: list, n: int) -> list[int]:
    """Номера фрагментов 1…n → позиции с нуля; чужие номера и повторы отбрасываются."""
    return list(dict.fromkeys(x - 1 for x in numbers if isinstance(x, int) and not isinstance(x, bool) and 1 <= x <= n))


def _with_rest(named: list[int], n: int) -> list[int]:
    """Не названные моделью кандидаты дописываются в исходном порядке — первичный поиск для них
    остаётся лучшей оценкой."""
    return named + [i for i in range(n) if i not in set(named)]


def parse_order(content: str, n: int) -> ParsedOrder:
    """Порядок из ответа «отсортируй все» (v1); valid — модель вернула перестановку всех номеров."""
    raw = json_list_field(content, "ranking") or []
    named = _positions(raw, n)
    return ParsedOrder(_with_rest(named, n), valid=len(raw) == len(named) == n)


def parse_listwise_grades(content: str, n: int) -> ParsedOrder:
    """Порядок из оценок 0–2 всей пачки (v2): по убыванию оценки, при равенстве — исходный порядок.
    Лишние оценки отбрасываются, недостающие и нераспознанные — нули."""
    raw = json_list_field(content, "grades") or []
    grades = [g if g in GRADES and not isinstance(g, bool) else None for g in raw[:n]]
    grades += [None] * (n - len(grades))
    return ParsedOrder(order_by_grades(grades), valid=len(raw) == n and None not in grades)


def parse_top(content: str, n: int) -> ParsedOrder:
    """Порядок из ответа «выбери лучшие» (v3): названные — впереди, остальные — в исходном порядке."""
    raw = json_list_field(content, "top")
    named = _positions(raw or [], n)
    return ParsedOrder(_with_rest(named, n), valid=raw is not None and len(raw) == len(named) <= TOP)


@dataclass(frozen=True)
class ListwisePrompt:
    """Версия listwise-промпта: системный промпт и разбор ответа в порядок кандидатов."""
    name: str
    system: str
    parse: Callable[[str, int], ParsedOrder]


RANKING_PROMPT = ListwisePrompt("v1: отсортировать все", RANKING_SYSTEM, parse_order)
GRADES_PROMPT = ListwisePrompt("v2: оценки 0–2 всей пачке", GRADES_SYSTEM, parse_listwise_grades)
TOP_PROMPT = ListwisePrompt(f"v3: выбрать до {TOP} лучших", TOP_SYSTEM, parse_top)
LISTWISE_PROMPTS = (RANKING_PROMPT, GRADES_PROMPT, TOP_PROMPT)


def listwise_request(question: str, passages: Sequence[str], prompt: ListwisePrompt) -> ChatRequest:
    return chat_request(MODEL, prompt.system, listwise_user(question, passages), TEMPERATURE, SEED, json_mode=True,
                        thinking=THINKING)


def pointwise_request(question: str, passage: str) -> ChatRequest:
    return chat_request(MODEL, POINTWISE_SYSTEM, f"Вопрос: {question}\n\nФрагмент:\n{passage}", TEMPERATURE, SEED,
                        json_mode=True, thinking=THINKING)


def parse_grade(content: str) -> int | None:
    """Оценка 0–2 из ответа pointwise; None, если ответ не распознан."""
    grade = json_object_or_empty(content).get("grade")
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
    prompt: ListwisePrompt

    def __call__(self, query: str, ranking: Ranking) -> Reranked:
        head = ranking[:self.pool]
        response = self.chat.complete(listwise_request(query, [self.passages[doc] for doc, _ in head], self.prompt))
        order = self.prompt.parse(response.content, len(head)).order
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
