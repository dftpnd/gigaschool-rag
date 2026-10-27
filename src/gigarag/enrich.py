"""LLM-обработка чанков: резюме и гипотетические вопросы (HyDE-style) для индексации вместо текста.

Каждый запрос — один чанк, температура 0 и фиксированный seed: результат воспроизводим и кэшируется
(`gigarag.llm.CachedChat`). Чанки короче `MIN_TOKENS` не обрабатываются: резюме реплики «+1» или
вопросы к подписи из трёх слов ничего не добавляют, а стоят столько же запросов. Такие чанки
индексируются своим текстом в обоих вариантах.
"""

from collections.abc import Sequence
from dataclasses import dataclass

from gigarag.llm import CachedChat, ChatRequest, ChatResponse, chat_request, parse_json_object
from gigarag.parallel import parallel_map

MODEL = "gpt-4.1-mini-2025-04-14"
TEMPERATURE = 0.0
SEED = 42
PROMPT_VERSION = 1
MIN_TOKENS = 30
QUESTIONS_PER_CHUNK = 3
# Цены OpenAI за 1 млн токенов (вход, выход) для gpt-4.1-mini на момент запуска — для оценки стоимости.
PRICE_PER_MILLION = (0.40, 1.60)

SUMMARY_SYSTEM = f"""Ты помогаешь строить поисковый индекс по материалам о балансе Dota 2 (патчноуты, статьи,
комментарии, посты Telegram). Перескажи фрагмент в 1–3 предложениях на русском. Сохрани главное: героев,
предметы, способности, номера патчей, числа изменений, мнения и выводы. Не добавляй ничего, чего нет
во фрагменте. Ответ — только пересказ, без вступлений. (версия промпта {PROMPT_VERSION})"""

QUESTIONS_SYSTEM = f"""Ты помогаешь строить поисковый индекс по материалам о балансе Dota 2 (патчноуты, статьи,
комментарии, посты Telegram). Придумай {QUESTIONS_PER_CHUNK} разных вопроса, которые мог бы задать игрок и на
которые этот фрагмент отвечает. Вопросы на русском, конкретные: называй героев, предметы, патчи, если они есть
во фрагменте. Не задавай вопросов, ответа на которые во фрагменте нет.
Ответ — JSON: {{"questions": ["...", "...", "..."]}} (версия промпта {PROMPT_VERSION})"""


def summary_request(text: str) -> ChatRequest:
    return chat_request(MODEL, SUMMARY_SYSTEM, text, TEMPERATURE, SEED)


def questions_request(text: str) -> ChatRequest:
    return chat_request(MODEL, QUESTIONS_SYSTEM, text, TEMPERATURE, SEED, json_mode=True)


def parse_questions(content: str) -> list[str]:
    questions = parse_json_object(content).get("questions", [])
    return [q.strip() for q in questions if isinstance(q, str) and q.strip()]


@dataclass(frozen=True)
class Usage:
    requests: int
    prompt_tokens: int
    completion_tokens: int
    cached: int

    @property
    def cost_usd(self) -> float:
        """Стоимость по ценам `PRICE_PER_MILLION`: вход × цена входа + выход × цена выхода."""
        price_in, price_out = PRICE_PER_MILLION
        return (self.prompt_tokens * price_in + self.completion_tokens * price_out) / 1e6


def usage(responses: Sequence[ChatResponse]) -> Usage:
    return Usage(len(responses), sum(r.prompt_tokens for r in responses),
                 sum(r.completion_tokens for r in responses), sum(r.cached for r in responses))


def eligible(token_counts: Sequence[int], min_tokens: int = MIN_TOKENS) -> list[int]:
    """Индексы чанков, которые отправляются в LLM."""
    return [i for i, n in enumerate(token_counts) if n >= min_tokens]


def complete_all(chat: CachedChat, requests: Sequence[ChatRequest], workers: int = 16) -> list[ChatResponse]:
    return parallel_map(chat.complete, requests, workers)
