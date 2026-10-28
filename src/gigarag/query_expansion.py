"""Multi-query: LLM переписывает вопрос игрока в несколько поисковых запросов, каждый ищется отдельно,
выдачи сливаются RRF.

Вопросы gold — длинные, разговорные, со сленгом и опечатками («импетус у энчи», «прирсо рост силы»),
а патчноуты называют героев, предметы и способности официальными английскими именами. Переформулировки
приводят вопрос к языку корпуса (это помогает и BM25, и dense), а у составных вопросов — дают отдельный
запрос на каждую часть. Исходный вопрос всегда ищется тоже: переформулировка может потерять деталь.
"""


from gigarag.enrich import MODEL, SEED, TEMPERATURE
from gigarag.llm import CachedChat, ChatRequest, chat_request, json_list_field
from gigarag.pipeline import Expansion, LlmCall

PARAPHRASES = 3
PROMPT_VERSION = 1
THINKING = False  # рассуждение deepseek-flash — тысячи скрытых выходных токенов на запрос, а ответ — три строки

SYSTEM = f"""Ты переписываешь вопросы игроков Dota 2 в поисковые запросы к базе материалов о балансе игры:
официальные патчноуты (на русском, с английскими названиями героев, предметов и способностей), статьи
и комментарии dota2.ru, посты Telegram-каналов.

Дай {PARAPHRASES} разных запроса на русском:
1. Короткий нейтральный пересказ вопроса: исправь опечатки, замени сленг и сокращения официальными английскими
   названиями героев, предметов и способностей, как в патчноутах (энча → Enchantress, импетус → Impetus,
   войд → Faceless Void, таймлок → Time Lock).
2. Запрос в стиле строки патчноута: у какой сущности и какое изменение искать.
3. Если в вопросе несколько частей — запрос про часть, которую не покрывают первые два. Если часть одна —
   более общий вопрос о той же теме: как её обсуждали игроки и что писали в статьях.

Сохраняй номера патчей, имена и числа из вопроса. Не добавляй фактов, которых в вопросе нет: не угадывай
значения, патчи и названия, если вопрос их не называет.
Ответ — JSON: {{"queries": ["...", "...", "..."]}} (версия промпта {PROMPT_VERSION})"""


def expansion_request(question: str) -> ChatRequest:
    return chat_request(MODEL, SYSTEM, question, TEMPERATURE, SEED, json_mode=True,
                        thinking=THINKING)


def parse_queries(content: str) -> list[str]:
    """Переформулировки из ответа модели; пустой список, если ответ не по формату (тогда ищется только
    исходный вопрос)."""
    queries = json_list_field(content, "queries") or []
    return [q.strip() for q in queries if isinstance(q, str) and q.strip()]


def search_queries(question: str, paraphrases: list[str], limit: int = PARAPHRASES) -> tuple[str, ...]:
    """Исходный вопрос и до `limit` переформулировок без повторов (модель иногда повторяет вопрос дословно)."""
    return tuple(dict.fromkeys([question, *paraphrases[:limit]]))


def multi_query_expander(chat: CachedChat):
    """Этап трансформации для `Pipeline`: один запрос к LLM на вопрос."""
    def expand(question: str) -> Expansion:
        response = chat.complete(expansion_request(question))
        return Expansion(search_queries(question, parse_queries(response.content)),
                         LlmCall(response.seconds, response.prompt_tokens, response.completion_tokens))
    return expand
