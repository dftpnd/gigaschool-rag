"""Преобразования текста документа."""

import html

from gigarag.patterns import collapse_whitespace, strip_tags, word_tokens

MIN_CHECKED_WORD = 5  # короче — служебные слова и союзы, их отсутствие в доказательствах ничего не значит
STEM_CHARS = 5        # грубая основа слова: «изменения» и «изменение» совпадают, «нулифаер» и «нулик» — нет


def visible_text(raw: str) -> str:
    """То, что видит читатель: теги → пробел, HTML-сущности декодированы, пробелы схлопнуты."""
    return collapse_whitespace(html.unescape(strip_tags(raw)))


def plain_lower(raw: str) -> str:
    """Текст без тегов и сущностей в нижнем регистре — для поиска ключевых слов."""
    return html.unescape(strip_tags(raw)).lower()


def latin_letter_share(text: str) -> float:
    """Доля латинских букв среди всех букв текста; 0 для текста без букв."""
    letters = [ch for ch in text if ch.isalpha()]
    return sum(ch.isascii() for ch in letters) / len(letters) if letters else 0.0


def is_mostly_latin(text: str, threshold: float = 0.5) -> bool:
    return latin_letter_share(text) > threshold


def contains_quote(text: str, quote: str) -> bool:
    """Цитата дословно есть в тексте — с точностью до регистра и пробелов."""
    return bool(quote.strip()) and _normalized(quote) in _normalized(text)


def _normalized(text: str) -> str:
    return collapse_whitespace(text).lower()


def unsupported_words(answer: str, evidence_texts: list[str]) -> list[str]:
    """Слова ответа (от MIN_CHECKED_WORD букв), основы которых (первые STEM_CHARS букв) нет ни в одном
    доказательстве, — подсказка ревьюеру: ответ может приписывать источнику то, чего в нём нет
    (например, раскрыть сленговое «нулик» не тем предметом)."""
    stems = {w[:STEM_CHARS] for text in evidence_texts for w in word_tokens(text.lower())}
    words = (w for w in word_tokens(answer.lower()) if len(w) >= MIN_CHECKED_WORD and not w.isdigit())
    return list(dict.fromkeys(w for w in words if w[:STEM_CHARS] not in stems))
