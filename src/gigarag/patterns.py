"""Все регулярные выражения проекта.

Шаблоны скомпилированы один раз и приватны: снаружи доступны только функции
с семантическими именами — так видно, *что* ищется, а не *как*.
"""

import re
from collections.abc import Callable, Iterable

Span = tuple[int, int]

_TAG = re.compile(r"<[^>]+>")
_ENTITY = re.compile(r"&(?:[a-zA-Z][a-zA-Z0-9]*|#\d+|#[xX][0-9a-fA-F]+);")
_URL = re.compile(r"https?://[^\s\"'<>]+|www\.[^\s\"'<>]+|t\.me/[^\s\"'<>]+")
_WHITESPACE = re.compile(r"\s+")
_ABSTENTION = re.compile(r"^\W*не знаю\b", re.IGNORECASE)
_SENTENCE_END = re.compile(r"(?<=[.!?…])\s+")
_LETTERS = re.compile(r"[^\W\d_]+")
_DIGITS = re.compile(r"\d+")
_PUNCTUATION = re.compile(r"[.,!?:;…\-–—«»\"'()\[\]]")
_SPECIAL = re.compile(r"[^\w\s]|_")
_WORD_TOKEN = re.compile(r"\w+")
_PATCH_VERSION = re.compile(r"7\.(\d+)([a-z]?)")
_PATCH_MENTION = re.compile(r"(?<![\d.])7\.(\d{2})([a-z]?)(?![\w.])", re.IGNORECASE)
_PATCH_IN_SLUG = re.compile(r"-7-(\d{2}[a-z]?)")
_NEWS_LINK = re.compile(r'href="(/news/\d+-[^"?#]+)"')
_DATETIME_ATTR = re.compile(r'<time datetime="([^"]+)"')
_ADS_MARKER = re.compile(r"промокод|ставк|казино|букмекер|фрибет|реклама|erid|бездеп", re.IGNORECASE)
_TOPIC_WORD = re.compile(r"(?u)\b[^\W\d_]{3,}\b")
# числовое значение: целое или с десятичной запятой; не часть номера патча (7.36b) и не ссылка на фрагмент [2]
_VALUE = re.compile(r"(?<![\d.,\[])\d+(?:,\d+)?(?!\.\d|\d|,\d|\])")
_NUMERIC_CHANGE = re.compile(r"\bс\s+[\d.,/%]+\s*%?\s+(?:до|на)\s+[\d.,/%]+")
_HAS_DIGIT = re.compile(r"\d")
_LINE_BREAK_TAG = re.compile(r"<br\s*/?>", re.IGNORECASE)
_BLOCK_END_TAG = re.compile(r"</(?:p|div|li|ul|ol|h[1-6]|tr|table|blockquote)>", re.IGNORECASE)
_INLINE_SPACE = re.compile(r"[^\S\n]+")
_EXTRA_BLANK_LINES = re.compile(r"\n{3,}")
_INVISIBLE_CHAR = re.compile("[\u00ad\u200b-\u200f\u2060\ufe0e\ufe0f\ufeff]")
# Знаки, которые несут смысл: пунктуация, а также % + / = < > № ° × в числах патчноутов («+10%», «1/2/3»).
_DECORATIVE_SYMBOL = re.compile(r"[^\w\s.,!?:;…\-–—«»\"'()\[\]%+/=<>№°×]|_")


def _spans(rx: re.Pattern, text: str) -> list[Span]:
    return [m.span() for m in rx.finditer(text)]


# ---------- HTML ----------

def strip_tags(text: str, replacement: str = " ") -> str:
    """Заменяет каждый HTML-тег на `replacement`."""
    return _TAG.sub(replacement, text)


def tag_spans(text: str) -> list[Span]:
    return _spans(_TAG, text)


def entity_spans(text: str) -> list[Span]:
    """Позиции HTML-сущностей: `&nbsp;`, `&#8212;`, `&#x2014;`."""
    return _spans(_ENTITY, text)


def has_html_entity(text: str) -> bool:
    return _ENTITY.search(text) is not None


def tags_to_line_breaks(html: str) -> str:
    """`<br>` и конец блочного тега (`</p>`, `</div>`, `</li>`…) → перевод строки:
    после снятия тегов абзацы и строки остаются, и чанкерам есть на что опираться."""
    return _BLOCK_END_TAG.sub("\n", _LINE_BREAK_TAG.sub("\n", html))


# ---------- ссылки ----------

def url_spans(text: str) -> list[Span]:
    """Позиции URL, в том числе внутри атрибутов `href`/`src`."""
    return _spans(_URL, text)


def has_url(text: str) -> bool:
    return _URL.search(text) is not None


def remove_urls(text: str) -> str:
    return _URL.sub(" ", text)


def news_links(html: str) -> list[str]:
    """Относительные ссылки на статьи dota2.ru (`/news/<id>-<slug>`) в порядке появления, без повторов."""
    return list(dict.fromkeys(_NEWS_LINK.findall(html)))


def page_numbers(html: str, link: str) -> set[int]:
    """Номера страниц `<link>?page=N`, на которые ссылается HTML."""
    return {int(n) for n in re.findall(re.escape(link) + r"\?page=(\d+)", html)}


def telegram_post_ids(html: str, channel: str) -> list[int]:
    """Номера постов канала из атрибутов `data-post="<канал>/<id>"`."""
    return [int(i) for i in re.findall(rf'data-post="{re.escape(channel)}/(\d+)"', html, re.IGNORECASE)]


def datetime_attrs(html: str) -> list[str]:
    """Значения `<time datetime="...">` в порядке появления."""
    return _DATETIME_ATTR.findall(html)


# ---------- классы символов ----------

def letter_spans(text: str) -> list[Span]:
    return _spans(_LETTERS, text)


def digit_spans(text: str) -> list[Span]:
    return _spans(_DIGITS, text)


def punctuation_spans(text: str) -> list[Span]:
    """Обычная пунктуация: точки, запятые, тире, кавычки, скобки."""
    return _spans(_PUNCTUATION, text)


def special_spans(text: str) -> list[Span]:
    """Любой не-буквенно-цифровой непробельный символ и подчёркивание."""
    return _spans(_SPECIAL, text)


def remove_decorative_symbols(text: str) -> str:
    """Убирает оформительские знаки (•, ►, ✔, |, #, @…), оставляя пунктуацию и знаки, нужные числам."""
    return _DECORATIVE_SYMBOL.sub(" ", text)


def remove_invisible_chars(text: str) -> str:
    """Мягкий перенос, пробелы нулевой ширины, метки направления и селекторы вариантов эмодзи."""
    return _INVISIBLE_CHAR.sub("", text)


# ---------- слова и пробелы ----------

def collapse_whitespace(text: str) -> str:
    return _WHITESPACE.sub(" ", text).strip()


def tidy_lines(text: str) -> str:
    """Пробелы внутри строки схлопнуты, строки обрезаны, подряд не больше одной пустой строки:
    границы абзацев сохраняются, лишние пробелы — нет."""
    lines = (_INLINE_SPACE.sub(" ", line).strip() for line in text.split("\n"))
    return _EXTRA_BLANK_LINES.sub("\n\n", "\n".join(lines)).strip()


def sentences(text: str) -> list[str]:
    """Предложения: разрез по пробелу после «.», «!», «?» или «…»; пустые отброшены."""
    return [s for s in _SENTENCE_END.split(text) if s]


def word_tokens(text: str) -> list[str]:
    return _WORD_TOKEN.findall(text)


def topic_words(text: str) -> list[str]:
    """Слова для подписей тем: только буквы, от трёх символов (токенизатор для CountVectorizer)."""
    return _TOPIC_WORD.findall(text)


# ---------- предметная область ----------

def parse_patch_version(version: str) -> tuple[int, str] | None:
    """`"7.36c"` → `(36, "c")`; не версия 7.xx → None."""
    m = _PATCH_VERSION.match(version)
    return (int(m.group(1)), m.group(2)) if m else None


def patch_mentions(text: str) -> list[str]:
    """Номера патчей, упомянутые в тексте, без повторов: «в 7.41 и 7.41F» → ["7.41", "7.41f"]."""
    return list(dict.fromkeys(f"7.{m.group(1)}{m.group(2).lower()}" for m in _PATCH_MENTION.finditer(text)))


def patch_from_slug(slug: str) -> str | None:
    """`"...-patc-7-41f-v-dota-2..."` → `"7.41f"`."""
    m = _PATCH_IN_SLUG.search(slug)
    return f"7.{m.group(1)}" if m else None


def has_numeric_change(text: str) -> bool:
    """Правка с числами: «урон увеличен с 75 до 80», «с 3 на 2»."""
    return _NUMERIC_CHANGE.search(text) is not None


def has_digit(text: str) -> bool:
    return _HAS_DIGIT.search(text) is not None


def mentions_word(text: str, phrase: str) -> bool:
    """Фраза встречается в тексте как отдельное слово (без учёта регистра)."""
    return re.search(word_boundary_pattern(phrase), text, re.IGNORECASE) is not None


def is_abstention(answer: str) -> bool:
    """Ответ — отказ: начинается с «Не знаю» (так велит отвечать системный промпт при незнании)."""
    return bool(_ABSTENTION.match(answer))


def has_ads_marker(text: str) -> bool:
    """Явные маркеры рекламы ставок. «Бонус» не входит: в патчноутах это игровой термин."""
    return _ADS_MARKER.search(text) is not None


def word_boundary_pattern(phrase: str) -> str:
    """Фраза как отдельное слово, без интерпретации спецсимволов."""
    return rf"\b{re.escape(phrase)}\b"


def build_phrase_finder(phrases: Iterable[str]) -> Callable[[str], set[str]]:
    """Функция «какие из фраз встречаются в тексте как отдельные слова» — один проход по тексту.
    Возвращает фразы в исходном написании; длинные фразы проверяются раньше коротких."""
    by_lower = {p.lower(): p for p in phrases}
    ordered = sorted(by_lower, key=len, reverse=True)
    rx = re.compile("|".join(word_boundary_pattern(p) for p in ordered), re.IGNORECASE)
    return lambda text: {by_lower[m.group(0).lower()] for m in rx.finditer(text)}


def build_any_matcher(patterns: Iterable[str]) -> Callable[[str], bool]:
    """Предикат «текст содержит хотя бы один из шаблонов» (без учёта регистра)."""
    rx = re.compile("|".join(patterns), re.IGNORECASE)
    return lambda text: rx.search(text) is not None


def value_spans(text: str) -> list[tuple[int, int]]:
    """Позиции числовых значений в тексте (урон, проценты, стоимость), без номеров патчей и ссылок [n]."""
    return [m.span() for m in _VALUE.finditer(text)]
