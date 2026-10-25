"""Классификация символов и токенов сырого текста: что в нём слова, а что шум.

Каждому символу присваивается класс; при пересечении побеждает класс с бо́льшим
приоритетом (порядок в `TokenClass`). Токен получает класс своего первого символа.
"""

from collections.abc import Callable, Sequence
from enum import IntEnum

import emoji
import numpy as np
import pandas as pd

from gigarag import patterns
from gigarag.patterns import Span


class TokenClass(IntEnum):
    """Класс символа; значение — приоритет при пересечении."""
    SPACE = 0
    WORD = 1
    NUM = 2
    PUNCT = 3
    SPECIAL = 4
    EMOJI = 5
    HTML = 6
    URL = 7


CLASS_NAMES = [c.name.lower() for c in TokenClass]
CONTENT_CLASSES = ["word", "num", "punct", "special", "emoji", "html", "url"]
REMOVABLE_NOISE = ["html", "url", "emoji", "special"]

OffsetsFn = Callable[[str], Sequence[tuple[int, int]]]


def emoji_spans(text: str) -> list[Span]:
    return [(e["match_start"], e["match_end"]) for e in emoji.emoji_list(text)]


def _span_layers(text: str) -> list[tuple[TokenClass, list[Span]]]:
    """Слои разметки от низкого приоритета к высокому — более поздний перезаписывает."""
    return [
        (TokenClass.WORD, patterns.letter_spans(text)),
        (TokenClass.SPECIAL, patterns.special_spans(text)),
        (TokenClass.PUNCT, patterns.punctuation_spans(text)),
        (TokenClass.NUM, patterns.digit_spans(text)),
        (TokenClass.EMOJI, emoji_spans(text)),
        (TokenClass.HTML, patterns.tag_spans(text) + patterns.entity_spans(text)),
        (TokenClass.URL, patterns.url_spans(text)),
    ]


def char_classes(text: str) -> np.ndarray:
    """Класс каждого символа текста (массив `int8` длины `len(text)`)."""
    codes = np.zeros(len(text), dtype=np.int8)
    for cls, spans in _span_layers(text):
        for start, end in spans:
            codes[start:end] = cls
    return codes


def token_class_counts(text: str, offsets: OffsetsFn) -> np.ndarray:
    """Число токенов каждого класса; `offsets` — символьные границы токенов от токенизатора."""
    starts = np.array([s for s, e in offsets(text) if e > s], dtype=np.int64)
    if not len(starts):
        return np.zeros(len(TokenClass), dtype=np.int64)
    return np.bincount(char_classes(text)[starts], minlength=len(TokenClass))


def class_shares_percent(counts: pd.DataFrame, by: str, classes: list[str] = CONTENT_CLASSES) -> pd.DataFrame:
    """Доли классов в процентах внутри каждой группы `by` и строка «ВСЕГО» по всему корпусу."""
    totals = counts.groupby(by)[classes].sum()
    totals.loc["ВСЕГО"] = totals.sum()
    return totals.div(totals.sum(axis=1), axis=0) * 100


def removable_noise_percent(shares: pd.DataFrame) -> pd.Series:
    """Сумма долей классов, которые предобработка может удалить без потери смысла."""
    return shares[REMOVABLE_NOISE].sum(axis=1)
