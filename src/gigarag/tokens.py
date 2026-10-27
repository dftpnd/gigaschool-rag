"""Токенизатор Hugging Face с закреплённой ревизией: длины и границы токенов в единицах эмбеддера."""

from collections.abc import Sequence
from typing import Any

from gigarag.noise import OffsetsFn


def load_tokenizer(name: str, revision: str) -> Any:
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(name, revision=revision)


def token_counts(tokenizer: Any, texts: Sequence[str]) -> list[int]:
    """Число токенов каждого текста без служебных `<s>`/`</s>`."""
    return [len(ids) for ids in tokenizer(list(texts), add_special_tokens=False)["input_ids"]]


def offsets_fn(tokenizer: Any) -> OffsetsFn:
    """Функция «текст → символьные границы токенов» для подсчёта классов токенов в `gigarag.noise`."""
    return lambda text: tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)["offset_mapping"]
