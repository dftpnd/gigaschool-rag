"""Источники корпуса: каждый превращает скачанные сырые файлы в документы `Document`."""

from collections.abc import Iterator
from typing import Protocol

from gigarag.corpus import Document
from gigarag.patterns import parse_patch_version


class DocumentSource(Protocol):
    """Источник документов; сборщик корпуса зависит только от этого протокола."""
    name: str

    def documents(self) -> Iterator[Document]: ...


def patch_at_least(version: str | None, since: str) -> bool:
    """Версия патча не старше `since` (сравнение 7.xx с буквенным суффиксом)."""
    parsed = parse_patch_version(version) if version else None
    return parsed is not None and parsed >= parse_patch_version(since)
