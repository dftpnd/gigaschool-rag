"""Документ корпуса и чтение/запись корпуса в JSONL."""

import hashlib
import json
from collections import Counter
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path


@dataclass(frozen=True)
class Document:
    """Документ сырого корпуса.

    `text` — сырой текст или HTML как в источнике; `meta` — признаки, специфичные для источника.
    Порядок полей задаёт порядок ключей в JSONL и влияет на sha256 корпуса.
    """
    doc_id: str
    source: str
    date: str | None
    patch: str | None
    url: str
    title: str
    text: str
    meta: dict = field(default_factory=dict)

    def with_meta(self, **extra) -> "Document":
        """Копия документа с дополненным `meta` (исходный не меняется)."""
        return replace(self, meta={**self.meta, **extra})

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)


def duplicate_ids(docs: Iterable[Document]) -> list[str]:
    return [doc_id for doc_id, n in Counter(d.doc_id for d in docs).items() if n > 1]


def ensure_unique_ids(docs: list[Document]) -> list[Document]:
    """`doc_id` — ключ для gold-набора и кэшей, поэтому дубли недопустимы."""
    dup = duplicate_ids(docs)
    if dup:
        raise ValueError(f"неуникальные doc_id ({len(dup)}): {dup[:5]}")
    return docs


def write_jsonl(docs: Iterable[Document], path: Path) -> str:
    """Пишет корпус построчно и возвращает sha256 файла."""
    payload = "".join(d.to_json() + "\n" for d in docs).encode("utf-8")
    path.write_bytes(payload)
    return sha256_hex(payload)


def read_jsonl(path: Path) -> tuple[list[dict], str]:
    """Строки корпуса как словари и sha256 файла."""
    payload = path.read_bytes()
    return [json.loads(line) for line in payload.decode("utf-8").splitlines()], sha256_hex(payload)


def sha256_hex(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()
