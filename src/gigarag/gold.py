"""Золотой набор: вопросы, эталонные ответы и размеченные релевантные отрывки.

Релевантность размечается на уровне отрывков (`passage_id`) и документов (`doc_id`) с оценкой:
2 — отрывок содержит ответ или его необходимую часть, 1 — по теме, но ответа не содержит.
Цитата (`quote`) — дословный фрагмент отрывка: по ней релевантность переносится на любые чанки КТ 2.
"""

import hashlib
import json
import random
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path

from gigarag.atomic import atomic_write_text

QUESTION_TYPES = ("factual", "composite", "abstract", "negative")
RELEVANT, RELATED = 2, 1


@dataclass(frozen=True)
class Evidence:
    passage_id: str
    doc_id: str
    grade: int
    quote: str = ""


@dataclass(frozen=True)
class GoldItem:
    qid: str
    qtype: str
    question: str
    answer: str
    evidence: tuple[Evidence, ...] = ()
    split: str = ""
    styles: tuple[str, ...] = ()
    original_question: str = ""
    seed_docs: tuple[str, ...] = ()

    def relevant_docs(self, min_grade: int = RELEVANT) -> list[str]:
        """Документы с оценкой не ниже `min_grade`, без повторов, в порядке разметки."""
        return list(dict.fromkeys(e.doc_id for e in self.evidence if e.grade >= min_grade))

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)

    @classmethod
    def from_dict(cls, d: dict) -> "GoldItem":
        return cls(**{**d, "evidence": tuple(Evidence(**e) for e in d.get("evidence", ())),
                      "styles": tuple(d.get("styles", ())), "seed_docs": tuple(d.get("seed_docs", ()))})


def review_digest(record: dict) -> str:
    """Отпечаток того, что видел ревьюер: вопрос, ответ, затравки и разметка судьи. Решение ревью хранит его,
    и финализация отказывается применять решение к кандидату с другим содержимым под тем же cid."""
    seen = {k: record[k] for k in ("qtype", "question", "answer", "seed_docs", "labels")}
    return hashlib.sha256(json.dumps(seen, ensure_ascii=False, sort_keys=True).encode()).hexdigest()[:16]


def save_gold(items: Iterable[GoldItem], path: Path) -> None:
    atomic_write_text(path, "".join(i.to_json() + "\n" for i in items))


def load_gold(path: Path) -> list[GoldItem]:
    return [GoldItem.from_dict(json.loads(line)) for line in path.read_text(encoding="utf-8").splitlines()]


def stratified_split(items: list[GoldItem], test_share: float, seed: int) -> dict[str, str]:
    """qid → "validation" | "test": внутри каждого типа вопросов доля test ≈ `test_share`,
    но не меньше одного вопроса каждого типа в обеих частях (если вопросов типа хотя бы два)."""
    by_type: dict[str, list[str]] = defaultdict(list)
    for item in sorted(items, key=lambda i: i.qid):
        by_type[item.qtype].append(item.qid)
    rng = random.Random(seed)
    assignment: dict[str, str] = {}
    for qtype in sorted(by_type):
        qids = by_type[qtype]
        rng.shuffle(qids)
        n_test = holdout_size(len(qids), test_share)
        assignment |= dict.fromkeys(qids[:n_test], "test") | dict.fromkeys(qids[n_test:], "validation")
    return assignment


def holdout_size(n: int, share: float) -> int:
    """Размер test-части группы из n вопросов: округление n·share, но 1 ≤ результат ≤ n − 1 при n ≥ 2."""
    return 0 if n < 2 else min(n - 1, max(1, round(n * share)))


@dataclass(frozen=True)
class Candidate:
    """Кандидат в золотой набор до ревью: вопрос, ответ, затравочные документы и цитаты генератора."""
    cid: str
    qtype: str
    question: str
    answer: str
    seed_docs: tuple[str, ...]
    seed_quotes: tuple[str, ...] = ()
    styles: tuple[str, ...] = ()
    original_question: str = ""
    meta: dict = field(default_factory=dict)
