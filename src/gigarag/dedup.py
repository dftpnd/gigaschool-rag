"""Точные дубликаты и почти-дубликаты.

Почти-дубликат — пара с точным Jaccard ≥ порога по множествам словесных n-грамм.
MinHash LSH используется только для генерации кандидатов: он ошибается в обе стороны,
поэтому порог кандидатов берётся ниже итогового, а пары проверяются точным Jaccard.
"""

import hashlib
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from datasketch import MinHash, MinHashLSH

from gigarag.patterns import collapse_whitespace, word_tokens

Pair = tuple[str, str]


def normalize_for_dedup(text: str) -> str:
    return collapse_whitespace(text.lower())


def content_hash(normalized: str) -> str:
    return hashlib.md5(normalized.encode()).hexdigest()


def word_shingles(words: list[str], n: int = 3) -> frozenset[str]:
    """Множество словесных n-грамм."""
    return frozenset(" ".join(words[i:i + n]) for i in range(len(words) - n + 1))


def shingle_sets(texts: Mapping[str, str], n: int = 3, min_words: int = 5) -> dict[str, frozenset[str]]:
    """n-граммы нормализованных текстов; тексты короче `min_words` слов не участвуют."""
    words = {doc_id: word_tokens(t) for doc_id, t in texts.items()}
    return {doc_id: word_shingles(w, n) for doc_id, w in words.items() if len(w) >= min_words}


def jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    return len(a & b) / len(a | b) if a or b else 0.0


def lsh_candidate_probability(similarity: float, bands: int, rows: int) -> float:
    """Вероятность, что пара с данным Jaccard станет кандидатом в LSH с `bands` × `rows`: 1 − (1 − s^r)^b."""
    return 1 - (1 - similarity ** rows) ** bands


def minhash_signatures(shingles: Mapping[str, frozenset[str]], num_perm: int, seed: int = 1) -> dict[str, MinHash]:
    def signature(sh: frozenset[str]) -> MinHash:
        m = MinHash(num_perm=num_perm, seed=seed)
        m.update_batch([s.encode() for s in sh])
        return m
    return {doc_id: signature(sh) for doc_id, sh in shingles.items()}


@dataclass(frozen=True)
class LshRun:
    threshold: float
    bands: int
    rows: int
    pairs: list[Pair]


def lsh_candidates(signatures: Mapping[str, MinHash], threshold: float, num_perm: int) -> LshRun:
    """Пары-кандидаты LSH (упорядоченные, без повторов) и выбранные библиотекой bands/rows."""
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    for doc_id, m in signatures.items():
        lsh.insert(doc_id, m)
    pairs = sorted({_ordered(a, b) for a, m in signatures.items() for b in lsh.query(m) if b != a})
    return LshRun(threshold, lsh.b, lsh.r, pairs)


def confirmed_pairs(pairs: Iterable[Pair], shingles: Mapping[str, frozenset[str]],
                    threshold: float) -> list[tuple[str, str, float]]:
    """Кандидаты с точным Jaccard ≥ порога, вместе со значением Jaccard."""
    scored = ((a, b, jaccard(shingles[a], shingles[b])) for a, b in pairs)
    return [p for p in scored if p[2] >= threshold]


def _ordered(a: str, b: str) -> Pair:
    return (a, b) if a <= b else (b, a)
