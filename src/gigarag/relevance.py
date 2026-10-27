"""Перенос разметки золотого набора на чанки.

Золотой набор размечен на уровне отрывков КТ 1, и у каждого доказательства есть дословная цитата.
Чанк считается содержащим доказательство, если в нём есть непрерывный кусок цитаты длиной не меньше
`MIN_QUOTE_COVERAGE` её слов. Так разметка не зависит от того, как резался документ: цитата,
разрезанная границей чанков, засчитывается тому чанку, где лежит бо́льшая её часть.
Слова сравниваются в каноническом виде (NFKC, нижний регистр, ё → е, без тегов),
поэтому чистка текста не ломает сопоставление.
"""

import unicodedata
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from difflib import SequenceMatcher
from functools import cache

from gigarag.chunking import Chunk
from gigarag.cleaning import canonical_doc_id
from gigarag.gold import RELEVANT, GoldItem
from gigarag.patterns import word_tokens
from gigarag.text import visible_text

MIN_QUOTE_COVERAGE = 0.5


def match_words(text: str) -> list[str]:
    """Слова текста в каноническом виде для сопоставления цитат."""
    return word_tokens(unicodedata.normalize("NFKC", visible_text(text)).lower().replace("ё", "е"))


def quote_coverage(chunk_words: Sequence[str], quote_words: Sequence[str]) -> float:
    """Доля слов цитаты в самом длинном общем непрерывном фрагменте цитаты и чанка (0…1)."""
    if not quote_words:
        return 0.0
    match = SequenceMatcher(None, chunk_words, quote_words, autojunk=False).find_longest_match()
    return match.size / len(quote_words)


@dataclass(frozen=True)
class EvidenceChunks:
    """Доказательство вопроса и чанки, в которых оно найдено (индексы в списке чанков)."""
    passage_id: str
    grade: int
    chunks: frozenset[int]


Qrels = dict[str, list[EvidenceChunks]]  # qid → доказательства


def build_qrels(items: Sequence[GoldItem], chunks: Sequence[Chunk], aliases: Mapping[str, str] | None = None,
                min_coverage: float = MIN_QUOTE_COVERAGE) -> Qrels:
    """Для каждого доказательства — чанки его документа (или документа, заменившего его при дедупликации),
    покрывающие цитату. Доказательство без найденных чанков остаётся с пустым множеством:
    чистка его потеряла, и это честно снижает полноту."""
    aliases = aliases or {}
    by_doc: dict[str, list[int]] = defaultdict(list)
    for i, chunk in enumerate(chunks):
        by_doc[chunk.doc_id].append(i)
    chunk_words = cache(lambda i: match_words(chunks[i].text))

    def covering(doc_id: str, quote: str) -> frozenset[int]:
        quote_words = match_words(quote)
        candidates = by_doc.get(canonical_doc_id(aliases, doc_id), [])
        return frozenset(i for i in candidates if quote_coverage(chunk_words(i), quote_words) >= min_coverage)

    return {item.qid: [EvidenceChunks(e.passage_id, e.grade, covering(e.doc_id, e.quote)) for e in item.evidence]
            for item in items}


def chunk_grades(evidence: Sequence[EvidenceChunks]) -> dict[int, int]:
    """Оценка каждого размеченного чанка — наибольшая из оценок доказательств, которые в нём есть."""
    grades: dict[int, int] = {}
    for e in evidence:
        for i in e.chunks:
            grades[i] = max(grades.get(i, 0), e.grade)
    return grades


def relevant_chunk_ids(evidence: Sequence[EvidenceChunks], min_grade: int = RELEVANT) -> frozenset[int]:
    return frozenset().union(*(e.chunks for e in evidence if e.grade >= min_grade))


def lost_evidence(qrels: Qrels, min_grade: int = RELEVANT) -> list[tuple[str, str]]:
    """Доказательства (qid, passage_id), которых нет ни в одном чанке: удалены чисткой или не сопоставлены."""
    return [(qid, e.passage_id) for qid, ev in qrels.items() for e in ev if e.grade >= min_grade and not e.chunks]
