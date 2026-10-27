"""Предобработка корпуса: шаги чистки, каждый — чистая функция «корпус → корпус».

Шаги применяются по очереди, и после каждого корпус измеряется отдельно: так виден эффект
каждого решения. Удалённый дубликат не теряется бесследно — `aliases` помнит, какой документ
его заменил, и релевантность из золотого набора переносится на оставшийся.
"""

import html
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from itertools import accumulate, islice

import emoji

from gigarag import dedup
from gigarag.corpus import Document
from gigarag.passages import html_blocks
from gigarag.patterns import (
    remove_decorative_symbols,
    remove_invisible_chars,
    remove_urls,
    strip_tags,
    tags_to_line_breaks,
    tidy_lines,
)

ARTICLE_SOURCE = "dota2ru_article"
TELEGRAM_PREFIX = "tg_"
SIGNATURE_MIN_REPEATS = 20  # последняя строка, повторённая хотя бы в 20 постах канала, — подпись, а не текст


@dataclass(frozen=True)
class CleanCorpus:
    """Корпус на очередном шаге чистки; `aliases` — удалённый doc_id → doc_id, который его заменил."""
    docs: tuple[Document, ...]
    aliases: Mapping[str, str] = field(default_factory=dict)

    def with_docs(self, docs: Iterable[Document], new_aliases: Mapping[str, str] | None = None) -> "CleanCorpus":
        return CleanCorpus(tuple(docs), {**self.aliases, **(new_aliases or {})})


Step = Callable[[CleanCorpus], CleanCorpus]


def text_step(transform: Callable[[Document], str]) -> Step:
    """Шаг, который меняет только текст каждого документа."""
    return lambda corpus: corpus.with_docs(replace(d, text=transform(d)) for d in corpus.docs)


def canonical_doc_id(aliases: Mapping[str, str], doc_id: str) -> str:
    """Документ, который в итоге заменил `doc_id` (цепочка точный дубликат → почти-дубликат)."""
    while doc_id in aliases:
        doc_id = aliases[doc_id]
    return doc_id


# ---------- преобразования текста ----------

def html_to_text(doc: Document) -> str:
    """Видимый текст со структурой строк. Статья — блоки тела статьи (`articleBody`) через пустую строку,
    без навигации и подвала; остальное — теги сняты, `<br>` и концы блоков стали переводами строк."""
    if doc.source == ARTICLE_SOURCE:
        return "\n\n".join(text for _, text in html_blocks(doc.text))
    return html.unescape(strip_tags(tags_to_line_breaks(doc.text), " "))


def normalize_text(text: str) -> str:
    """Unicode NFKC (неразрывный пробел → пробел, «ﬁ» → «fi»), невидимые символы удалены,
    пробелы схлопнуты внутри строк, пустые строки — не больше одной подряд."""
    return tidy_lines(remove_invisible_chars(unicodedata.normalize("NFKC", text)))


def without_urls(text: str) -> str:
    return tidy_lines(remove_urls(text))


def without_emoji_and_symbols(text: str) -> str:
    return tidy_lines(remove_decorative_symbols(emoji.replace_emoji(text, " ")))


def last_line(text: str) -> str:
    lines = text.rstrip().rsplit("\n", 1)
    return lines[-1].strip()


def strip_trailing_lines(text: str, signatures: frozenset[str]) -> str:
    """Срезает с конца текста строки-подписи, пока последняя строка — подпись."""
    while text and last_line(text) in signatures:
        text = text.rstrip().rpartition("\n")[0].rstrip()
    return text


def channel_signatures(docs: Iterable[Document], min_repeats: int = SIGNATURE_MIN_REPEATS) -> dict[str, frozenset[str]]:
    """Подписи каналов Telegram: последние строки постов, повторённые не меньше `min_repeats` раз в канале."""
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    for doc in docs:
        if doc.source.startswith(TELEGRAM_PREFIX) and doc.text.strip():
            counts[doc.source][last_line(doc.text)] += 1
    return {source: frozenset(line for line, n in c.items() if n >= min_repeats) for source, c in counts.items()}


# ---------- шаги ----------

strip_html: Step = text_step(html_to_text)
normalize_unicode: Step = text_step(lambda d: normalize_text(d.text))
drop_urls: Step = text_step(lambda d: without_urls(d.text))
drop_emoji_and_symbols: Step = text_step(lambda d: without_emoji_and_symbols(d.text))


def drop_channel_signatures(corpus: CleanCorpus) -> CleanCorpus:
    signatures = channel_signatures(corpus.docs)
    empty: frozenset[str] = frozenset()
    return corpus.with_docs(replace(d, text=strip_trailing_lines(d.text, signatures.get(d.source, empty)))
                            for d in corpus.docs)


def _earliest(doc: Document) -> tuple[str, str]:
    return doc.date or "9999", doc.doc_id


def drop_exact_duplicates(corpus: CleanCorpus) -> CleanCorpus:
    """Пустые документы удаляются (медиа без подписи), из точных копий остаётся самая ранняя."""
    groups: dict[str, list[Document]] = defaultdict(list)
    for doc in corpus.docs:
        if doc.text.strip():
            groups[dedup.content_hash(dedup.normalize_for_dedup(doc.text))].append(doc)
    keep = {key: min(group, key=_earliest) for key, group in groups.items()}
    aliases = {d.doc_id: keep[key].doc_id for key, group in groups.items() for d in group if d is not keep[key]}
    kept_ids = {d.doc_id for d in keep.values()}
    return corpus.with_docs((d for d in corpus.docs if d.doc_id in kept_ids), aliases)


@dataclass(frozen=True)
class NearDupParams:
    """Почти-дубликаты как в EDA КТ 1: кандидаты MinHash LSH, подтверждение точным Jaccard по словесным 3-граммам."""
    jaccard: float = 0.8
    lsh_threshold: float = 0.5
    num_perm: int = 128
    ngram: int = 3


NEAR_DUP = NearDupParams()


def drop_near_duplicates(corpus: CleanCorpus, params: NearDupParams = NEAR_DUP) -> CleanCorpus:
    """Из каждой группы почти-дубликатов остаётся самый длинный документ (в нём больше всего информации),
    при равной длине — самый ранний."""
    texts = {d.doc_id: dedup.normalize_for_dedup(d.text) for d in corpus.docs}
    shingles = dedup.shingle_sets(texts, n=params.ngram)
    signatures = dedup.minhash_signatures(shingles, params.num_perm)
    candidates = dedup.lsh_candidates(signatures, params.lsh_threshold, params.num_perm).pairs
    pairs = [(a, b) for a, b, _ in dedup.confirmed_pairs(candidates, shingles, params.jaccard)]
    by_id = {d.doc_id: d for d in corpus.docs}
    aliases = {}
    for group in dedup.connected_components(pairs):
        docs = [by_id[i] for i in group]
        best = min(docs, key=lambda d: (-len(d.text), *_earliest(d)))
        aliases |= {d.doc_id: best.doc_id for d in docs if d is not best}
    return corpus.with_docs((d for d in corpus.docs if d.doc_id not in aliases), aliases)


@dataclass(frozen=True)
class CleaningStage:
    name: str
    description: str
    step: Step


PIPELINE: tuple[CleaningStage, ...] = (
    CleaningStage("html", "снятие HTML: теги, сущности, навигация статей", strip_html),
    CleaningStage("unicode", "NFKC, невидимые символы, пробелы и пустые строки", normalize_unicode),
    CleaningStage("urls", "удаление ссылок", drop_urls),
    CleaningStage("emoji", "удаление эмодзи и оформительских символов", drop_emoji_and_symbols),
    CleaningStage("signatures", "подписи каналов Telegram", drop_channel_signatures),
    CleaningStage("exact_dup", "пустые документы и точные дубликаты", drop_exact_duplicates),
    CleaningStage("near_dup", "почти-дубликаты (MinHash + Jaccard ≥ 0,8)", drop_near_duplicates),
)


def run_stages(corpus: CleanCorpus, stages: Sequence[CleaningStage]) -> list[tuple[str, CleanCorpus]]:
    """Корпус после каждого шага по очереди: [(имя шага, корпус)]."""
    states = accumulate(stages, lambda state, stage: stage.step(state), initial=corpus)
    return list(zip((s.name for s in stages), islice(states, 1, None), strict=True))
