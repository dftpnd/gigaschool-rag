"""Метаданные чанков и их применение в поиске: фильтр по патчу, тематике, шуму и prepend в текст.

Структурные: дата, источник, патч, длина документа, позиция чанка. Контентные: тип контента, язык,
топ-5 ключевых слов (TF-IDF). Качественные: шумовой score — доля неалфавитных токенов.
"""

from bisect import bisect_right
from collections.abc import Callable, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from gigarag import patterns
from gigarag.chunking import Chunk
from gigarag.corpus import Document
from gigarag.noise import OffsetsFn, TokenClass, token_class_counts
from gigarag.stopwords import ENGLISH, RUSSIAN
from gigarag.text import latin_letter_share

CONTENT_TYPES = {"patchnotes": "патчноут", "dota2ru_article": "статья", "dota2ru_comment": "комментарий"}
POST = "пост"
OFFTOPIC = "офтоп"
# Язык по доле латиницы среди букв: патчноуты — русский текст с английскими именами (~30–45% латиницы),
# поэтому «смешанный» — отдельная категория, а не ошибка.
RUSSIAN_MAX_LATIN, ENGLISH_MIN_LATIN = 0.2, 0.6
KEYWORDS_PER_CHUNK = 5
SCORED_CLASSES = (TokenClass.WORD, TokenClass.NUM, TokenClass.PUNCT, TokenClass.SPECIAL, TokenClass.EMOJI,
                  TokenClass.HTML, TokenClass.URL)


def content_type(doc: Document) -> str:
    """Тип контента: патчноут, статья, комментарий, пост; нетематический пост Telegram — «офтоп»."""
    if doc.source in CONTENT_TYPES:
        return CONTENT_TYPES[doc.source]
    return POST if doc.meta.get("topical", True) else OFFTOPIC


def language(text: str) -> str:
    share = latin_letter_share(text)
    return "ru" if share <= RUSSIAN_MAX_LATIN else "en" if share >= ENGLISH_MIN_LATIN else "mixed"


def patch_at(date: str | None, releases: Sequence[tuple[str, str]]) -> str | None:
    """Патч, действовавший на дату: последний вышедший не позже `date`. `releases` — (дата, патч) по возрастанию."""
    if not date:
        return None
    i = bisect_right([d for d, _ in releases], date[:10])
    return releases[i - 1][1] if i else None


def patch_releases(docs: Sequence[Document]) -> list[tuple[str, str]]:
    """(дата выхода, патч) по возрастанию даты — по патчноутам корпуса."""
    first: dict[str, str] = {}
    for doc in docs:
        if doc.source == "patchnotes" and doc.patch and doc.date:
            first[doc.patch] = min(first.get(doc.patch, doc.date[:10]), doc.date[:10])
    return sorted((date, patch) for patch, date in first.items())


def noise_score(text: str, offsets: OffsetsFn) -> float:
    """Шумовой score: доля неалфавитных токенов (числа, знаки, эмодзи, разметка, ссылки) среди непробельных."""
    counts = token_class_counts(text, offsets)
    total = sum(counts[c] for c in SCORED_CLASSES)
    return 1 - counts[TokenClass.WORD] / total if total else 1.0


def top_keywords(texts: Sequence[str], k: int = KEYWORDS_PER_CHUNK) -> list[list[str]]:
    """k слов с наибольшим TF-IDF в каждом тексте; словарь — слова от трёх букв без стоп-слов, встречающиеся
    хотя бы в двух текстах и не больше чем в половине."""
    vectorizer = TfidfVectorizer(tokenizer=patterns.topic_words, lowercase=True, token_pattern=None,
                                 stop_words=sorted(RUSSIAN | ENGLISH), min_df=2, max_df=0.5)
    weights = vectorizer.fit_transform(texts).tocsr()
    vocabulary = vectorizer.get_feature_names_out()

    def row_top(i: int) -> list[str]:
        row = weights.getrow(i)
        order = np.argsort(-row.data, kind="stable")[:k]
        return [vocabulary[j] for j in row.indices[order]]

    return [row_top(i) for i in range(weights.shape[0])]


def chunk_metadata(chunks: Sequence[Chunk], docs: Sequence[Document], doc_tokens: Mapping[str, int],
                   offsets: OffsetsFn) -> pd.DataFrame:
    """Таблица метаданных: строка на чанк, в порядке `chunks`."""
    by_id = {d.doc_id: d for d in docs}
    releases = patch_releases(docs)
    per_doc = pd.Series([c.doc_id for c in chunks]).value_counts()
    rows = []
    for chunk in chunks:
        doc = by_id[chunk.doc_id]
        n = per_doc[chunk.doc_id]
        rows.append({
            "chunk_id": chunk.chunk_id, "doc_id": doc.doc_id, "source": doc.source, "date": doc.date,
            "patch": doc.patch or patch_at(doc.date, releases), "title": doc.title,
            "doc_tokens": doc_tokens[doc.doc_id], "position": chunk.position,
            "relative_position": chunk.position / (n - 1) if n > 1 else 0.0,
            "content_type": content_type(doc), "language": language(chunk.text),
            "noise_score": noise_score(chunk.text, offsets),
        })
    frame = pd.DataFrame(rows)
    frame["keywords"] = top_keywords([c.text for c in chunks])
    return frame


# ---------- применение ----------

def with_metadata_prefix(text: str, meta: Mapping) -> str:
    """Чанк с префиксом метаданных: заголовок документа, тип, патч и дата — то, чего часто нет в самом чанке
    (комментарий не называет патч, кусок статьи — её тему)."""
    patch = f"патч {meta['patch']}" if meta.get("patch") else ""
    head = " | ".join(part for part in (meta["title"], meta["content_type"], patch, (meta["date"] or "")[:10]) if part)
    return f"{head}\n{text}"


ChunkFilter = Callable[[pd.DataFrame, str], np.ndarray | None]  # (метаданные, запрос) → маска или None (без фильтра)


def patch_filter(meta: pd.DataFrame, chunk_texts: Sequence[str]) -> ChunkFilter:
    """Если запрос называет патч(и), остаются чанки этих патчей или упоминающие их в тексте."""
    mentions = [frozenset(patterns.patch_mentions(t)) for t in chunk_texts]
    patches = meta["patch"].to_numpy()

    def allowed(_: pd.DataFrame, query: str) -> np.ndarray | None:
        asked = set(patterns.patch_mentions(query))
        if not asked:
            return None
        return np.array([p in asked or bool(m & asked) for p, m in zip(patches, mentions, strict=True)])

    return allowed


def drop_types_filter(types: frozenset[str]) -> ChunkFilter:
    return lambda meta, _: ~meta["content_type"].isin(types).to_numpy()


def max_noise_filter(threshold: float) -> ChunkFilter:
    return lambda meta, _: (meta["noise_score"] <= threshold).to_numpy()


def filter_masks(chunk_filter: ChunkFilter, meta: pd.DataFrame, queries: Mapping[str, str]) -> dict[str, np.ndarray]:
    """Маска допустимых чанков для каждого запроса; запросы, к которым фильтр неприменим, — без маски."""
    masks = {qid: chunk_filter(meta, query) for qid, query in queries.items()}
    return {qid: mask for qid, mask in masks.items() if mask is not None}
