"""Отрывки документов — единица разметки релевантности в золотом наборе.

Короткие документы (посты, комментарии, блоки патчноутов) — один отрывок. Длинные статьи
режутся по структуре HTML: заголовок раздела и следующие за ним блоки, а слишком длинный раздел —
на куски не длиннее `max_chars`, без разрыва блоков.
"""

from collections.abc import Iterable, Iterator
from dataclasses import dataclass

from selectolax.lexbor import LexborHTMLParser

from gigarag.patterns import collapse_whitespace
from gigarag.text import visible_text

HEADING_TAGS = frozenset({"h1", "h2", "h3", "h4", "h5", "h6"})
BLOCK_TAGS = frozenset({"p", "ul", "ol", "table", "blockquote"}) | HEADING_TAGS


@dataclass(frozen=True)
class Passage:
    passage_id: str   # <doc_id>#<номер>
    doc_id: str
    text: str


def html_blocks(html: str) -> list[tuple[bool, str]]:
    """Блоки верхнего уровня статьи как (это заголовок?, видимый текст), пустые отброшены."""
    tree = LexborHTMLParser(html)
    root = tree.css_first("[itemprop=articleBody]") or tree.body
    blocks = ((node.tag in HEADING_TAGS, collapse_whitespace(node.text(separator=" ")))
              for node in root.iter() if node.tag in BLOCK_TAGS)
    return [(is_heading, text) for is_heading, text in blocks if text]


def sections(blocks: Iterable[tuple[bool, str]]) -> list[list[str]]:
    """Группы «заголовок + блоки до следующего заголовка»; текст до первого заголовка — своя группа."""
    groups: list[list[str]] = []
    for is_heading, text in blocks:
        if is_heading or not groups:
            groups.append([])
        groups[-1].append(text)
    return groups


def pack(texts: list[str], max_chars: int) -> list[str]:
    """Склеивает блоки подряд, пока длина не превысит `max_chars`; блок длиннее лимита — отдельный кусок."""
    chunks: list[list[str]] = [[]]
    for text in texts:
        if chunks[-1] and sum(map(len, chunks[-1])) + len(text) > max_chars:
            chunks.append([])
        chunks[-1].append(text)
    return [" ".join(c) for c in chunks if c]


def split_article(html: str, max_chars: int) -> list[str]:
    return [piece for group in sections(html_blocks(html)) for piece in pack(group, max_chars)]


def document_passages(doc: dict, max_chars: int = 1500) -> list[Passage]:
    """Отрывки документа корпуса (словарь из corpus.jsonl)."""
    texts = split_article(doc["text"], max_chars) if doc["source"] == "dota2ru_article" else [visible_text(doc["text"])]
    return [Passage(f"{doc['doc_id']}#{i}", doc["doc_id"], t) for i, t in enumerate(texts) if t]


def corpus_passages(docs: Iterable[dict], max_chars: int = 1500) -> Iterator[Passage]:
    for doc in docs:
        yield from document_passages(doc, max_chars)
