"""Отрывки документов — единица разметки релевантности в золотом наборе.

Короткие документы (посты, комментарии, блоки патчноутов) — один отрывок. Длинные статьи
режутся по структуре HTML: заголовок раздела и следующие за ним блоки, а слишком длинный раздел —
на куски не длиннее `max_chars` (блоки не разрываются, если сами не длиннее лимита).
Весь видимый текст статьи попадает в отрывки ровно один раз.
"""

import json
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from itertools import groupby

from selectolax.lexbor import LexborHTMLParser, LexborNode

from gigarag.corpus import sha256_hex
from gigarag.patterns import collapse_whitespace, sentences, word_tokens
from gigarag.text import visible_text

HEADING_TAGS = frozenset({"h1", "h2", "h3", "h4", "h5", "h6"})
BLOCK_TAGS = frozenset({"p", "ul", "ol", "table", "blockquote"}) | HEADING_TAGS
# Строчные узлы и голый текст: подряд идущие склеиваются в один блок. Всё прочее — контейнер, в него спускаемся.
INLINE_TAGS = frozenset({"-text", "a", "span", "strong", "b", "i", "em", "u", "s", "br", "img", "sup", "sub",
                         "font", "code", "small", "mark", "abbr", "del", "ins", "q", "cite", "time"})


@dataclass(frozen=True)
class Passage:
    passage_id: str   # <doc_id>#<номер>
    doc_id: str
    text: str


def html_blocks(html: str) -> list[tuple[bool, str]]:
    """Блоки статьи в порядке чтения как (это заголовок?, видимый текст), пустые отброшены.

    Обход рекурсивный: блок (`p`, `ul`, `h*`…) берётся целиком, в контейнеры (`div`, `li` вне списка…)
    спускаемся, подряд идущие строчные узлы и текст склеиваются — так текст не теряется и не дублируется.
    """
    tree = LexborHTMLParser(html)
    root = tree.css_first("[itemprop=articleBody]") or tree.body
    return [(is_heading, text) for is_heading, text in _walk(root) if text]


def _node_kind(node: LexborNode) -> str:
    return "block" if node.tag in BLOCK_TAGS else "inline" if node.tag in INLINE_TAGS else "container"


def _node_text(node: LexborNode) -> str:
    return node.text(separator=" ") if node.tag != "-text" else node.text()


def _walk(node: LexborNode) -> Iterator[tuple[bool, str]]:
    for kind, group in groupby(node.iter(include_text=True), key=_node_kind):
        if kind == "inline":
            yield False, collapse_whitespace(" ".join(map(_node_text, group)))
        elif kind == "block":
            yield from ((child.tag in HEADING_TAGS, collapse_whitespace(_node_text(child))) for child in group)
        else:
            yield from (block for child in group for block in _walk(child))


def sections(blocks: Iterable[tuple[bool, str]]) -> list[list[str]]:
    """Группы «заголовок + блоки до следующего заголовка»; текст до первого заголовка — своя группа."""
    groups: list[list[str]] = []
    for is_heading, text in blocks:
        if is_heading or not groups:
            groups.append([])
        groups[-1].append(text)
    return groups


def split_long(text: str, max_chars: int) -> list[str]:
    """Блок длиннее `max_chars` — на куски по предложениям, слишком длинное предложение — по словам,
    слово длиннее лимита — посимвольно. Каждый кусок не длиннее `max_chars`."""
    if len(text) <= max_chars:
        return [text]
    pieces = sentences(text)
    if len(pieces) == 1:
        pieces = text.split(" ")
    if len(pieces) == 1:
        return [text[i:i + max_chars] for i in range(0, len(text), max_chars)]
    return pack([part for piece in pieces for part in split_long(piece, max_chars)], max_chars)


def pack(texts: list[str], max_chars: int) -> list[str]:
    """Склеивает блоки через пробел, пока длина куска с разделителями не превысит `max_chars`.
    Блоки длиннее лимита предварительно режутся `split_long`, поэтому ни один кусок не длиннее `max_chars`."""
    chunks: list[str] = []
    for text in (part for t in texts for part in split_long(t, max_chars)):
        if chunks and len(chunks[-1]) + 1 + len(text) <= max_chars:
            chunks[-1] = f"{chunks[-1]} {text}"
        else:
            chunks.append(text)
    return chunks


def split_article(html: str, max_chars: int) -> list[str]:
    return [piece for group in sections(html_blocks(html)) for piece in pack(group, max_chars)]


def document_passages(doc: dict, max_chars: int = 1500) -> list[Passage]:
    """Отрывки документа корпуса (словарь из corpus.jsonl). Для статьи проверяется, что отрывки подряд
    дают тот же видимый текст, что и документ: без потерь и без повторов."""
    if doc["source"] != "dota2ru_article":
        texts = [visible_text(doc["text"])]
    else:
        texts = split_article(doc["text"], max_chars)
        if word_tokens(" ".join(texts)) != word_tokens(visible_text(doc["text"])):
            raise ValueError(f"{doc['doc_id']}: отрывки не совпадают с видимым текстом статьи")
    return [Passage(f"{doc['doc_id']}#{i}", doc["doc_id"], t) for i, t in enumerate(texts) if t]


def passages_fingerprint(passages: Iterable[Passage]) -> str:
    """sha256 упорядоченных пар (passage_id, текст): меняется текст, разбиение или порядок — меняется отпечаток."""
    payload = "".join(json.dumps([p.passage_id, p.text], ensure_ascii=False) + "\n" for p in passages)
    return sha256_hex(payload.encode("utf-8"))


def corpus_passages(docs: Iterable[dict], max_chars: int = 1500) -> Iterator[Passage]:
    for doc in docs:
        yield from document_passages(doc, max_chars)
