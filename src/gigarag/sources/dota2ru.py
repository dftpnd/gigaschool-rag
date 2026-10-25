"""Статьи раздела «Патчи» dota2.ru и комментарии к ним.

Статья и каждый комментарий — отдельные документы; комментарий хранит id статьи в `meta`.
Разбор страницы — чистые функции над HTML, источник только читает файлы.
"""

import hashlib
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path

from selectolax.lexbor import LexborHTMLParser, LexborNode

from gigarag.corpus import Document
from gigarag.dates import iso_date, unix_to_iso_date
from gigarag.patterns import page_numbers, patch_from_slug
from gigarag.sources import patch_at_least

BASE_URL = "https://dota2.ru"
EXTRA_PAGE_MARK = "__p"  # файл дополнительной страницы комментариев: <slug>__p<N>.html


@dataclass(frozen=True)
class ArticleRef:
    """Что известно о статье, к которой относятся комментарии."""
    article_id: str
    patch: str
    url: str
    title: str
    date: str | None


def article_id(slug: str) -> str:
    """`"67852-patc-7-41f-..."` → `"67852"`."""
    return slug.split("-", 1)[0]


def comment_fallback_id(article: str, message_html: str) -> str:
    """Стабильный id комментария без собственного номера (а не `hash()`, рандомизированный между запусками)."""
    return "h" + hashlib.blake2b(f"{article}\n{message_html}".encode(), digest_size=8).hexdigest()


def _text(node: LexborNode | None) -> str | None:
    return node.text(strip=True) if node else None


def _attr(node: LexborNode | None, name: str) -> str | None:
    return node.attributes.get(name) if node else None


def parse_article(html: str, slug: str, patch: str, fallback_date: str | None) -> tuple[ArticleRef, Document | None]:
    """Заголовок, дата и тело статьи. Документа нет, если на странице нет тела статьи."""
    tree = LexborHTMLParser(html)
    published = _attr(tree.css_first("time[datetime]"), "datetime")
    ref = ArticleRef(
        article_id=article_id(slug), patch=patch, url=f"{BASE_URL}/news/{slug}",
        title=_text(tree.css_first("h1")) or slug,
        date=iso_date(published) if published else fallback_date,
    )
    body = tree.css_first("[itemprop=articleBody]")
    doc = Document(
        doc_id=f"dota2ru:article:{ref.article_id}", source="dota2ru_article", date=ref.date, patch=patch,
        url=ref.url, title=ref.title, text=body.html, meta={},
    ) if body else None
    return ref, doc


def parse_comment(node: LexborNode, article: ArticleRef) -> Document | None:
    message = node.css_first(".global-content__comment-user--message")
    if not message:
        return None
    msg_html = message.html
    own_id = _attr(node.css_first("[data-reply-answer]"), "data-reply-answer")
    posted = _attr(node.css_first("[data-time]"), "data-time")
    return Document(
        doc_id=f"dota2ru:comment:{own_id or comment_fallback_id(article.article_id, msg_html)}",
        source="dota2ru_comment",
        date=unix_to_iso_date(int(posted)) if posted else article.date,
        patch=article.patch, url=article.url, title=f"Комментарий к «{article.title}»", text=msg_html,
        meta={"article_id": article.article_id, "author": _text(node.css_first(".global-content__comment-name")),
              "has_quote": "blockquote" in msg_html or "bbCodeQuote" in msg_html},
    )


def parse_comments(html: str, article: ArticleRef) -> list[Document]:
    nodes = LexborHTMLParser(html).css(".global-content__comment-wrap")
    return [doc for doc in (parse_comment(n, article) for n in nodes) if doc]


# ---------- файлы ----------

def is_extra_page(path: Path) -> bool:
    return EXTRA_PAGE_MARK in path.stem


def extra_page_number(path: Path) -> int:
    return int(path.stem.rsplit("p", 1)[1])


def pending_comment_pages(first_html: str, link: str, downloaded: Callable[[int], bool]) -> list[int]:
    """Номера дополнительных страниц комментариев, на которые ссылается первая страница статьи,
    но которые ещё не скачаны (`downloaded(n)` — есть ли страница n)."""
    return [n for n in sorted(page_numbers(first_html, link) - {1}) if not downloaded(n)]


def comment_pages(raw_dir: Path, first: Path) -> list[Path]:
    """Первая страница статьи и дополнительные страницы комментариев по порядку."""
    extra = sorted(raw_dir.glob(f"{first.stem}{EXTRA_PAGE_MARK}*.html"), key=extra_page_number)
    return [first, *extra]


@dataclass(frozen=True)
class Dota2ruSource:
    raw_dir: Path
    since_patch: str
    patch_dates: dict[str, str]
    name: str = "dota2ru"

    def documents(self) -> Iterator[Document]:
        for first in sorted(p for p in self.raw_dir.glob("*.html") if not is_extra_page(p)):
            patch = patch_from_slug(first.stem)
            if not patch_at_least(patch, self.since_patch):
                continue
            pages = [p.read_text(encoding="utf-8") for p in comment_pages(self.raw_dir, first)]
            ref, article = parse_article(pages[0], first.stem, patch, self.patch_dates.get(patch))
            if article:
                yield article
            for page in pages:
                yield from parse_comments(page, ref)
