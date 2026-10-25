"""Публичные Telegram-каналы через веб-превью `t.me/s/<канал>`.

Из канала берутся тематические посты и случайный офтоп в заданной доле от их числа.
Выборка детерминирована: популяция упорядочена по `doc_id`, генератор — свой на канал.
"""

import random
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

from selectolax.lexbor import LexborHTMLParser, LexborNode

from gigarag.corpus import Document
from gigarag.dates import iso_date
from gigarag.text import plain_lower

VIDEO_SELECTOR = ".tgme_widget_message_video_player, .tgme_widget_message_roundvideo_player"


def _text(node: LexborNode | None) -> str | None:
    return node.text(strip=True) if node else None


def parse_post(node: LexborNode, channel: str) -> Document:
    post = node.attributes["data-post"]
    body = node.css_first(".tgme_widget_message_text")
    posted = node.css_first("time[datetime]")
    return Document(
        doc_id=f"tg:{post}", source=f"tg_{channel}",
        date=iso_date(posted.attributes["datetime"]) if posted else None, patch=None,
        url=f"https://t.me/{post}", title=f"@{channel}", text=body.html if body else "",
        meta={
            "forwarded_from": _text(node.css_first(".tgme_widget_message_forwarded_from_name")),
            "has_photo": node.css_first(".tgme_widget_message_photo_wrap") is not None,
            "has_video": node.css_first(VIDEO_SELECTOR) is not None,
            "views": _text(node.css_first(".tgme_widget_message_views")),
        },
    )


def parse_feed_page(html: str, channel: str) -> list[Document]:
    return [parse_post(n, channel) for n in LexborHTMLParser(html).css(".tgme_widget_message[data-post]")]


def merge_pages(pages: Iterable[list[Document]]) -> list[Document]:
    """Посты всех страниц без повторов, по `doc_id`. Соседние страницы пересекаются —
    при повторе остаётся первая встреченная версия, поэтому порядок страниц должен быть фиксирован."""
    first_seen: dict[str, Document] = {}
    for page in pages:
        for post in page:
            first_seen.setdefault(post.doc_id, post)
    return sorted(first_seen.values(), key=lambda d: d.doc_id)


def in_window(post: Document, since_date: str) -> bool:
    return bool(post.date and post.date >= since_date and post.text)


def offtopic_quota(topical_count: int, share: float, available: int) -> int:
    """Сколько офтопа добавить: доля `share` от числа тематических, но не больше, чем есть."""
    return min(available, int(topical_count * share))


def select_with_offtopic(posts: list[Document], is_topical: Callable[[str], bool],
                         share: float, rng: random.Random) -> list[Document]:
    """Тематические посты плюс случайный офтоп; каждому проставляется `meta.topical`.
    Результат отсортирован по (дата, doc_id)."""
    topical = [p for p in posts if is_topical(plain_lower(p.text))]
    rest = [p for p in posts if not is_topical(plain_lower(p.text))]
    offtopic = rng.sample(rest, offtopic_quota(len(topical), share, len(rest)))
    marked = [p.with_meta(topical=True) for p in topical] + [p.with_meta(topical=False) for p in offtopic]
    return sorted(marked, key=lambda d: (d.date, d.doc_id))


def feed_pages(channel_dir: Path) -> list[Path]:
    """Файлы страниц ленты по номеру первого поста — фиксированный порядок обхода."""
    return sorted(channel_dir.glob("*.html"), key=lambda p: int(p.stem))


@dataclass(frozen=True)
class TelegramSource:
    raw_dir: Path
    since_date: str
    is_topical: Callable[[str], bool]
    offtopic_share: float
    seed: int
    name: str = "telegram"
    log: Callable[[str], None] = print

    def documents(self) -> Iterator[Document]:
        for channel_dir in sorted(p for p in self.raw_dir.iterdir() if p.is_dir()):
            yield from self.channel_documents(channel_dir)

    def channel_documents(self, channel_dir: Path) -> list[Document]:
        channel = channel_dir.name
        pages = (parse_feed_page(p.read_text(encoding="utf-8"), channel) for p in feed_pages(channel_dir))
        posts = [p for p in merge_pages(pages) if in_window(p, self.since_date)]
        rng = random.Random(f"{self.seed}:{channel}")
        selected = select_with_offtopic(posts, self.is_topical, self.offtopic_share, rng)
        topical = sum(d.meta["topical"] for d in selected)
        self.log(f"  {channel}: постов в окне {len(posts)}, тематических {topical}, офтоп {len(selected) - topical}")
        return selected
