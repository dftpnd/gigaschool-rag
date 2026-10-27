"""Статьи раздела «Патчи» на dota2.ru вместе с комментариями.

HTML страниц сохраняется целиком, с навигацией, рекламой и скриптами — чистка
будет в КТ 2. Комментарии у статьи бывают на нескольких страницах (?page=N).
Повторный запуск докачивает только недостающее: отсутствующие статьи и отсутствующие
страницы комментариев у уже скачанных (пагинация читается из сохранённой первой страницы).
Каждый файл пишется атомарно, поэтому прерванная загрузка не оставляет обрезанных файлов.
"""

from collections.abc import Iterator
from pathlib import Path

from gigarag.atomic import atomic_write_text
from gigarag.http import PoliteFetcher
from gigarag.patterns import news_links
from gigarag.sources.dota2ru import BASE_URL, EXTRA_PAGE_MARK, pending_comment_pages
from settings import RAW_DOTA2RU

LISTING = f"{BASE_URL}/news/patches/"


def listing_links(fetcher: PoliteFetcher) -> Iterator[list[str]]:
    """Ссылки на статьи постранично, пока страница раздела даёт новые ссылки."""
    page = 1
    while links := news_links(fetcher.get(LISTING, page=page).text):
        yield links
        page += 1


def all_article_links(fetcher: PoliteFetcher) -> list[str]:
    seen: dict[str, None] = {}
    for page, links in enumerate(listing_links(fetcher), start=1):
        new = [link for link in links if link not in seen]
        if not new:
            break
        seen.update(dict.fromkeys(new))
        print(f"страница {page}: +{len(new)}", flush=True)
    return list(seen)


def article_slug(link: str) -> str:
    return link.rsplit("/", 1)[-1]


def first_page_path(link: str) -> Path:
    return RAW_DOTA2RU / f"{article_slug(link)}.html"


def extra_page_path(link: str, n: int) -> Path:
    return RAW_DOTA2RU / f"{article_slug(link)}{EXTRA_PAGE_MARK}{n}.html"


def complete_article(fetcher: PoliteFetcher, link: str) -> int:
    """Докачивает первую страницу (если её нет) и недостающие страницы комментариев; возвращает число скачанных."""
    first = first_page_path(link)
    downloaded = 0
    if not first.exists():
        atomic_write_text(first, fetcher.get(BASE_URL + link).text)
        downloaded += 1
    def downloaded_page(n: int) -> bool:
        return extra_page_path(link, n).exists()

    for n in pending_comment_pages(first.read_text(encoding="utf-8"), link, downloaded_page):
        atomic_write_text(extra_page_path(link, n), fetcher.get(BASE_URL + link, page=n).text)
        downloaded += 1
    return downloaded


def main() -> None:
    RAW_DOTA2RU.mkdir(parents=True, exist_ok=True)
    fetcher = PoliteFetcher(min_interval=1.0)
    links = all_article_links(fetcher)
    print("статей:", len(links), flush=True)
    for link in links:
        if downloaded := complete_article(fetcher, link):
            print(article_slug(link)[:70], f"скачано стр.: {downloaded}", flush=True)


if __name__ == "__main__":
    main()
