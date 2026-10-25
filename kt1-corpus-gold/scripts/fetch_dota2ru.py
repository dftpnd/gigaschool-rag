"""Статьи раздела «Патчи» на dota2.ru вместе с комментариями.

HTML страниц сохраняется целиком, с навигацией, рекламой и скриптами — чистка
будет в КТ 2. Комментарии у статьи бывают на нескольких страницах (?page=N).
Повторный запуск пропускает уже скачанные статьи.
"""

from collections.abc import Iterator

from gigarag.http import PoliteFetcher
from gigarag.patterns import news_links, page_numbers
from gigarag.sources.dota2ru import BASE_URL, EXTRA_PAGE_MARK
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


def download_article(fetcher: PoliteFetcher, link: str) -> int:
    """Первая страница статьи и дополнительные страницы комментариев; возвращает число доп. страниц."""
    slug = link.rsplit("/", 1)[-1]
    html = fetcher.get(BASE_URL + link).text
    (RAW_DOTA2RU / f"{slug}.html").write_text(html, encoding="utf-8")
    extra = sorted(page_numbers(html, link) - {1})
    for n in extra:
        page_html = fetcher.get(BASE_URL + link, page=n).text
        (RAW_DOTA2RU / f"{slug}{EXTRA_PAGE_MARK}{n}.html").write_text(page_html, encoding="utf-8")
    return len(extra)


def main() -> None:
    RAW_DOTA2RU.mkdir(parents=True, exist_ok=True)
    fetcher = PoliteFetcher(min_interval=1.0)
    links = all_article_links(fetcher)
    print("статей:", len(links), flush=True)
    for link in links:
        slug = link.rsplit("/", 1)[-1]
        if (RAW_DOTA2RU / f"{slug}.html").exists():
            continue
        extra = download_article(fetcher, link)
        print(slug[:70], f"+{extra} стр." if extra else "", flush=True)


if __name__ == "__main__":
    main()
