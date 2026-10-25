"""Статьи раздела «Патчи» на dota2.ru вместе с комментариями.

HTML страниц сохраняется целиком, с навигацией, рекламой и скриптами — чистка
будет в КТ 2. Комментарии у статьи бывают на нескольких страницах (?page=N).
"""

import re

from common import RAW, Fetcher

BASE = "https://dota2.ru"


def article_links(html: str) -> list[str]:
    return list(dict.fromkeys(re.findall(r'href="(/news/\d+-[^"?#]+)"', html)))


def main():
    out = RAW / "dota2ru"
    out.mkdir(parents=True, exist_ok=True)
    f = Fetcher(delay=1.0)

    links: list[str] = []
    page = 1
    while True:
        html = f.get(f"{BASE}/news/patches/", page=page).text
        found = [l for l in article_links(html) if l not in links]
        if not found:
            break
        links += found
        print(f"страница {page}: +{len(found)}", flush=True)
        page += 1
    print("статей:", len(links), flush=True)

    for link in links:
        slug = link.rsplit("/", 1)[-1]
        first = out / f"{slug}.html"
        if first.exists():
            continue
        html = f.get(BASE + link).text
        first.write_text(html, encoding="utf-8")
        # дополнительные страницы комментариев, если есть
        pages = {int(n) for n in re.findall(re.escape(link) + r'\?page=(\d+)', html)}
        for n in sorted(pages - {1}):
            (out / f"{slug}__p{n}.html").write_text(f.get(BASE + link, page=n).text, encoding="utf-8")
        print(slug[:70], f"+{len(pages - {1})} стр." if len(pages) > 1 else "", flush=True)


if __name__ == "__main__":
    main()
