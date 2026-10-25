"""Публичная лента Telegram-канала через веб-превью t.me/s/<канал>.

Листает назад через ?before=<id> и сохраняет HTML каждой страницы
в data/raw/telegram/<канал>/<id первого поста>.html. Повторный запуск
продолжает с самого старого скачанного поста.

    python fetch_telegram.py Dota2 gabenews --since 2023-01-01
"""

import argparse
import re

from common import RAW, Fetcher


def fetch_channel(f: Fetcher, channel: str, since: str):
    out = RAW / "telegram" / channel
    out.mkdir(parents=True, exist_ok=True)
    done = sorted(int(p.stem) for p in out.glob("*.html"))
    before = done[0] if done else None
    pages = 0
    while True:
        html = f.get(f"https://t.me/s/{channel}", **({"before": before} if before else {})).text
        ids = [int(i) for i in re.findall(rf'data-post="{channel}/(\d+)"', html, re.I)]
        if not ids:
            break
        (out / f"{min(ids)}.html").write_text(html, encoding="utf-8")
        pages += 1
        dates = re.findall(r'<time datetime="([^"]+)"', html)
        oldest = min(dates) if dates else ""
        if pages % 20 == 0:
            print(f"{channel}: {pages} стр., пост {min(ids)}, {oldest[:10]}", flush=True)
        if min(ids) <= 1 or (oldest and oldest[:10] < since):
            break
        before = min(ids)
    print(f"{channel}: готово, {pages} новых стр.", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("channels", nargs="+")
    ap.add_argument("--since", default="2018-01-01", help="не листать старше этой даты (YYYY-MM-DD)")
    args = ap.parse_args()
    f = Fetcher(delay=1.5)
    for ch in args.channels:
        fetch_channel(f, ch, args.since)


if __name__ == "__main__":
    main()
