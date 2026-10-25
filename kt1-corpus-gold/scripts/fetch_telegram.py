"""Публичная лента Telegram-канала через веб-превью t.me/s/<канал>.

Листает назад через ?before=<id> и сохраняет HTML каждой страницы
в data/raw/telegram/<канал>/<id первого поста>.html. Повторный запуск
продолжает с самого старого скачанного поста.

    python fetch_telegram.py Dota2 gabenews --since 2023-01-01
"""

import argparse
from pathlib import Path

from gigarag.atomic import atomic_write_text
from gigarag.dates import iso_date
from gigarag.http import PoliteFetcher
from gigarag.patterns import datetime_attrs, telegram_post_ids
from settings import RAW_TELEGRAM

PROGRESS_EVERY = 20  # печатать прогресс раз в столько страниц


def oldest_downloaded(channel_dir: Path) -> int | None:
    ids = [int(p.stem) for p in channel_dir.glob("*.html")]
    return min(ids) if ids else None


def oldest_date(html: str) -> str:
    dates = datetime_attrs(html)
    return iso_date(min(dates)) if dates else ""


def reached_end(min_post_id: int, oldest: str, since: str) -> bool:
    """Лента кончилась (первый пост канала) или ушла старше границы `since`."""
    return min_post_id <= 1 or bool(oldest and oldest < since)


def fetch_channel(fetcher: PoliteFetcher, channel: str, since: str) -> int:
    channel_dir = RAW_TELEGRAM / channel
    channel_dir.mkdir(parents=True, exist_ok=True)
    before, pages = oldest_downloaded(channel_dir), 0
    while True:
        html = fetcher.get(f"https://t.me/s/{channel}", **({"before": before} if before else {})).text
        ids = telegram_post_ids(html, channel)
        if not ids:
            break
        first, oldest = min(ids), oldest_date(html)
        atomic_write_text(channel_dir / f"{first}.html", html)
        pages += 1
        if pages % PROGRESS_EVERY == 0:
            print(f"{channel}: {pages} стр., пост {first}, {oldest}", flush=True)
        if reached_end(first, oldest, since):
            break
        before = first
    print(f"{channel}: готово, {pages} новых стр.", flush=True)
    return pages


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("channels", nargs="+")
    ap.add_argument("--since", default="2018-01-01", help="не листать старше этой даты (YYYY-MM-DD)")
    args = ap.parse_args()
    fetcher = PoliteFetcher(min_interval=1.5)
    for channel in args.channels:
        fetch_channel(fetcher, channel, args.since)


if __name__ == "__main__":
    main()
