"""Пути и параметры КТ 1 — единое место для конфигурации сборщиков и сборки корпуса."""

from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data"
RAW = DATA / "raw"
RAW_PATCHES = RAW / "patches"
RAW_DOTA2RU = RAW / "dota2ru"
RAW_TELEGRAM = RAW / "telegram"
CORPUS = DATA / "corpus.jsonl"

SINCE_PATCH = "7.34"     # окно корпуса: с этого патча есть все четыре источника
OFFTOPIC_SHARE = 0.2     # офтоп Telegram в доле от числа тематических постов
SEED = 42
