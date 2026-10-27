"""Пути и параметры всех КТ — единое место конфигурации скриптов и ноутбуков."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"          # сырьё, корпус и кэши — вне git
GOLD_DIR = ROOT / "gold"      # золотой набор и решения ревью — в git
ENV_FILE = ROOT / ".env"
CACHE = DATA / "cache"
LLM_CACHE = DATA / "llm_cache"
RAW = DATA / "raw"
RAW_PATCHES = RAW / "patches"
RAW_DOTA2RU = RAW / "dota2ru"
RAW_TELEGRAM = RAW / "telegram"
CORPUS = DATA / "corpus.jsonl"

# КТ 1: сборка корпуса
SINCE_PATCH = "7.34"     # окно корпуса: с этого патча есть все четыре источника
OFFTOPIC_SHARE = 0.2     # офтоп Telegram в доле от числа тематических постов
SEED = 42
