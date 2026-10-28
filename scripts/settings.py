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

# КТ 5: ручная разметка ответов
# выборка → файл оценок; dev — подбор версии судьи, holdout — итоговая kappa. Выборки пишет ноутбук kt5_generation
# в data/: в них тексты чанков корпуса (контент третьих сторон), а в git — только оценки по id ответа.
KT5_DATA = DATA / "kt5"
ANNOTATION_SETS = {
    "dev": (KT5_DATA / "annotation_dev.json", GOLD_DIR / "kt5_human_labels.json"),
    "holdout": (KT5_DATA / "annotation_holdout.json", GOLD_DIR / "kt5_human_labels_holdout.json"),
}
ANNOTATION_HOST, ANNOTATION_PORT = "0.0.0.0", 8766  # все интерфейсы: разметка с другого устройства через VPN
