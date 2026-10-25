"""Официальные патчноуты Dota 2 на русском с dota2.com/datafeed.

Сохраняет каждый патч как есть (JSON) в data/raw/patches/<версия>.json,
список патчей и справочники героев/предметов/способностей для расшифровки id.
Повторный запуск докачивает только отсутствующие патчи.
"""

import json
from pathlib import Path

from gigarag.http import PoliteFetcher
from settings import RAW_PATCHES

API = "https://www.dota2.com/datafeed"
LANG = "russian"
REFERENCE_LISTS = ("herolist", "itemlist", "abilitylist")


def save_json(path: Path, data, indent: int | None = None) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=indent), encoding="utf-8")


def main() -> None:
    RAW_PATCHES.mkdir(parents=True, exist_ok=True)
    fetcher = PoliteFetcher(min_interval=0.5)

    for name in REFERENCE_LISTS:
        save_json(RAW_PATCHES / f"_{name}.json", fetcher.get(f"{API}/{name}", language=LANG).json())

    patches = fetcher.get(f"{API}/patchnoteslist", language=LANG).json()["patches"]
    save_json(RAW_PATCHES / "_patchnoteslist.json", patches, indent=1)

    missing = [p["patch_number"] for p in patches if not (RAW_PATCHES / f"{p['patch_number']}.json").exists()]
    for version in missing:
        path = RAW_PATCHES / f"{version}.json"
        save_json(path, fetcher.get(f"{API}/patchnotes", version=version, language=LANG).json())
        print(version, path.stat().st_size // 1024, "КБ", flush=True)
    print("патчей:", len(patches))


if __name__ == "__main__":
    main()
