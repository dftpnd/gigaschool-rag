"""Официальные патчноуты Dota 2 на русском с dota2.com/datafeed.

Сохраняет каждый патч как есть (JSON) в data/raw/patches/<версия>.json,
список патчей и справочники героев/предметов/способностей для расшифровки id.
"""

import json

from common import RAW, Fetcher

API = "https://www.dota2.com/datafeed"
LANG = "russian"


def main():
    out = RAW / "patches"
    out.mkdir(parents=True, exist_ok=True)
    f = Fetcher(delay=0.5)

    for name in ("herolist", "itemlist", "abilitylist"):
        data = f.get(f"{API}/{name}", language=LANG).json()
        (out / f"_{name}.json").write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

    patches = f.get(f"{API}/patchnoteslist", language=LANG).json()["patches"]
    (out / "_patchnoteslist.json").write_text(json.dumps(patches, ensure_ascii=False, indent=1), encoding="utf-8")

    for p in patches:
        ver = p["patch_number"]
        path = out / f"{ver}.json"
        if path.exists():
            continue
        data = f.get(f"{API}/patchnotes", version=ver, language=LANG).json()
        path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        print(ver, len(path.read_bytes()) // 1024, "КБ", flush=True)
    print("патчей:", len(patches))


if __name__ == "__main__":
    main()
