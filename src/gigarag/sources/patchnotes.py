"""Официальные патчноуты Dota 2 (JSON с dota2.com/datafeed).

Документ — блок «сущность в патче»: общие изменения, предмет, нейтральный предмет,
герой или нейтральный крип. Рендер блоков в текст — чистые функции над JSON.
"""

import json
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from gigarag.corpus import Document
from gigarag.dates import unix_to_iso_date
from gigarag.patterns import strip_tags
from gigarag.sources import patch_at_least

GENERAL_ITEM_ID = -1  # ability_id общих примечаний раздела предметов


@dataclass(frozen=True)
class Names:
    """Справочники для расшифровки id: герои и предметы/способности."""
    heroes: dict[int, str]
    abilities: dict[int, str]

    def hero(self, hero_id: int) -> str:
        return self.heroes.get(hero_id, f"hero_{hero_id}")

    def ability(self, ability_id: int, fallback_prefix: str = "ability") -> str:
        return self.abilities.get(ability_id, f"{fallback_prefix}_{ability_id}")


# ---------- рендер блоков (чистые функции) ----------

def render_note(note: dict) -> str:
    """Строка правки: маркер «- » с отступом по `indent_level`, кроме служебных строк (`hide_dot`)."""
    text = note.get("note", "")
    if note.get("hide_dot"):
        return text
    return "  " * (note.get("indent_level", 1) - 1) + "- " + text


def render_notes(notes: list[dict] | None) -> list[str]:
    return [render_note(n) for n in notes or []]


def render_ability(ability: dict, names: Names) -> list[str]:
    return [f"{names.ability(ability.get('ability_id'))}:", *render_notes(ability.get("ability_notes"))]


def render_subsection(sub: dict, names: Names) -> list[str]:
    """Подраздел героя (врождённая способность, аспект): заголовок, общие правки, способности."""
    abilities = [line for ab in sub.get("abilities") or [] for line in render_ability(ab, names)]
    return [f"[{sub.get('title', '')}]", *render_notes(sub.get("general_notes")), *abilities]


def render_hero(hero: dict, names: Names) -> list[str]:
    abilities = [line for ab in hero.get("abilities") or [] for line in render_ability(ab, names)]
    subsections = [line for sub in hero.get("subsections") or [] for line in render_subsection(sub, names)]
    talents = ["Таланты:", *render_notes(hero["talent_notes"])] if hero.get("talent_notes") else []
    return [*render_notes(hero.get("hero_notes")), *abilities, *subsections, *talents]


def item_entity_name(item: dict, kind: str, names: Names) -> str:
    if item["ability_id"] == GENERAL_ITEM_ID:
        return "Нейтральные предметы: общее" if kind == "neutral_items" else "Предметы: общее"
    return names.ability(item["ability_id"], "item")


def general_entity_name(block: dict, index: int) -> str:
    title = strip_tags(block.get("title") or "", "") or "Общие изменения"
    return f"{title} #{index}" if index else title


def patch_blocks(data: dict, names: Names) -> Iterator[tuple[str, str, list[str]]]:
    """Блоки патча в порядке источника: (вид, сущность, строки текста)."""
    for i, block in enumerate(data.get("general_notes") or []):
        yield "general", general_entity_name(block, i), render_notes(block.get("generic"))
    for kind in ("items", "neutral_items"):
        for item in data.get(kind) or []:
            yield kind, item_entity_name(item, kind, names), render_notes(item.get("ability_notes"))
    for hero in data.get("heroes") or []:
        yield "hero", names.hero(hero["hero_id"]), render_hero(hero, names)
    for creep in data.get("neutral_creeps") or []:
        name = creep.get("localized_name") or creep["name"]
        yield "neutral_creep", name, render_notes(creep.get("neutral_creep_notes"))


def unique_suffixes(keys: list[tuple[str, str]]) -> list[str]:
    """Суффикс `#n` для второго и следующих повторов ключа — чтобы id внутри патча были уникальны."""
    seen: Counter = Counter()

    def suffix(key):
        seen[key] += 1
        return f"#{seen[key]}" if seen[key] > 1 else ""
    return [suffix(k) for k in keys]


def patch_documents(version: str, timestamp: int, data: dict, names: Names) -> list[Document]:
    blocks = list(patch_blocks(data, names))
    suffixes = unique_suffixes([(kind, entity) for kind, entity, _ in blocks])
    date, url = unix_to_iso_date(timestamp), f"https://www.dota2.com/patches/{version}"
    return [
        Document(
            doc_id=f"patch:{version}:{kind}:{entity}{suffix}", source="patchnotes", date=date, patch=version,
            url=url, title=f"Патч {version} — {entity}", text=f"Патч {version} — {entity}\n" + "\n".join(lines),
            meta={"kind": kind, "entity": entity},
        )
        for (kind, entity, lines), suffix in zip(blocks, suffixes, strict=True)
    ]


# ---------- справочники ----------

def _items(payload: dict) -> list[dict]:
    return payload["result"]["data"]["itemabilities"]


def _heroes(payload: dict) -> list[dict]:
    return payload["result"]["data"]["heroes"]


def load_names(raw_dir: Path) -> Names:
    read = lambda name: json.loads((raw_dir / name).read_text())  # noqa: E731
    abilities = {a["id"]: a["name_loc"] or a["name_english_loc"]
                 for name in ("_itemlist.json", "_abilitylist.json") for a in _items(read(name))}
    return Names(heroes={h["id"]: h["name_loc"] for h in _heroes(read("_herolist.json"))}, abilities=abilities)


def entity_vocabulary(raw_dir: Path) -> list[str]:
    """Имена героев и предметов из справочников (без рецептов и коротких имён) — для тематического фильтра."""
    read = lambda name: json.loads((raw_dir / name).read_text())  # noqa: E731
    heroes = [h["name_loc"] for h in _heroes(read("_herolist.json"))]
    items = [a["name_loc"] for a in _items(read("_itemlist.json"))
             if a["name_loc"] and not a["name_loc"].startswith("Recipe") and len(a["name_loc"]) > 3]
    return heroes + items


def patch_dates(raw_dir: Path) -> dict[str, str]:
    """Версия патча → дата выхода."""
    patches = json.loads((raw_dir / "_patchnoteslist.json").read_text())
    return {p["patch_number"]: unix_to_iso_date(p["patch_timestamp"]) for p in patches}


# ---------- источник ----------

@dataclass(frozen=True)
class PatchnotesSource:
    raw_dir: Path
    since_patch: str
    name: str = "patchnotes"

    def documents(self) -> Iterator[Document]:
        names = load_names(self.raw_dir)
        for patch in json.loads((self.raw_dir / "_patchnoteslist.json").read_text()):
            version = patch["patch_number"]
            if patch_at_least(version, self.since_patch):
                data = json.loads((self.raw_dir / f"{version}.json").read_text())
                yield from patch_documents(version, patch["patch_timestamp"], data, names)
