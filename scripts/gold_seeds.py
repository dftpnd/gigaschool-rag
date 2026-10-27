"""Отбор затравочных документов для генерации вопросов золотого набора.

Все функции чистые: корпус (список словарей с полем `visible`) и генератор случайных чисел
передаются явно, поэтому затравки воспроизводимы при том же корпусе и seed.
"""

import random
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date

from gigarag.patterns import build_phrase_finder, has_digit, has_numeric_change

Doc = dict
Predicate = Callable[[Doc], bool]

ENTITY_KINDS = frozenset({"hero", "items", "neutral_items"})
MAJOR_PATCH_LEN = 4                      # "7.36" — патч без буквенного суффикса
AFTER_PATCH_DAYS = 30                    # пост Telegram считается реакцией на патч в этом окне
ABSTRACT_THEMES = ("нейтрал", "аспект", "винрейт", "staging", "баг")


@dataclass(frozen=True)
class Seed:
    qtype: str
    doc_ids: tuple[str, ...]
    kind: str = ""          # подвид затравки, например "two_patches" или вид негативного вопроса


# ---------- предикаты ----------

def length_between(lo: int, hi: int) -> Predicate:
    return lambda d: lo <= len(d["visible"]) <= hi


def is_entity_block(d: Doc) -> bool:
    return d["source"] == "patchnotes" and d["meta"].get("kind") in ENTITY_KINDS


def is_topical_post(d: Doc) -> bool:
    return d["source"].startswith("tg_") and d["meta"].get("topical") is True


def is_major_patch(version: str | None) -> bool:
    return bool(version) and len(version) == MAJOR_PATCH_LEN


def days_between(earlier: str, later: str) -> int:
    return (date.fromisoformat(later) - date.fromisoformat(earlier)).days


def all_of(*predicates: Predicate) -> Predicate:
    return lambda d: all(p(d) for p in predicates)


# ---------- выборка ----------

def sample(rng: random.Random, items: list, n: int) -> list:
    return rng.sample(items, min(n, len(items)))


def round_robin_by(items: list[Doc], key: Callable[[Doc], str], n: int, rng: random.Random) -> list[Doc]:
    """n элементов, по очереди из разных групп `key` — чтобы затравки покрывали разные патчи."""
    groups: dict[str, list[Doc]] = defaultdict(list)
    for item in items:
        groups[key(item)].append(item)
    queues = [rng.sample(g, len(g)) for _, g in sorted(groups.items())]
    rng.shuffle(queues)
    picked: list[Doc] = []
    while len(picked) < n and any(queues):
        picked += [q.pop() for q in queues if q][: n - len(picked)]
    return picked


# ---------- затравки по типам ----------

def factual_seeds(docs: list[Doc], rng: random.Random, n_patch: int, n_post: int) -> list[Seed]:
    is_block = all_of(is_entity_block, length_between(80, 700))
    blocks = [d for d in docs if is_block(d) and has_numeric_change(d["visible"])]
    posts = [d for d in docs if all_of(is_topical_post, length_between(150, 700))(d) and has_digit(d["visible"])]
    picked = round_robin_by(blocks, lambda d: d["patch"], n_patch, rng) + sample(rng, posts, n_post)
    return [Seed("factual", (d["doc_id"],), "patch" if d["source"] == "patchnotes" else "post") for d in picked]


def two_patches_seeds(docs: list[Doc], rng: random.Random, n: int) -> list[Seed]:
    """Одна сущность в двух разных патчах — вопрос о том, как она менялась."""
    by_entity: dict[str, list[Doc]] = defaultdict(list)
    for d in docs:
        if all_of(is_entity_block, length_between(80, 900))(d) and has_numeric_change(d["visible"]):
            by_entity[d["meta"]["entity"]].append(d)
    multi = sorted(e for e, ds in by_entity.items() if len({d["patch"] for d in ds}) >= 2)
    seeds = []
    for entity in sample(rng, multi, n):
        first, second = rng.sample(sorted(by_entity[entity], key=lambda d: d["doc_id"]), 2)
        if first["patch"] != second["patch"]:
            seeds.append(Seed("composite", (first["doc_id"], second["doc_id"]), "two_patches"))
    return seeds


def entity_blocks_by_name(docs: list[Doc]) -> dict[str, list[Doc]]:
    names: dict[str, list[Doc]] = defaultdict(list)
    for d in docs:
        if is_entity_block(d) and len(d["meta"]["entity"]) > 3:
            names[d["meta"]["entity"]].append(d)
    return names


def patch_and_post_seeds(docs: list[Doc], rng: random.Random, n: int) -> list[Seed]:
    """Правка в патчноуте и пост Telegram о той же сущности в течение месяца после патча."""
    blocks = entity_blocks_by_name(docs)
    find_names = build_phrase_finder(blocks)
    posts = [d for d in docs if all_of(is_topical_post, length_between(150, 900))(d)]
    pairs = [(block, post) for post in posts for name in sorted(find_names(post["visible"]))
             for block in blocks[name] if 0 <= days_between(block["date"], post["date"]) <= AFTER_PATCH_DAYS]
    return [Seed("composite", (b["doc_id"], p["doc_id"]), "patch_and_post") for b, p in sample(rng, pairs, n)]


def patch_and_comment_seeds(docs: list[Doc], rng: random.Random, n: int) -> list[Seed]:
    """Правка в патчноуте и комментарий к статье того же патча, где эта сущность упомянута."""
    blocks = entity_blocks_by_name(docs)
    find_names = build_phrase_finder(blocks)
    comments = [d for d in docs if d["source"] == "dota2ru_comment" and len(d["visible"]) >= 80]
    pairs = [(block, c) for c in comments for name in sorted(find_names(c["visible"]))
             for block in blocks[name] if block["patch"] == c["patch"]]
    return [Seed("composite", (b["doc_id"], c["doc_id"]), "patch_and_comment") for b, c in sample(rng, pairs, n)]


def abstract_seeds(docs: list[Doc], rng: random.Random, per_group: int = 5) -> list[Seed]:
    """Группы документов на одну тему: общие изменения крупного патча, комментарии к нему, темы Telegram."""
    def top_longest(items: list[Doc], k: int) -> tuple[str, ...]:
        return tuple(d["doc_id"] for d in sorted(items, key=lambda d: (-len(d["visible"]), d["doc_id"]))[:k])

    general = defaultdict(list)
    comments = defaultdict(list)
    for d in docs:
        if d["source"] == "patchnotes" and d["meta"].get("kind") == "general" and is_major_patch(d["patch"]):
            general[d["patch"]].append(d)
        if d["source"] == "dota2ru_comment" and is_major_patch(d["patch"]):
            comments[d["patch"]].append(d)
    rich_general = sorted(p for p, ds in general.items() if len(ds) >= 3)
    rich_comments = sorted(p for p, ds in comments.items() if len(ds) >= 20)
    seeds = [Seed("abstract", top_longest(general[p], 6), "patch_general")
             for p in sample(rng, rich_general, per_group)]
    seeds += [Seed("abstract", top_longest(comments[p], 8), "patch_reaction")
              for p in sample(rng, rich_comments, per_group)]
    for theme in ABSTRACT_THEMES[:per_group]:
        themed = [d for d in docs if is_topical_post(d) and theme in d["visible"].lower() and len(d["visible"]) >= 150]
        seeds.append(Seed("abstract", tuple(d["doc_id"] for d in sample(rng, themed, 8)), f"theme:{theme}"))
    return seeds


def negative_seeds(plan: dict[str, int]) -> list[Seed]:
    return [Seed("negative", (), kind) for kind, n in plan.items() for _ in range(n)]


def all_seeds(docs: list[Doc], seed: int) -> list[Seed]:
    """Затравки всех типов с запасом ~50% к целевым квотам (15/15/10/10) — часть отсеется на ревью."""
    rng = random.Random(seed)
    return [
        *factual_seeds(docs, rng, n_patch=16, n_post=6),
        *two_patches_seeds(docs, rng, n=8),
        *patch_and_post_seeds(docs, rng, n=7),
        *patch_and_comment_seeds(docs, rng, n=7),
        *abstract_seeds(docs, rng, per_group=5),
        *negative_seeds({"out_of_window": 4, "future": 3, "nonexistent": 3, "false_premise": 4, "other_game": 2}),
    ]
