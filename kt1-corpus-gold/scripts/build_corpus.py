"""Сборка сырого корпуса КТ 1 из скачанных источников в data/corpus.jsonl.

Ничего не чистим: HTML-разметка статей и комментариев, эмодзи и ссылки
в постах Telegram остаются как есть — чистка и её эффект измеряются в КТ 2.

Окно — патчи 7.34+ (с августа 2023): за этот период есть все четыре источника.
Из Telegram берутся посты по теме патчей и механик плюс случайный офтоп
в доле OFFTOPIC_SHARE от тематических — как реалистичный шум.
"""

import hashlib
import json
import random
import re
from collections import Counter
from datetime import datetime, timezone

from selectolax.lexbor import LexborHTMLParser

from common import RAW

OUT = RAW.parent / "corpus.jsonl"
SINCE_PATCH = "7.34"
OFFTOPIC_SHARE = 0.2
SEED = 42

# Тематические маркеры для постов Telegram: термины патчей и баланса.
# Имена героев и предметов добавляются из справочников dota2.com.
TERMS = [
    r"патч", r"нерф", r"понерф", r"баф", r"апнул", r"ап\b", r"ребаланс", r"обновлени",
    r"\b7\.\d\d[a-z]?\b", r"аспект", r"фасет", r"врожд[её]нн", r"талант", r"мет[аеуы]\b",
    r"имб", r"предмет", r"нейтрал", r"патчнот", r"изменени", r"способност", r"ульт",
]
# Русские и сленговые имена героев, которые чаще встречаются в постах, чем английские.
RU_HEROES = [
    "пудж", "инвокер", "войд", "фейсл", "сф", "шадоу финд", "невермор", "антимаг", "мипо",
    "тинкер", "течис", "пак", "кунка", "акс", "лион", "лина", "зевс", "снайпер", "дровка",
    "морф", "террор", "медуза", "спектра", "фантомка", "пл", "наги", "вайпер", "дазл", "опыт",
    "рики", "слардар", "бист", "бета", "урса", "трент", "брюмастер", "батрайдер",
    "энигма", "марси", "мурлок", "слайс", "джагер", "джаггернаут", "кез", "рингмастер",
    "хускар", "сларк", "тролль", "бристл", "тимбер", "лешрак", "шторм", "эмбер", "земля",
    "аркан", "мидас", "блинк", "бкб", "аганим", "аган", "шард", "рапира", "дагон",
]


def ts_date(ts: int) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).date().isoformat()


def patch_key(ver: str) -> tuple:
    m = re.match(r"7\.(\d+)([a-z]?)", ver)
    return (int(m.group(1)), m.group(2)) if m else (0, "")


# ---------- патчноуты ----------

def note_lines(notes, indent_base=0) -> list[str]:
    out = []
    for n in notes or []:
        text = n.get("note", "")
        if n.get("hide_dot"):
            out.append(text)
        else:
            out.append("  " * (n.get("indent_level", 1) - 1 + indent_base) + "- " + text)
    return out


def ability_block(ab, names) -> list[str]:
    name = names.get(ab.get("ability_id"), f"ability_{ab.get('ability_id')}")
    return [f"{name}:"] + note_lines(ab.get("ability_notes"))


def hero_text(h, names) -> list[str]:
    lines = note_lines(h.get("hero_notes"))
    for ab in h.get("abilities") or []:
        lines += ability_block(ab, names)
    for sub in h.get("subsections") or []:
        lines.append(f"[{sub.get('title', '')}]")
        lines += note_lines(sub.get("general_notes"))
        for ab in sub.get("abilities") or []:
            lines += ability_block(ab, names)
    if h.get("talent_notes"):
        lines.append("Таланты:")
        lines += note_lines(h["talent_notes"])
    return lines


def patch_docs():
    d = RAW / "patches"
    heroes = {h["id"]: h["name_loc"] for h in json.loads((d / "_herolist.json").read_text())["result"]["data"]["heroes"]}
    names = {}
    for f in ("_itemlist.json", "_abilitylist.json"):
        for a in json.loads((d / f).read_text())["result"]["data"]["itemabilities"]:
            names[a["id"]] = a["name_loc"] or a["name_english_loc"]
    patches = json.loads((d / "_patchnoteslist.json").read_text())
    for p in patches:
        ver = p["patch_number"]
        if patch_key(ver) < patch_key(SINCE_PATCH):
            continue
        data = json.loads((d / f"{ver}.json").read_text())
        date = ts_date(p["patch_timestamp"])
        url = f"https://www.dota2.com/patches/{ver}"
        used = Counter()

        def doc(kind, entity, lines):
            # одна сущность может встретиться в патче дважды — id делаем уникальным
            used[(kind, entity)] += 1
            n = used[(kind, entity)]
            return {
                "doc_id": f"patch:{ver}:{kind}:{entity}" + (f"#{n}" if n > 1 else ""),
                "source": "patchnotes", "date": date, "patch": ver, "url": url,
                "title": f"Патч {ver} — {entity}",
                "text": f"Патч {ver} — {entity}\n" + "\n".join(lines),
                "meta": {"kind": kind, "entity": entity},
            }

        for i, g in enumerate(data.get("general_notes") or []):
            title = re.sub(r"<[^>]+>", "", g.get("title") or "") or "Общие изменения"
            yield doc("general", f"{title} #{i}" if i else title, note_lines(g.get("generic")))
        for kind in ("items", "neutral_items"):
            for it in data.get(kind) or []:
                # ability_id -1 — общие примечания раздела, без конкретного предмета
                if it["ability_id"] == -1:
                    name = "Нейтральные предметы: общее" if kind == "neutral_items" else "Предметы: общее"
                else:
                    name = names.get(it["ability_id"], f"item_{it['ability_id']}")
                yield doc(kind, name, note_lines(it.get("ability_notes")))
        for h in data.get("heroes") or []:
            yield doc("hero", heroes.get(h["hero_id"], f"hero_{h['hero_id']}"), hero_text(h, names))
        for c in data.get("neutral_creeps") or []:
            yield doc("neutral_creep", c.get("localized_name") or c["name"], note_lines(c.get("neutral_creep_notes")))


# ---------- dota2.ru ----------

def dota2ru_docs(patch_dates):
    d = RAW / "dota2ru"
    for f in sorted(d.glob("*.html")):
        if "__p" in f.stem:
            continue
        m = re.search(r"-7-(\d{2}[a-z]?)", f.stem)
        ver = f"7.{m.group(1)}" if m else None
        if not ver or patch_key(ver) < patch_key(SINCE_PATCH):
            continue
        pages = [f] + sorted(d.glob(f"{f.stem}__p*.html"), key=lambda p: int(p.stem.rsplit("p", 1)[1]))
        t = LexborHTMLParser(f.read_text(encoding="utf-8"))
        url = f"https://dota2.ru/news/{f.stem}"
        h1 = t.css_first("h1")
        body = t.css_first("[itemprop=articleBody]")
        tm = t.css_first("time[datetime]")
        date = tm.attributes["datetime"][:10] if tm else patch_dates.get(ver)
        title = h1.text(strip=True) if h1 else f.stem
        aid = f.stem.split("-", 1)[0]
        if body:
            yield {
                "doc_id": f"dota2ru:article:{aid}", "source": "dota2ru_article", "date": date,
                "patch": ver, "url": url, "title": title, "text": body.html,
                "meta": {},
            }
        for page in pages:
            pt = t if page == f else LexborHTMLParser(page.read_text(encoding="utf-8"))
            for c in pt.css(".global-content__comment-wrap"):
                msg = c.css_first(".global-content__comment-user--message")
                if not msg:
                    continue
                rep = c.css_first("[data-reply-answer]")
                # без id на странице — стабильный digest, а не hash(): тот рандомизирован между запусками
                cid = rep.attributes["data-reply-answer"] if rep else \
                    "h" + hashlib.blake2b(f"{aid}\n{msg.html}".encode(), digest_size=8).hexdigest()
                ctime = c.css_first("[data-time]")
                author = c.css_first(".global-content__comment-name")
                yield {
                    "doc_id": f"dota2ru:comment:{cid}", "source": "dota2ru_comment",
                    "date": ts_date(int(ctime.attributes["data-time"])) if ctime else date,
                    "patch": ver, "url": url, "title": f"Комментарий к «{title}»",
                    "text": msg.html,
                    "meta": {"article_id": aid, "author": author.text(strip=True) if author else None,
                             "has_quote": "blockquote" in msg.html or "bbCodeQuote" in msg.html},
                }


# ---------- Telegram ----------

def tg_posts(channel):
    """Посты канала, отсортированные по doc_id.

    Соседние страницы ленты пересекаются; при повторе берётся версия поста со страницы
    с меньшим номером — порядок обхода фиксирован, поэтому результат не зависит от ФС.
    """
    seen = {}
    pages = sorted((RAW / "telegram" / channel).glob("*.html"), key=lambda p: int(p.stem))
    for f in pages:
        t = LexborHTMLParser(f.read_text(encoding="utf-8"))
        for m in t.css(".tgme_widget_message[data-post]"):
            pid = m.attributes["data-post"]
            if pid in seen:
                continue
            txt = m.css_first(".tgme_widget_message_text")
            tm = m.css_first("time[datetime]")
            fwd = m.css_first(".tgme_widget_message_forwarded_from_name")
            views = m.css_first(".tgme_widget_message_views")
            seen[pid] = {
                "doc_id": f"tg:{pid}", "source": f"tg_{channel}",
                "date": tm.attributes["datetime"][:10] if tm else None, "patch": None,
                "url": f"https://t.me/{pid}", "title": f"@{channel}",
                "text": txt.html if txt else "",
                "meta": {
                    "forwarded_from": fwd.text(strip=True) if fwd else None,
                    "has_photo": bool(m.css_first(".tgme_widget_message_photo_wrap")),
                    "has_video": bool(m.css_first(".tgme_widget_message_video_player, .tgme_widget_message_roundvideo_player")),
                    "views": views.text(strip=True) if views else None,
                },
            }
    return sorted(seen.values(), key=lambda p: p["doc_id"])


def tg_docs(since_date, topic_re):
    for ch in sorted(p.name for p in (RAW / "telegram").iterdir()):
        posts = [p for p in tg_posts(ch) if p["date"] and p["date"] >= since_date and p["text"]]
        plain = lambda p: re.sub(r"<[^>]+>", " ", p["text"]).lower()
        topical = [p for p in posts if topic_re.search(plain(p))]
        rest = [p for p in posts if not topic_re.search(plain(p))]
        # свой генератор на канал и популяция в фиксированном порядке (по doc_id) —
        # выборка офтопа зависит только от seed и набора постов
        rng = random.Random(f"{SEED}:{ch}")
        off = rng.sample(rest, min(len(rest), int(len(topical) * OFFTOPIC_SHARE)))
        for p in topical:
            p["meta"]["topical"] = True
        for p in off:
            p["meta"]["topical"] = False
        print(f"  {ch}: постов в окне {len(posts)}, тематических {len(topical)}, офтоп {len(off)}")
        yield from sorted(topical + off, key=lambda p: (p["date"], p["doc_id"]))


def main():
    plist = json.loads((RAW / "patches" / "_patchnoteslist.json").read_text())
    patch_dates = {p["patch_number"]: ts_date(p["patch_timestamp"]) for p in plist}
    since_date = patch_dates[SINCE_PATCH]

    d = RAW / "patches"
    names = [h["name_loc"] for h in json.loads((d / "_herolist.json").read_text())["result"]["data"]["heroes"]]
    names += [a["name_loc"] for a in json.loads((d / "_itemlist.json").read_text())["result"]["data"]["itemabilities"]
              if a["name_loc"] and not a["name_loc"].startswith("Recipe") and len(a["name_loc"]) > 3]
    topic_re = re.compile(
        "|".join(TERMS + [rf"\b{re.escape(w)}\b" for w in RU_HEROES] + [rf"\b{re.escape(n.lower())}\b" for n in sorted(set(names))]),
        re.I,
    )

    docs = [doc for gen in (patch_docs(), dota2ru_docs(patch_dates), tg_docs(since_date, topic_re)) for doc in gen]
    # doc_id — ключ для gold-набора и кэшей, поэтому дубли недопустимы
    dup = [k for k, v in Counter(d["doc_id"] for d in docs).items() if v > 1]
    if dup:
        raise SystemExit(f"неуникальные doc_id ({len(dup)}): {dup[:5]}")
    with OUT.open("w", encoding="utf-8") as out:
        for doc in docs:
            out.write(json.dumps(doc, ensure_ascii=False) + "\n")
    digest = hashlib.sha256(OUT.read_bytes()).hexdigest()
    for k, v in Counter(d["source"] for d in docs).items():
        print(f"{k:18} {v}")
    print("всего:", len(docs), "→", OUT)
    print("sha256:", digest)


if __name__ == "__main__":
    main()
