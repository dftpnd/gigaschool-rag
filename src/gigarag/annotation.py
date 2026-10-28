"""Ручная разметка ответов RAG по шкалам судьи: выборка, страница разметки и хранение оценок.

Разметчик видит ровно то, что видит судья (вопрос, фрагменты, ответ), и те же шкалы `judge.SCALES`;
конфигурация поиска, эталон gold и оценки судьи на странице не показываются, чтобы не подсказывать.
Оценки хранятся в `gold/` как {id ответа: {критерий: балл, "digest": хэш содержимого}}: id — только `config|qid`,
а хэш вопроса, фрагментов и ответа привязывает оценку к тому тексту, который видел разметчик. Если ответ
перегенерирован и изменился, оценка с тем же id считается недействительной.
"""

import hashlib
import html
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

from gigarag.atomic import atomic_write_text
from gigarag.judge import CRITERIA, SCALES, SCORES

Labels = dict[str, dict]  # id → {критерий: балл, "digest": хэш содержимого}
DIGEST = "digest"


def content_digest(record: Mapping) -> str:
    """sha256 того, что видит разметчик: вопрос, фрагменты по порядку и ответ."""
    payload = json.dumps([record["question"], list(record["passages"]), record["answer"]], ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def write_sample(path: Path, records: Sequence[Mapping]) -> None:
    """Выборка для разметки: id, вопрос, фрагменты и ответ каждого ответа."""
    atomic_write_text(path, json.dumps(list(records), ensure_ascii=False, indent=1))


def read_sample(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def clean_scores(scores: Mapping) -> dict[str, int]:
    """Только известные критерии с баллом из шкалы: мусор из формы не попадает в файл оценок."""
    return {c: s for c, s in scores.items() if c in CRITERIA and s in SCORES and not isinstance(s, bool)}


def read_labels(path: Path) -> Labels:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def with_label(labels: Labels, record: Mapping, scores: Mapping) -> Labels:
    """Новые оценки ответа поверх старых — если старые сделаны для того же содержимого; иначе старые
    отбрасываются. Входной словарь не меняется."""
    digest = content_digest(record)
    previous = labels.get(record["id"], {})
    kept = previous if previous.get(DIGEST) == digest else {}
    return {**labels, record["id"]: {**kept, **clean_scores(scores), DIGEST: digest}}


def write_labels(path: Path, labels: Labels) -> None:
    atomic_write_text(path, json.dumps(labels, ensure_ascii=False, indent=1, sort_keys=True))


def valid_labels(labels: Labels, records: Sequence[Mapping]) -> dict[str, dict[str, int]]:
    """Оценки ответов выборки, сделанные для их текущего содержимого: {id: {критерий: балл}}."""
    return {r["id"]: clean_scores(labels[r["id"]]) for r in records
            if labels.get(r["id"], {}).get(DIGEST) == content_digest(r)}


def reuse_labels(pools: Sequence[Labels], records: Sequence[Mapping]) -> Labels:
    """Оценки из прежних файлов разметки для ответов, содержимое которых не изменилось (тот же id и хэш)."""
    return {r["id"]: pool[r["id"]] for pool in reversed(pools) for r in records
            if pool.get(r["id"], {}).get(DIGEST) == content_digest(r)}


def is_complete(labels: Labels, record: Mapping) -> bool:
    return set(valid_labels(labels, [record]).get(record["id"], {})) == set(CRITERIA)


def _scale_html(criterion: str) -> str:
    scale = SCALES[criterion]
    levels = "".join(f"<li><b>{score}</b> — {html.escape(anchor)}</li>"
                     for score, anchor in zip((5, 4, 3, 2, 1), scale.anchors, strict=True))
    return (f"<details class='scale'><summary>{scale.title}: {html.escape(scale.definition)}</summary>"
            f"<ul>{levels}</ul><p>{html.escape(scale.abstention)}</p></details>")


def first_unlabeled(records: Sequence[Mapping], labels: Labels) -> int:
    """Номер (с 1) первого ответа без всех трёх оценок; 1, если размечено всё."""
    return next((n for n, r in enumerate(records, 1) if not is_complete(labels, r)), 1)


def page(records: Sequence[Mapping]) -> str:
    """Страница разметки: один ответ на адрес /1 … /N, три шкалы 1–5; каждая оценка сохраняется сразу."""
    data = json.dumps([{"id": r["id"], "question": r["question"], "passages": r["passages"],
                        "answer": r["answer"]} for r in records], ensure_ascii=False).replace("</", "<\\/")
    # «</» внутри строк JSON закрыл бы <script> раньше времени, если в чанке встретится «</script>»
    scales = "".join(_scale_html(c) for c in CRITERIA)
    criteria = json.dumps([[c, SCALES[c].title] for c in CRITERIA])
    return _PAGE.replace("__DATA__", data).replace("__SCALES__", scales).replace("__CRITERIA__", criteria)


_PAGE = """<!doctype html>
<html lang="ru"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Разметка ответов КТ 5</title>
<style>
:root { --bg:#fafaf7; --fg:#1d1d1b; --muted:#6b6b66; --card:#fff; --line:#e2e1da; --accent:#2f6f4f; --warn:#a4462c; }
@media (prefers-color-scheme: dark) { :root { --bg:#161614; --fg:#ecebe4; --muted:#a09f97; --card:#20201d;
  --line:#3a3a35; --accent:#7cc39c; --warn:#e48b6f; } }
body { margin:0; background:var(--bg); color:var(--fg); font:16px/1.5 system-ui, sans-serif; }
main { max-width: 900px; margin: 0 auto; padding: 16px 16px 120px; }
header { display:flex; gap:12px; align-items:center; flex-wrap:wrap; justify-content:space-between; }
.card { background:var(--card); border:1px solid var(--line); border-radius:10px; padding:16px; margin:12px 0; }
.q { font-size:18px; font-weight:600; }
.passage { border-left:3px solid var(--line); padding:4px 12px; margin:10px 0; white-space:pre-wrap; font-size:14px; }
.answer { white-space:pre-wrap; border-left:3px solid var(--accent); padding:4px 12px; }
.crit { margin:14px 0; } .crit b { display:block; margin-bottom:6px; }
.crit label { display:inline-block; margin-right:6px; }
.crit input { position:absolute; opacity:0; width:1px; height:1px; }
.crit span { display:inline-block; min-width:48px; text-align:center; padding:10px 0; border:1px solid var(--line);
  border-radius:8px; cursor:pointer; font-size:18px; }
.crit input:focus-visible + span { outline:2px solid var(--fg); }
.crit input:checked + span { background:var(--accent); color:var(--bg); border-color:var(--accent); }
.muted { color:var(--muted); } .err { color:var(--warn); } .ok { color:var(--accent); }
.dots { display:flex; flex-wrap:wrap; gap:4px; } .dot { width:26px; height:26px; border-radius:5px; font-size:12px;
  display:flex; align-items:center; justify-content:center; border:1px solid var(--line); color:var(--fg);
  text-decoration:none; }
.dot.done { background:var(--accent); color:var(--bg); } .dot.cur { outline:2px solid var(--fg); }
details.scale { margin:6px 0; } details.scale ul { margin:4px 0; }
nav.bar { position:fixed; left:0; right:0; bottom:0; background:var(--card); border-top:1px solid var(--line);
  padding:10px 16px; display:flex; gap:10px; align-items:center; justify-content:center; flex-wrap:wrap; }
nav.bar a { font-size:17px; padding:12px 22px; border-radius:10px; border:1px solid var(--line); color:var(--fg);
  text-decoration:none; background:var(--bg); } nav.bar a.primary { background:var(--accent); color:var(--bg);
  border-color:var(--accent); } nav.bar a.off { opacity:.35; pointer-events:none; }
#status { min-width:12em; text-align:center; }
</style></head><body><main>
<header><h1 style="margin:8px 0">Разметка ответов</h1><span id="progress" class="muted"></span></header>
<div class="dots" id="dots"></div>
<div class="card"><b>Главное</b><ul>
<li>Оценивайте ответ <b>относительно фрагментов</b>, а не по своему знанию игры.</li>
<li><b>Достоверность</b> — только «нет ли в ответе того, чего нет во фрагментах». Не «верно ли в игре»
и не «полезен ли ответ».</li>
<li><b>Ответ «Не знаю…»</b>: достоверность — 5 (ничего не утверждает). Релевантность и использование контекста —
5 и 5, если ответа во фрагментах нет, и 1 и 1, если ответ во фрагментах есть.</li>
<li>Ответил на часть вопроса и прямо сказал, какой части нет, — релевантность 3.</li>
<li>Каждая оценка сохраняется сразу при клике; у каждого ответа свой адрес — его можно открыть снова.</li>
</ul></div>
<details class="card"><summary><b>Шкалы</b> (те же, что у судьи)</summary>__SCALES__</details>
<div class="card"><div class="muted" id="pos"></div><p class="q" id="question"></p>
<div><b>Ответ системы</b><div class="answer" id="answer"></div></div>
<div id="form"></div>
<details open><summary><b>Фрагменты контекста</b></summary><div id="passages"></div></details></div>
</main>
<nav class="bar"><a id="prev">← Назад</a><span id="status" class="muted"></span>
<a id="next" class="primary">Дальше →</a></nav>
<script>
const DATA = __DATA__; const CRITERIA = __CRITERIA__;
const $ = id => document.getElementById(id);
const n = Math.min(Math.max(parseInt(location.pathname.slice(1)) || 1, 1), DATA.length);
const r = DATA[n - 1];
let labels = {};
function done(id) { return CRITERIA.every(([c]) => labels[id] && labels[id][c]); }
function progress() {
  $("progress").textContent = `размечено ${DATA.filter(x => done(x.id)).length} из ${DATA.length}`;
  $("dots").innerHTML = DATA.map((x, k) => `<a href="/${k + 1}" class="dot${done(x.id) ? " done" : ""}` +
    `${k + 1 === n ? " cur" : ""}">${k + 1}</a>`).join("");
  $("status").innerHTML = done(r.id) ? '<span class="ok">✓ все три оценки сохранены</span>'
    : `<span class="muted">осталось оценок: ${CRITERIA.filter(([c]) => !(labels[r.id] || {})[c]).length}</span>`;
}
async function post(criterion, score) {
  try { const res = await fetch("/labels", { method: "POST", headers: {"Content-Type": "application/json"},
      body: JSON.stringify({ id: r.id, scores: { [criterion]: score } }) });
    if (!res.ok) throw new Error(res.status); labels = await res.json(); progress();
  } catch (e) { $("status").innerHTML = `<span class="err">Не сохранилось: ${e.message}</span>`; }
}
document.title = `Ответ ${n} из ${DATA.length} — разметка`;
$("pos").textContent = `Ответ ${n} из ${DATA.length}`;
$("question").textContent = r.question; $("answer").textContent = r.answer;
r.passages.forEach((p, k) => { const d = document.createElement("div"); d.className = "passage";
  d.textContent = `[${k + 1}] ${p}`; $("passages").appendChild(d); });
$("prev").href = `/${n - 1}`; $("next").href = `/${n + 1}`;
if (n === 1) $("prev").classList.add("off");
if (n === DATA.length) $("next").classList.add("off");
fetch("/labels").then(res => res.json()).then(l => {
  labels = l; const mine = labels[r.id] || {};
  $("form").innerHTML = CRITERIA.map(([c, title]) => `<div class="crit"><b>${title}</b>` +
    [1, 2, 3, 4, 5].map(s => `<label><input type="radio" name="${c}" value="${s}"` +
    (mine[c] === s ? " checked" : "") + `><span>${s}</span></label>`).join("") + `</div>`).join("");
  document.querySelectorAll(".crit input").forEach(el => el.onchange = () => post(el.name, +el.value));
  progress();
});
</script></body></html>"""
