"""Ручная разметка ответов RAG по шкалам судьи: выборка, страница разметки и хранение оценок.

Разметчик видит ровно то, что видит судья (вопрос, фрагменты, ответ), и те же шкалы `judge.SCALES`;
конфигурация поиска, эталон gold и оценки судьи на странице не показываются, чтобы не подсказывать.
Оценки хранятся в `gold/` как {id ответа: {критерий: балл}} — их сравнивает с судьёй weighted kappa.
"""

import html
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

from gigarag.atomic import atomic_write_text
from gigarag.judge import CRITERIA, SCALES, SCORES

Labels = dict[str, dict[str, int]]


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


def with_label(labels: Labels, record_id: str, scores: Mapping) -> Labels:
    """Новые оценки ответа поверх старых; входной словарь не меняется."""
    return {**labels, record_id: {**labels.get(record_id, {}), **clean_scores(scores)}}


def write_labels(path: Path, labels: Labels) -> None:
    atomic_write_text(path, json.dumps(labels, ensure_ascii=False, indent=1, sort_keys=True))


def is_complete(labels: Labels, record_id: str) -> bool:
    return set(labels.get(record_id, {})) == set(CRITERIA)


def _scale_html(criterion: str) -> str:
    scale = SCALES[criterion]
    levels = "".join(f"<li><b>{score}</b> — {html.escape(anchor)}</li>"
                     for score, anchor in zip((5, 4, 3, 2, 1), scale.anchors, strict=True))
    return (f"<details class='scale'><summary>{scale.title}: {html.escape(scale.definition)}</summary>"
            f"<ul>{levels}</ul><p>{html.escape(scale.abstention)}</p></details>")


def page(records: Sequence[Mapping]) -> str:
    """Страница разметки: по одному ответу, три шкалы 1–5, сохранение на сервер по кнопке."""
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
main { max-width: 900px; margin: 0 auto; padding: 16px; }
header { display:flex; gap:12px; align-items:center; flex-wrap:wrap; justify-content:space-between; }
.card { background:var(--card); border:1px solid var(--line); border-radius:10px; padding:16px; margin:12px 0; }
.q { font-size:18px; font-weight:600; }
.passage { border-left:3px solid var(--line); padding:4px 12px; margin:10px 0; white-space:pre-wrap; font-size:14px; }
.answer { white-space:pre-wrap; border-left:3px solid var(--accent); padding:4px 12px; }
.crit { margin:14px 0; } .crit b { display:block; margin-bottom:6px; }
.crit label { display:inline-block; margin-right:6px; }
.crit input { display:none; }
.crit span { display:inline-block; min-width:40px; text-align:center; padding:8px 0; border:1px solid var(--line);
  border-radius:8px; cursor:pointer; }
.crit input:checked + span { background:var(--accent); color:var(--bg); border-color:var(--accent); }
button { font:inherit; padding:8px 16px; border-radius:8px; border:1px solid var(--line); background:var(--card);
  color:var(--fg); cursor:pointer; }
button.primary { background:var(--accent); color:var(--bg); border-color:var(--accent); }
.muted { color:var(--muted); } .status { min-height:1.5em; } .err { color:var(--warn); }
.dots { display:flex; flex-wrap:wrap; gap:4px; } .dot { width:22px; height:22px; border-radius:5px; font-size:11px;
  display:flex; align-items:center; justify-content:center; border:1px solid var(--line); cursor:pointer; }
.dot.done { background:var(--accent); color:var(--bg); } .dot.cur { outline:2px solid var(--fg); }
details.scale { margin:6px 0; } details.scale ul { margin:4px 0; }
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
</ul></div>
<div class="card"><b>Шкалы</b> (те же, что у судьи)__SCALES__</div>
<div class="card"><div class="muted" id="pos"></div><p class="q" id="question"></p>
<div><b>Ответ системы</b><div class="answer" id="answer"></div></div>
<div id="form"></div>
<div style="display:flex; gap:8px; flex-wrap:wrap"><button id="prev">← Назад</button>
<button class="primary" id="save">Сохранить и дальше →</button></div>
<p class="status" id="status"></p>
<details open><summary><b>Фрагменты контекста</b></summary><div id="passages"></div></details></div>
</main><script>
const DATA = __DATA__; const CRITERIA = __CRITERIA__;
let labels = {}; let i = 0;
const $ = id => document.getElementById(id);
function done(id) { return CRITERIA.every(([c]) => labels[id] && labels[id][c]); }
function render() {
  const r = DATA[i];
  $("pos").textContent = `Ответ ${i + 1} из ${DATA.length}`;
  $("question").textContent = r.question; $("answer").textContent = r.answer;
  $("passages").innerHTML = ""; r.passages.forEach((p, n) => { const d = document.createElement("div");
    d.className = "passage"; d.textContent = `[${n + 1}] ${p}`; $("passages").appendChild(d); });
  $("form").innerHTML = CRITERIA.map(([c, title]) => `<div class="crit"><b>${title}</b>` +
    [1, 2, 3, 4, 5].map(s => `<label><input type="radio" name="${c}" value="${s}"` +
    ((labels[r.id] || {})[c] === s ? " checked" : "") + `><span>${s}</span></label>`).join("") + `</div>`).join("");
  const total = DATA.filter(r => done(r.id)).length;
  $("progress").textContent = `размечено ${total} из ${DATA.length}`;
  $("dots").innerHTML = DATA.map((r, n) => `<div class="dot${done(r.id) ? " done" : ""}${n === i ? " cur" : ""}"` +
    ` data-n="${n}">${n + 1}</div>`).join("");
  document.querySelectorAll(".dot").forEach(d => d.onclick = () => { i = +d.dataset.n; render(); });
  document.querySelectorAll(".crit input").forEach(el => el.onchange = async () => {
    if (await post(checked())) { refreshProgress(); $("status").textContent = "Сохранено"; } });
  $("status").textContent = ""; window.scrollTo(0, 0);
}
function checked() {
  const scores = {};
  for (const [c] of CRITERIA) { const el = document.querySelector(`input[name="${c}"]:checked`);
    if (el) scores[c] = +el.value; }
  return scores;
}
async function post(scores) {
  // каждая отмеченная оценка уходит на сервер сразу: переход к другому ответу ничего не теряет
  try { const res = await fetch("labels", { method: "POST", headers: {"Content-Type": "application/json"},
      body: JSON.stringify({ id: DATA[i].id, scores }) });
    if (!res.ok) throw new Error(res.status); labels = await res.json(); return true;
  } catch (e) { $("status").innerHTML = `<span class="err">Не сохранилось: ${e.message}</span>`; return false; }
}
function refreshProgress() {
  const total = DATA.filter(r => done(r.id)).length;
  $("progress").textContent = `размечено ${total} из ${DATA.length}`;
  document.querySelectorAll(".dot").forEach(d => d.classList.toggle("done", done(DATA[+d.dataset.n].id)));
}
async function save() {
  const scores = checked();
  if (Object.keys(scores).length < CRITERIA.length) {
    $("status").innerHTML = '<span class="err">Поставьте все три оценки</span>'; return; }
  if (!await post(scores)) return;
  const next = DATA.findIndex((r, n) => n > i && !done(r.id));
  i = next >= 0 ? next : Math.min(i + 1, DATA.length - 1); render();
  if (DATA.every(r => done(r.id))) $("status").textContent = "Все ответы размечены, оценки сохранены на сервере.";
}
$("save").onclick = save; $("prev").onclick = () => { i = Math.max(0, i - 1); render(); };
fetch("labels").then(r => r.json()).then(l => { labels = l; const first = DATA.findIndex(r => !done(r.id));
  i = first >= 0 ? first : 0; render(); });
</script></body></html>"""
