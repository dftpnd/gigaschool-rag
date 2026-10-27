# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.6
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # КТ 2 — предобработка, чанкинг и метаданные
#
# Отправная точка — сырой корпус КТ 1 (19 665 документов, HTML и эмодзи на месте) и золотой набор
# из 50 вопросов. Решения и выводы — в `reports/kt2.md`, здесь — эксперименты и цифры.
#
# **Как считается качество.** Gold размечен на уровне отрывков КТ 1, у каждого доказательства есть дословная
# цитата. Чанк содержит доказательство, если в нём лежит непрерывный кусок цитаты не короче половины её слов
# (`gigarag.relevance`). Поэтому разметка переносится на любое разбиение и переживает чистку, а удалённый
# дубликат передаёт свои доказательства документу, который его заменил.
#
# - **Hit@k** — в топ-k есть хотя бы одно доказательство вопроса;
# - **R@k** — доля доказательств вопроса (оценка 2), найденных в топ-k;
# - **P@k** — доля чанков топ-k с доказательством (оценка снизу: разметка неполна);
# - **MRR** — по первому релевантному чанку в топ-10.
#
# Негативные вопросы доказательств не имеют, в метриках ретрива их нет: остаётся 29 вопросов validation
# и 11 test. Все решения принимаются по validation; один вопрос — это 3,4 п. п. Hit@k, поэтому рядом
# со средними показано, сколько вопросов конфигурация улучшила и ухудшила.
#
# Каждая конфигурация прогоняется двумя поисковиками: **BM25** (`rank-bm25`, k1 = 1,5, b = 0,75, слова
# в нижнем регистре) и **dense BGE-M3** (косинус, fp16 на GPU, до 512 токенов).

# %%
from itertools import pairwise
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import gigarag.chunking
import gigarag.cleaning
import gigarag.noise
import gigarag.patterns
from gigarag import cleaning, llm, metrics, noise, tokens
from gigarag.cache import CacheStore, code_fingerprint, load_or_compute, read_parquet, write_parquet
from gigarag.chunking import ChunkerResources, ChunkerSpec, build_chunker, chunk_length_stats
from gigarag.corpus import Document, read_jsonl
from gigarag.embeddings import load_sentence_model
from gigarag.experiment import Workbench, docs_fingerprint
from gigarag.gold import load_gold

pd.set_option("display.max_colwidth", 120)
pd.set_option("display.width", 200)
plt.rcParams["figure.dpi"] = 110

ROOT = Path.cwd().parent  # ноутбук исполняется из notebooks/
DATA = ROOT / "data"
rows, CORPUS_SHA = read_jsonl(DATA / "corpus.jsonl")
raw_docs = tuple(Document(**r) for r in rows)
items = load_gold(ROOT / "gold" / "gold.jsonl")
cache = CacheStore(DATA / "cache" / "kt2", CORPUS_SHA)

BGE_M3, BGE_M3_REV = "BAAI/bge-m3", "5617a9f61b028005a4858fdac845db406aefb181"
E5_SMALL, E5_SMALL_REV = "intfloat/multilingual-e5-small", "614241f622f53c4eeff9890bdc4f31cfecc418b3"
CHONKY, CHONKY_REV = "mirth/chonky_mmbert_small_multilingual_1", "4a92de5574053676cbc5fd06bac86a6e13588000"
MAX_SEQ = 512

tokenizer = tokens.load_tokenizer(BGE_M3, BGE_M3_REV)
encoder = load_sentence_model(BGE_M3, BGE_M3_REV, max_seq_length=MAX_SEQ, half=True)
bench = Workbench(
    cache=cache, items=items, tokenizer=tokenizer, encoder=encoder,
    encoder_params={"model": BGE_M3, "revision": BGE_M3_REV, "max_seq": MAX_SEQ, "half": True},
    code=code_fingerprint(gigarag.patterns, gigarag.cleaning, gigarag.chunking),
)
resources = ChunkerResources(tokenizer)
print(len(raw_docs), "документов, sha256 корпуса:", CORPUS_SHA[:16], "| вопросов:", len(items))

# %% [markdown]
# ## 1. Baseline: сырой корпус, TokenChunker(512)
#
# Документы режутся токенами BGE-M3 по 512 без перекрытия — ровно контекст эмбеддера. Никакой чистки:
# HTML-разметка статей и постов индексируется как есть.

# %%
TOKEN_512 = ChunkerSpec("token", chunk_size=512)


def token_chunks(docs):
    return bench.chunks(docs, TOKEN_512.params(), lambda: build_chunker(TOKEN_512, resources))


raw_chunks, raw_tokens, raw_chunk_seconds = token_chunks(raw_docs)
baseline = bench.run("raw", raw_chunks)
pd.concat({split: metrics.comparison_table(baseline.per_question, split, metrics.METRIC_COLUMNS)
           for split in ("validation", "test")})

# %% [markdown]
# Baseline по типам вопросов (validation):

# %%
metrics.comparison_table(baseline.per_question.assign(config=baseline.per_question["qtype"]))

# %% [markdown]
# ## 2. Предобработка
#
# Шаги применяются по очереди (`gigarag.cleaning.PIPELINE`), после каждого корпус заново режется
# TokenChunker(512) и индексируется. Так эффект каждого шага виден отдельно.
#
# - `html` — у статей только тело (`articleBody`), блоки через пустую строку; у остальных теги сняты,
#   `<br>` и концы блоков стали переводом строки, сущности декодированы;
# - `unicode` — NFKC, невидимые символы (мягкий перенос, zero-width, селекторы эмодзи), пробелы внутри строк,
#   лишние пустые строки;
# - `urls` — ссылки удалены;
# - `emoji` — эмодзи и оформительские символы (•, ►, ✔, `|`, `#`, `@`) удалены; пунктуация и `% + / = < > № ° ×`
#   оставлены — они нужны числам патчноутов;
# - `signatures` — подписи каналов Telegram: последняя строка поста, повторённая в канале ≥ 20 раз
#   (`@Dota2`, «Gabe News | Dota 2»);
# - `exact_dup` — пустые документы (медиа без подписи) и точные копии; из копий остаётся самая ранняя;
# - `near_dup` — почти-дубликаты как в EDA: кандидаты MinHash LSH (0,5), подтверждение Jaccard ≥ 0,8
#   по 3-граммам слов; остаётся самый длинный.

# %%
stages = [("raw", cleaning.CleanCorpus(raw_docs)), *cleaning.run_stages(cleaning.CleanCorpus(raw_docs),
                                                                         cleaning.PIPELINE)]
offsets = tokens.offsets_fn(tokenizer)
NOISE_CODE = code_fingerprint(gigarag.patterns, gigarag.noise, gigarag.cleaning)


def class_counts(docs) -> pd.DataFrame:
    """Токены BGE-M3 каждого класса (слово, число, HTML, URL, эмодзи…) по документам."""
    return load_or_compute(
        cache.path("class_counts", "parquet", docs=docs_fingerprint(docs), code=NOISE_CODE, tokenizer=BGE_M3_REV),
        lambda: pd.DataFrame(np.vstack([noise.token_class_counts(d.text, offsets) for d in docs]),
                             columns=noise.CLASS_NAMES).assign(source=[d.source for d in docs]),
        load=read_parquet, save=write_parquet,
    )


stage_rows, stage_results = [], {}
for name, state in stages:
    chunks, chunk_tokens, _ = token_chunks(state.docs)
    stage_results[name] = bench.run(name, chunks, state.aliases)
    counts = class_counts(state.docs)
    shares = noise.class_shares_percent(counts, by="source")
    stage_rows.append({"шаг": name, "документов": len(state.docs),
                       "токенов, млн": counts[noise.CONTENT_CLASSES].sum().sum() / 1e6,
                       "шум, %": noise.removable_noise_percent(shares)["ВСЕГО"], "чанков": len(chunks),
                       "токенов в чанке": chunk_tokens.mean()})
stage_table = pd.DataFrame(stage_rows).set_index("шаг").round(2)
stage_per_q = pd.concat([r.per_question for r in stage_results.values()])
stage_metrics = metrics.comparison_table(stage_per_q)
stage_table.join(stage_metrics.set_axis([" ".join(c) for c in stage_metrics.columns], axis=1))

# %% [markdown]
# «Шум» — доля токенов удаляемых классов (HTML, URL, эмодзи, оформительские символы) среди всех непробельных
# токенов BGE-M3, как в EDA КТ 1. Сколько вопросов изменил каждый шаг (R@5 относительно предыдущего шага):

# %%
pd.concat({name: metrics.paired_changes(stage_per_q, prev, name)
           for (prev, _), (name, _) in pairwise(stages)}, names=["шаг"])

# %% [markdown]
# Дедупликация: сколько отсеяно и куда ушли доказательства gold.

# %%
dedup_rows = []
for (_, before), (name, after) in pairwise(stages[-3:]):
    removed = len(before.docs) - len(after.docs)
    empty = sum(not d.text.strip() for d in before.docs) if name == "exact_dup" else 0
    dedup_rows.append({"шаг": name, "было": len(before.docs), "удалено": removed, "из них пустых": empty,
                       "удалено, %": removed / len(before.docs) * 100})
removed_total = len(raw_docs) - len(stages[-1][1].docs)
print(f"всего отсеяно {removed_total} документов ({removed_total / len(raw_docs):.1%} корпуса)")
pd.DataFrame(dedup_rows).set_index("шаг").round(2)

# %% [markdown]
# Контроль для шага `signatures`: весь конвейер без него. Если без подписей поиск хуже, это не значит, что
# подписи полезны. Разбор выдачи показал два механических эффекта BM25: посты становятся короче, и нормировка
# длины (b = 0,75) поднимает их над патчноутами; слова «Dota 2» из 10 тыс. подписей получают высокий IDF.
# Решение принимается по validation сравнением двух полных конвейеров.

# %%
no_signatures = [s for s in cleaning.PIPELINE if s.name != "signatures"]
control_state = cleaning.run_stages(cleaning.CleanCorpus(raw_docs), no_signatures)[-1][1]
control = bench.run("все шаги без signatures", token_chunks(control_state.docs)[0], control_state.aliases)
signature_check = pd.concat([stage_results["near_dup"].per_question.assign(config="все шаги"), control.per_question])
display(metrics.comparison_table(signature_check, columns=metrics.METRIC_COLUMNS))
metrics.paired_changes(signature_check, "все шаги без signatures", "все шаги")

# %% [markdown]
# ## 3. Чанкеры Chonkie
#
# Корпус — после предобработки **без шага `signatures`**: контроль выше показал, что без него validation лучше
# по всем метрикам обоих поисковиков. Настройки чанкеров фиксированы и «разумны»,
# одинаковы для всех: предел 512 токенов BGE-M3, без перекрытия. Семантические чанкеры ищут границы
# по эмбеддингам BGE-M3 (той же модели, что ищет), порог 0,5 — середина сетки ablation (раздел 4).
#
# - **Token** — ровно по 512 токенов.
# - **Sentence** — предложения (`. ! ?` и перевод строки) упаковываются до 512 токенов.
# - **Recursive** — абзацы → предложения → знаки → слова, до 512.
# - **Semantic** — разрез в локальных минимумах сглаженной близости «окно из 3 предложений ↔ следующее».
#   В Chonkie 1.7 `threshold` — не косинус, а доля оставляемых минимумов: 0,1 — только самые глубокие провалы,
#   0,9 — почти все.
# - **SDPM** — Semantic с `skip_window=1`: группа сливается с соседней через одну, если косинус их эмбеддингов
#   ≥ `threshold` (в Chonkie 1.7 SDPMChunker стал режимом SemanticChunker).
# - **Neural** — NeuralChunker на `chonky_mmbert_small_multilingual_1`: токен-классификатор границ. Модель
#   обучена на европейских языках без русского — проверяем, переносится ли.

# %%
from gigarag.chunking import neural_chunker
from gigarag.embeddings import memo_embeddings

clean = control_state  # все шаги, кроме signatures
e5_small = load_sentence_model(E5_SMALL, E5_SMALL_REV, max_seq_length=MAX_SEQ, half=True)
boundary_embeddings = {BGE_M3: memo_embeddings(encoder), E5_SMALL: memo_embeddings(e5_small, prefix="query: ")}
resources = ChunkerResources(tokenizer, embeddings=boundary_embeddings.__getitem__,
                             neural_model=lambda: neural_chunker(CHONKY, CHONKY_REV))
chunker_specs = [
    TOKEN_512,
    ChunkerSpec("sentence"),
    ChunkerSpec("recursive"),
    ChunkerSpec("semantic", threshold=0.5, embedder=BGE_M3),
    ChunkerSpec("semantic", threshold=0.5, skip_window=1, embedder=BGE_M3),
    ChunkerSpec("neural", chunk_size=0),
]


def run_chunker(spec: ChunkerSpec, docs=clean.docs, aliases=clean.aliases):
    chunks, chunk_tokens, seconds = bench.chunks(docs, spec.params(), lambda: build_chunker(spec, resources))
    result = bench.run(spec.label, chunks, aliases, chunk_tokens=chunk_tokens)
    stats = {"config": spec.label, **chunk_length_stats(chunk_tokens), "чанкинг, с": seconds,
             "BM25, с": result.seconds["bm25"], "эмбеддинг, с": result.seconds["dense"]}
    return chunks, result, stats


chunker_runs = {spec.label: run_chunker(spec) for spec in chunker_specs}
chunker_stats = pd.DataFrame([stats for _, _, stats in chunker_runs.values()]).set_index("config").round(1)
chunker_per_q = pd.concat([result.per_question for _, result, _ in chunker_runs.values()])
chunker_stats

# %%
metrics.comparison_table(chunker_per_q)

# %% [markdown]
# По типам вопросов (validation, R@5):

# %%
by_type = metrics.comparison_table(chunker_per_q, columns=["R@5"], by=["config", "qtype"])
by_type.droplevel(1, axis=1).unstack("qtype")

# %% [markdown]
# **SentenceChunker на постах без знаков препинания.** Пост короче 512 токенов любой чанкер оставляет
# целым, поэтому разница видна только на длинных. Длинный пост без точек режется только по переводам строк.
# Если и их нет, предложение одно и упаковывать нечего: чанк обрезается по пределу токенов, как у Token.

# %%
sentence_chunks = chunker_runs[ChunkerSpec("sentence").label][0]
by_doc = pd.Series([c.doc_id for c in sentence_chunks]).value_counts()
unpunctuated = [d for d in clean.docs if d.source.startswith("tg_") and by_doc.get(d.doc_id, 0) > 1
                and gigarag.patterns.sentences(d.text.replace("\n", " ")).__len__() <= 2]
print("длинных постов Telegram с ≤ 2 предложениями:", len(unpunctuated))
pd.DataFrame([{"chunk_id": c.chunk_id, "начало": c.text[:90], "конец": c.text[-60:]}
              for c in sentence_chunks if unpunctuated and c.doc_id == unpunctuated[0].doc_id])

# %% [markdown]
# ## 4. Ablation семантического чанкинга
#
# **Сетка порогов** — 0,1 / 0,3 / 0,5 / 0,7 / 0,9. В Chonkie 1.7 порог — квантиль локальных минимумов
# близости, а не косинус. Поэтому сетка равномерно покрывает весь диапазон: от «режем только по самым
# глубоким провалам» до «режем по любому». Пороги по косинусу пришлось бы подбирать под каждую модель:
# у e5 близости соседних предложений сидят в 0,8–0,95, у BGE-M3 — в 0,4–0,8.
#
# **Два эмбеддера границ** — `multilingual-e5-small` (118M, быстрый) и `BGE-M3` (568M). Один фактор
# за раз: поиск всегда делает BGE-M3, меняется только модель, по которой режется текст.
# Эмбеддинги предложений кэшируются в памяти, поэтому время эмбеддинга считается один раз на модель
# (у первого порога), остальные пороги — только разрез.

# %%
THRESHOLDS = (0.1, 0.3, 0.5, 0.7, 0.9)
ablation_specs = [ChunkerSpec("semantic", threshold=t, embedder=m) for m in (E5_SMALL, BGE_M3) for t in THRESHOLDS]
ablation_runs = {spec.label: (spec, *run_chunker(spec)) for spec in ablation_specs}
ablation = pd.DataFrame([{**stats, "embedder": spec.embedder.split("/")[-1], "threshold": spec.threshold}
                         for spec, _, _, stats in ablation_runs.values()])
ablation_per_q = pd.concat([result.per_question for _, _, result, _ in ablation_runs.values()])
ablation_metrics = metrics.comparison_table(ablation_per_q, columns=["R@5", "R@10", "MRR"])
ablation_metrics.columns = [" ".join(c) for c in ablation_metrics.columns]
ablation_table = ablation.set_index("config")[["embedder", "threshold", "chunks", "mean", "median", "<50_%",
                                               ">500_%", "чанкинг, с"]].join(ablation_metrics).round(1)
ablation_table

# %%
fig, axes = plt.subplots(1, 3, figsize=(14, 3.6))
for embedder, group in ablation_table.groupby("embedder", sort=False):
    axes[0].plot(group["threshold"], group["mean"], marker="o", label=embedder)
    axes[1].plot(group["threshold"], group["bm25 R@5"], marker="o", label=embedder)
    axes[2].plot(group["threshold"], group["dense R@5"], marker="o", label=embedder)
for ax, title in zip(axes, ["средний размер чанка, токенов", "BM25 R@5, %", "dense R@5, %"], strict=True):
    ax.set_title(title)
    ax.set_xlabel("threshold (квантиль минимумов)")
    ax.legend()
plt.tight_layout()

# %% [markdown]
# **Как выбор модели меняет разбиение.** Для каждого документа, который хоть одна модель разрезала,
# сравниваем множества границ при одном пороге: доля общих границ (Jaccard) и число чанков.

# %%
from gigarag.chunking import boundary_agreement

agreement_rows = []
for threshold in THRESHOLDS:
    e5_chunks = ablation_runs[ChunkerSpec("semantic", threshold=threshold, embedder=E5_SMALL).label][1]
    bge_chunks = ablation_runs[ChunkerSpec("semantic", threshold=threshold, embedder=BGE_M3).label][1]
    agreement_rows.append({"threshold": threshold, **boundary_agreement(e5_chunks, bge_chunks)})
pd.DataFrame(agreement_rows).set_index("threshold").round(3)

# %% [markdown]
# **Визуализация.** Один и тот же текст — раздел статьи dota2.ru о патче — при трёх порогах и двух моделях.
# Каждая строка — предложение, цвет — номер чанка: смена цвета — граница.

# %%
from gigarag.chunking import sentence_chunk_labels

example_doc = min((d for d in clean.docs if d.source == "dota2ru_article" and len(d.text) > 3000),
                  key=lambda d: len(d.text))
example_text = example_doc.text[:3000]
fig, axes = plt.subplots(1, 6, figsize=(15, 6), sharey=True)
for col, (model, threshold) in enumerate([(m, t) for m in (E5_SMALL, BGE_M3) for t in (0.1, 0.5, 0.9)]):
    spec = ChunkerSpec("semantic", threshold=threshold, embedder=model)
    pieces = build_chunker(spec, resources)([example_text])[0]
    labels, sentences = sentence_chunk_labels(example_text, pieces)
    axes[col].imshow(np.array(labels)[:, None] % 10, cmap="tab10", aspect="auto", vmin=0, vmax=9)
    axes[col].set_title(f"{model.split('/')[-1]}\nt={threshold}: {len(pieces)} чанков", fontsize=9)
    axes[col].set_xticks([])
axes[0].set_yticks(range(len(sentences)))
axes[0].set_yticklabels([s[:45] for s in sentences], fontsize=6)
plt.suptitle(f"{example_doc.title[:80]} — первые 3000 символов", fontsize=10)
plt.tight_layout()

# %%
pieces_low = build_chunker(ChunkerSpec("semantic", threshold=0.1, embedder=BGE_M3), resources)([example_text])[0]
pieces_high = build_chunker(ChunkerSpec("semantic", threshold=0.9, embedder=BGE_M3), resources)([example_text])[0]
print("=== BGE-M3, t=0.1 ===")
print("\n--- ✄ ---\n".join(p.strip()[:300] for p in pieces_low[:4]))
print("\n=== BGE-M3, t=0.9 ===")
print("\n--- ✄ ---\n".join(p.strip()[:300] for p in pieces_high[:8]))

# %% [markdown]
# ### Размер чанка для отобранного кандидата
#
# Recursive и Sentence лучше остальных и почти равны; дальше — Recursive: он режет по абзацам, а в корпусе
# абзац — естественная единица (блок патчноута, раздел статьи). Единственный гиперпараметр — предел размера.
# 1024 не пробуем: BGE-M3 эмбеддит не больше 512 токенов, и хвост большого чанка dense-поиск не увидел бы.

# %%
size_specs = [ChunkerSpec("recursive", chunk_size=size) for size in (128, 256, 512)]
size_runs = {spec.label: run_chunker(spec) for spec in size_specs}
size_per_q = pd.concat([result.per_question for _, result, _ in size_runs.values()])
size_metrics = metrics.comparison_table(size_per_q)
size_metrics.columns = [" ".join(c) for c in size_metrics.columns]
pd.DataFrame([stats for _, _, stats in size_runs.values()]).set_index("config")[
    ["chunks", "mean", "median", "<50_%", ">500_%"]].join(size_metrics).round(1)

# %%
BEST = ChunkerSpec("recursive", chunk_size=512)
best_chunks, best_result, best_stats = run_chunker(BEST)
best_tokens = bench.chunks(clean.docs, BEST.params(), lambda: build_chunker(BEST, resources))[1]
print("лучший простой чанкер:", BEST.label, "| чанков:", len(best_chunks))

# %% [markdown]
# ## 5. ParentDocumentRetriever
#
# Родитель — Recursive-чанк размера P (абзацы целиком), ребёнок — Token-чанк размера C внутри родителя.
# Индексируются дети (BGE-M3 и BM25), выдаются родители: score родителя — лучший score его детей,
# повторы схлопываются. Мелкий ребёнок точнее эмбеддится, крупный родитель отдаёт генератору контекст.
# Родитель длиннее 512 токенов не эмбеддится сам, поэтому P может быть больше контекста BGE-M3.

# %%
from gigarag.chunking import child_chunks
from gigarag.experiment import Entries

PARENT_CHILD = [(512, 128), (1024, 128), (1024, 256), (2048, 256)]
parent_runs, parent_stats = {}, []
for parent_size, child_size in PARENT_CHILD:
    parent_spec = ChunkerSpec("recursive", chunk_size=parent_size)
    child_spec = ChunkerSpec("token", chunk_size=child_size)
    parents, parent_tokens, parent_seconds = bench.chunks(clean.docs, parent_spec.params(),
                                                          lambda s=parent_spec: build_chunker(s, resources))
    children, owners = child_chunks(parents, build_chunker(child_spec, resources))
    config = f"parent {parent_size} / child {child_size}"
    result = bench.run(config, parents, clean.aliases, entries=Entries([c.text for c in children], owners),
                       chunk_tokens=parent_tokens)
    parent_runs[config] = (parents, parent_tokens, result)
    parent_stats.append({"config": config, "родителей": len(parents), "детей": len(children),
                         "токенов в родителе": parent_tokens.mean(), "чанкинг, с": parent_seconds,
                         "эмбеддинг детей, с": result.seconds["dense"]})
parent_per_q = pd.concat([best_result.per_question, *(r.per_question for _, _, r in parent_runs.values())])
parent_metrics = metrics.comparison_table(parent_per_q)
parent_metrics.columns = [" ".join(c) for c in parent_metrics.columns]
pd.DataFrame(parent_stats).set_index("config").round(1).join(parent_metrics, how="right")

# %% [markdown]
# По типам вопросов: R@5 на validation относительно лучшего простого чанкера и сколько токенов контекста
# в среднем занимают топ-5 (`tokens@5`): выигрыш родителей отчасти покупается объёмом.

# %%
parent_by_type = metrics.comparison_table(parent_per_q, columns=["R@5"], by=["config", "qtype"])
display(parent_by_type.droplevel(1, axis=1).unstack("qtype"))
parent_per_q[parent_per_q["split"] == "validation"].groupby(["config", "retriever"], sort=False)[
    "tokens@5"].mean().unstack("retriever").round(0)

# %% [markdown]
# ## 6. Метаданные
#
# Извлекаются для чанков лучшего простого чанкера (`gigarag.metadata.chunk_metadata`):
#
# - **структурные** — дата, источник, патч (у постов Telegram — патч, действовавший на дату поста),
#   длина документа в токенах, позиция чанка (номер и относительная позиция);
# - **контентные** — тип (патчноут / статья / комментарий / пост / офтоп), язык по доле латиницы
#   (ru ≤ 20% < mixed < 60% ≤ en), топ-5 ключевых слов по TF-IDF;
# - **качественный** — шумовой score: доля неалфавитных токенов BGE-M3 (числа, знаки, эмодзи) среди непробельных.
#
# Ключевые слова — TF-IDF, а не KeyBERT: на 20 тыс. чанков он считается за секунды и детерминирован.
# Для коротких постов KeyBERT выбирал бы фразы из тех же 20–50 слов.

# %%
import gigarag.metadata
from gigarag import metadata

doc_token_counts = dict(zip([d.doc_id for d in clean.docs],
                            tokens.token_counts(tokenizer, [d.text for d in clean.docs]), strict=True))
meta = load_or_compute(
    cache.path("chunk_meta", "parquet", docs=docs_fingerprint(clean.docs), chunker=BEST.label,
               code=code_fingerprint(gigarag.metadata, gigarag.noise, gigarag.patterns), tokenizer=BGE_M3_REV),
    lambda: metadata.chunk_metadata(best_chunks, clean.docs, doc_token_counts, offsets),
    load=read_parquet, save=write_parquet,
)
meta.sample(6, random_state=42)[["chunk_id", "source", "date", "patch", "doc_tokens", "position", "content_type",
                                 "language", "noise_score", "keywords"]]

# %%
meta.groupby("content_type").agg(чанков=("chunk_id", "size"), шум_медиана=("noise_score", "median"),
                                 ru=("language", lambda s: (s == "ru").mean() * 100),
                                 mixed=("language", lambda s: (s == "mixed").mean() * 100),
                                 en=("language", lambda s: (s == "en").mean() * 100)).round(2)

# %% [markdown]
# Шумовой score у патчноутов самый высокий: в правке «с 30% до 35%» числа и знаки — суть, а не мусор.
# Поэтому фильтр «выбросить шумные чанки» опасен — проверяем его ниже вместе с полезными фильтрами.
#
# ### Применение 1: фильтрация при ретриве
#
# Фильтр — маска допустимых чанков до выбора топ-k. Три сценария:
#
# - **по патчу** — если вопрос называет патч («7.36c»), остаются чанки этого патча или упоминающие его
#   в тексте. Номер патча берётся из самого запроса — никакой подсказки из gold;
# - **без офтопа** — нетематические посты Telegram (20% от тематических, оставлены в КТ 1 как шум) исключены;
# - **без шумных** — чанки с шумовым score > 0,5 исключены.
#
# Precision считается только на вопросах, к которым фильтр применим.

# %%
best_texts = [c.text for c in best_chunks]
filters = {
    "фильтр по патчу": metadata.patch_filter(meta, best_texts),
    "без офтопа": metadata.drop_types_filter(frozenset({metadata.OFFTOPIC})),
    "без шумных (score > 0,5)": metadata.max_noise_filter(0.5),
}
filter_rows = []
filter_runs = {}
for name, chunk_filter in filters.items():
    masks = metadata.filter_masks(chunk_filter, meta, bench.queries)
    filtered = bench.run(name, best_chunks, clean.aliases, masks=masks, chunk_tokens=best_tokens)
    filter_runs[name] = filtered
    applies = best_result.per_question["qid"].isin(masks)
    before = best_result.per_question[applies].assign(config="без фильтра")
    after = filtered.per_question[filtered.per_question["qid"].isin(masks)]
    table = metrics.comparison_table(pd.concat([before, after]), columns=["P@5", "R@5", "Hit@5", "MRR"])
    table.index = pd.MultiIndex.from_tuples([(name, cfg) for cfg in table.index], names=["фильтр", "выдача"])
    filter_rows.append(table.assign(вопросов=before.query("split == 'validation'")["qid"].nunique()))
pd.concat(filter_rows)

# %% [markdown]
# ### Применение 2: метаданные в тексте чанка (prepend)
#
# Перед индексированием к чанку добавляется строка «заголовок документа | тип | патч | дата». Комментарий
# или кусок статьи сам по себе не называет патч и тему — префикс их добавляет. Выдаётся и оценивается
# тот же чанк, меняется только индексируемый текст.

# %%
prefixed = [metadata.with_metadata_prefix(text, row) for text, row in zip(best_texts, meta.to_dict("records"),
                                                                           strict=True)]
print(prefixed[next(i for i, c in enumerate(best_chunks) if c.doc_id.startswith("dota2ru:comment"))][:300])
prepend = bench.run("prepend метаданных", best_chunks, clean.aliases,
                    entries=Entries(prefixed, list(range(len(best_chunks)))), chunk_tokens=best_tokens)
prepend_per_q = pd.concat([best_result.per_question, prepend.per_question])
display(metrics.comparison_table(prepend_per_q, columns=metrics.METRIC_COLUMNS))
display(metrics.paired_changes(prepend_per_q, BEST.label, "prepend метаданных"))
metrics.comparison_table(prepend_per_q, columns=["R@5"], by=["config", "qtype"]).droplevel(1, axis=1).unstack("qtype")

# %% [markdown]
# ## 7. LLM-обработка чанков
#
# Модель `deepseek-flash` (DeepSeek-V4.1-Flash, дешёвая массовая обработка), t = 0, по одному чанку на запрос,
# промпты — `gigarag.enrich` (версия 1). Обрабатываются чанки от 30 токенов: резюме реплики «+1» или вопросы
# к подписи из трёх слов ничего не дают. Короткие чанки в обоих вариантах индексируются своим текстом.
#
# - **Суммаризация** — индексируется резюме вместо текста чанка.
# - **Гипотетические вопросы** — 3 вопроса на чанк, каждый — отдельная запись индекса; score чанка — лучший
#   из его вопросов.
#
# Выдаётся и оценивается исходный чанк: релевантность — по цитате в его тексте.

# %%
import gigarag.enrich
from gigarag import enrich
from gigarag.cache import load_or_compute_timed, read_json, write_json
from gigarag.experiment import texts_fingerprint
from gigarag.llm import CachedChat, OpenAIChat

chat = CachedChat(OpenAIChat.from_env(ROOT / ".env", enrich.PROVIDER, enrich.PROVIDER_BASE_URL), DATA / "llm_cache")
llm_targets = enrich.eligible(best_tokens)
print(f"в LLM идут {len(llm_targets)} чанков из {len(best_chunks)} ({len(llm_targets) / len(best_chunks):.0%})")
ENRICH_CODE = code_fingerprint(gigarag.enrich)


def enrich_all(name: str, make_request):
    """Ответы LLM на все отобранные чанки и время первого прогона (повторный — из кэша запросов)."""
    def compute():
        responses = enrich.complete_all(chat, [make_request(best_texts[i]) for i in llm_targets])
        return [{"content": r.content, "prompt_tokens": r.prompt_tokens, "completion_tokens": r.completion_tokens}
                for r in responses]
    path = cache.path(f"llm_{name}", "json", texts=texts_fingerprint([best_texts[i] for i in llm_targets]),
                      code=ENRICH_CODE)
    return load_or_compute_timed(path, compute, load=read_json, save=write_json)


summaries, summary_seconds = enrich_all("summary", enrich.summary_request)
questions, questions_seconds = enrich_all("questions", enrich.questions_request)
llm_usage = pd.DataFrame([
    {"обработка": name, "запросов": len(records), "входных токенов": sum(r["prompt_tokens"] for r in records),
     "выходных токенов": sum(r["completion_tokens"] for r in records), "время, мин": seconds / 60,
     "стоимость, $": enrich.Usage(len(records), sum(r["prompt_tokens"] for r in records),
                                  sum(r["completion_tokens"] for r in records), 0).cost_usd}
    for name, records, seconds in [("суммаризация", summaries, summary_seconds),
                                   ("вопросы", questions, questions_seconds)]
]).set_index("обработка").round(2)
llm_usage

# %%
summary_text = dict(zip(llm_targets, (r["content"].strip() for r in summaries), strict=True))
summary_entries = Entries([summary_text.get(i, text) for i, text in enumerate(best_texts)],
                          list(range(len(best_chunks))))
chunk_questions = dict(zip(llm_targets, (enrich.parse_questions(r["content"]) for r in questions), strict=True))
question_pairs = [(q, i) for i, text in enumerate(best_texts) for q in chunk_questions.get(i) or [text]]
question_entries = Entries([q for q, _ in question_pairs], [i for _, i in question_pairs])
broken_json = sum(not llm.is_json_object(r["content"]) for r in questions)
no_questions = sum(not chunk_questions[i] for i in llm_targets)
print(f"Ответов без вопросов — {no_questions} из {len(llm_targets)}: битый JSON — {broken_json}, "
      f"остальные — отказ модели (пустой список, «фрагмент не про Dota»). Такие чанки индексируются своим текстом")
example = llm_targets[len(llm_targets) // 2]
print("ЧАНК:", best_texts[example][:400], "\n\nРЕЗЮМЕ:", summary_text[example],
      "\n\nВОПРОСЫ:", chunk_questions[example])

# %%
summary_run = bench.run("резюме вместо текста", best_chunks, clean.aliases, entries=summary_entries,
                        chunk_tokens=best_tokens)
questions_run = bench.run("вопросы вместо текста", best_chunks, clean.aliases, entries=question_entries,
                          chunk_tokens=best_tokens)
llm_per_q = pd.concat([best_result.per_question, summary_run.per_question, questions_run.per_question])
display(metrics.comparison_table(llm_per_q, columns=metrics.METRIC_COLUMNS))
display(pd.concat({name: metrics.paired_changes(llm_per_q, BEST.label, name)
                   for name in ("резюме вместо текста", "вопросы вместо текста")}))
metrics.comparison_table(llm_per_q, columns=["R@5"], by=["config", "qtype"]).droplevel(1, axis=1).unstack("qtype")

# %% [markdown]
# ## 8. Итоговая таблица
#
# Строка — конфигурация «предобработка + чанкер + метаданные + LLM». Validation и test — отдельно;
# test не участвовал ни в одном выборе. Время индексирования — чанкинг + эмбеддинг записей BGE-M3
# (+ LLM-обработка, где она есть), с. BM25 строится за доли секунды и в сумму не входит.

# %%
best_semantic = ChunkerSpec("semantic", threshold=0.1, embedder=E5_SMALL)
best_parent = "parent 1024 / child 128"
final_configs = [
    ("сырой корпус", "token(512)", "—", "—", baseline, raw_chunk_seconds, raw_tokens),
    ("вся чистка без signatures", "token(512)", "—", "—", control, None, None),
    ("вся чистка без signatures", BEST.label, "—", "—", best_result, best_stats["чанкинг, с"], best_tokens),
    ("вся чистка без signatures", best_semantic.label, "—", "—", ablation_runs[best_semantic.label][2],
     ablation_runs[best_semantic.label][3]["чанкинг, с"], None),
    ("вся чистка без signatures", chunker_specs[4].label, "—", "—", chunker_runs[chunker_specs[4].label][1],
     chunker_runs[chunker_specs[4].label][2]["чанкинг, с"], None),
    ("вся чистка без signatures", best_parent, "—", "—", parent_runs[best_parent][2], None,
     parent_runs[best_parent][1]),
    ("вся чистка без signatures", BEST.label, "prepend", "—", prepend, best_stats["чанкинг, с"], best_tokens),
    ("вся чистка без signatures", BEST.label, "фильтр по патчу", "—", filter_runs["фильтр по патчу"],
     best_stats["чанкинг, с"], best_tokens),
    ("вся чистка без signatures", BEST.label, "—", "резюме", summary_run, best_stats["чанкинг, с"] + summary_seconds,
     best_tokens),
    ("вся чистка без signatures", BEST.label, "—", "вопросы", questions_run,
     best_stats["чанкинг, с"] + questions_seconds, best_tokens),
]
final_rows = []
for preprocessing, chunker, meta_use, llm_use, run, chunk_seconds, chunk_tokens_ in final_configs:
    per_q = run.per_question
    row = {"предобработка": preprocessing, "чанкер": chunker, "метаданные": meta_use, "LLM": llm_use}
    for split in ("validation", "test"):
        means = per_q[per_q["split"] == split].groupby("retriever")[["Hit@5", "R@5", "MRR"]].mean() * 100
        row |= {f"{split[:4]} {r} {m}": means.loc[r, m] for r in ("bm25", "dense") for m in ("Hit@5", "R@5")}
    row["токенов в чанке"] = float(np.mean(chunk_tokens_)) if chunk_tokens_ is not None else np.nan
    row["индексирование, с"] = (chunk_seconds or 0) + run.seconds["dense"] if chunk_seconds is not None else np.nan
    final_rows.append(row)
final_table = pd.DataFrame(final_rows).round(1)
final_table
