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
# # КТ 5 — генерация ответов и оценка качества
#
# Решения и выводы — в `reports/kt5.md`, здесь — эксперименты и цифры.
#
# **Две конфигурации контекста** — одинаковый генератор, разный поиск, чтобы изолировать вклад ретрива:
#
# - **baseline retrieval** — топ-5 baseline КТ 3: BGE-M3 + BM25, alpha = 0,6, фильтр по патчу;
# - **best retrieval** — топ-5 лучшей конфигурации КТ 4: тот же поиск + LLM-реранкер listwise
#   (промпт v2 «оценки 0–2 всей пачке»), пул 20.
#
# **Генератор и судья** — `deepseek-flash` без рассуждения (`thinking: disabled`), t = 0, seed 42. Промпт
# генерации — `gigarag.generation` (версия 1), не меняется в течение КТ 5. Судья — `gigarag.judge`: три критерия
# отдельными запросами, шкала 1–5 с описанием каждого балла.
#
# **Выборки** — как в КТ 3–4: решения по validation (36 вопросов), test (14) — для контроля. Ответы
# генерируются на все 50 вопросов gold.

# %%
import random
from pathlib import Path

import numpy as np
import pandas as pd

from gigarag import annotation, baseline, generation, judge, metadata, metrics, tokens
from gigarag.cache import CacheStore, load_or_compute, read_npy, write_npy
from gigarag.corpus import Document, read_jsonl
from gigarag.embeddings import encode, load_sentence_model
from gigarag.enrich import PROVIDER, PROVIDER_BASE_URL, Usage
from gigarag.experiment import texts_fingerprint
from gigarag.gold import load_gold
from gigarag.llm import CachedChat, OpenAIChat
from gigarag.llm_rerank import GRADES_PROMPT, ListwiseLlmReranker
from gigarag.parallel import parallel_map
from gigarag.patterns import is_abstention
from gigarag.pipeline import HybridRetriever, Pipeline, run_all
from gigarag.relevance import build_qrels
from gigarag.retrieval import Bm25Index
from gigarag.search import ranked_ids
from gigarag.stats import agreement_rate, weighted_kappa

pd.set_option("display.max_colwidth", 160)
pd.set_option("display.width", 220)

ROOT = Path.cwd().parent  # ноутбук исполняется из notebooks/
DATA = ROOT / "data"
rows, CORPUS_SHA = read_jsonl(DATA / "corpus.jsonl")
items = load_gold(ROOT / "gold" / "gold.jsonl")
cache = CacheStore(DATA / "cache" / "kt4", CORPUS_SHA)  # чанки и векторы — те же, что в КТ 4
chat = CachedChat(OpenAIChat.from_env(ROOT / ".env", PROVIDER, PROVIDER_BASE_URL), DATA / "llm_cache")
repeat_chat = CachedChat(chat.client, DATA / "llm_cache" / "kt5_repeat")  # второй прогон судьи — заново

tokenizer = tokens.load_tokenizer(baseline.EMBEDDER.name, baseline.EMBEDDER.revision)
corpus = baseline.build_corpus(tuple(Document(**r) for r in rows), tokenizer, cache)
passages = corpus.dense_texts
embedder = load_sentence_model(baseline.EMBEDDER.name, baseline.EMBEDDER.revision,
                               max_seq_length=baseline.EMBEDDER.max_seq_length, half=True)
vectors = load_or_compute(
    cache.path("vectors", "npy", texts=texts_fingerprint(passages), half=True, **baseline.EMBEDDER.cache_params()),
    lambda: encode(embedder, passages), load=read_npy, save=write_npy)
retriever = HybridRetriever(vectors, Bm25Index.build(corpus.texts, baseline.BM25_K1, baseline.BM25_B),
                            baseline.ALPHA, baseline.POOL)


def encode_queries(texts) -> np.ndarray:
    return encode(embedder, list(texts), batch_size=len(texts))


gold_queries = {i.qid: i.question for i in items}
by_qid = {i.qid: i for i in items}
patch_masks = metadata.filter_masks(metadata.patch_filter(corpus.meta, corpus.texts), corpus.meta, gold_queries)
print(f"{len(corpus.chunks)} чанков, {len(items)} вопросов gold")

# %% [markdown]
# ## 1. Контекст: топ-5 двух конфигураций поиска

# %%
BASELINE, BEST = "baseline retrieval", "best retrieval"
LLM_POOL = 20
pipelines = {BASELINE: Pipeline(retriever, encode_queries),
             BEST: Pipeline(retriever, encode_queries,
                            rerank=ListwiseLlmReranker(chat, passages, LLM_POOL, GRADES_PROMPT))}
traces = {name: run_all(p, gold_queries, patch_masks) for name, p in pipelines.items()}
contexts = {name: {qid: [doc for doc, _ in t.ranking[:generation.CONTEXT_K]] for qid, t in runs.items()}
            for name, runs in traces.items()}
same = sum(set(contexts[BASELINE][q]) == set(contexts[BEST][q]) for q in gold_queries)
print(f"вопросов с тем же набором топ-5 в обеих конфигурациях: {same} из {len(gold_queries)}")

# %% [markdown]
# ## 2. Генерация
#
# Один запрос на вопрос и конфигурацию: системный промпт `generation.SYSTEM` и пронумерованные топ-5 чанков.
# Отказ распознаётся по началу ответа «Не знаю» (`patterns.is_abstention`).

# %%
print(generation.SYSTEM)
answer_keys = [(name, qid) for name in (BASELINE, BEST) for qid in gold_queries]


def context_of(key: tuple[str, str]) -> list[str]:
    name, qid = key
    return [passages[doc] for doc in contexts[name][qid]]


responses = dict(zip(answer_keys, parallel_map(
    lambda key: chat.complete(generation.answer_request(gold_queries[key[1]], context_of(key))), answer_keys),
    strict=True))
answers = {key: r.content for key, r in responses.items()}
generation_usage = Usage(len(responses), sum(r.prompt_tokens for r in responses.values()),
                         sum(r.completion_tokens for r in responses.values()), 0)
print(f"токенов на ответ: {generation_usage.prompt_tokens / len(responses):.0f} вход, "
      f"{generation_usage.completion_tokens / len(responses):.0f} выход; "
      f"${generation_usage.cost_usd / len(responses) * 1000:.2f} на 1000 ответов; "
      f"медиана ответа API {np.median([r.seconds for r in responses.values()]):.2f} с")

# %%
answer_table = pd.DataFrame([{"config": name, "qid": qid, "qtype": by_qid[qid].qtype, "split": by_qid[qid].split,
                              "отказ": is_abstention(answers[(name, qid)])} for name, qid in answer_keys])
(answer_table.pivot_table(index="config", columns="qtype", values="отказ", aggfunc="mean", sort=False)
 .mul(100).round(1).rename_axis(columns="доля отказов, %"))

# %% [markdown]
# ## 3. Выборка для ручной разметки
#
# 30 ответов из 100 — случайная выборка (seed 42) из пар «конфигурация, вопрос», упорядоченных по ключу.
# Разметчик видит то же, что судья: вопрос, пять фрагментов и ответ; конфигурация, эталон gold и оценки
# судьи скрыты. Разметка — `scripts/kt5_annotate.py` (локальный веб-сервер), оценки — `gold/kt5_human_labels.json`.

# %%
ANNOTATION_SIZE = 30
annotation_keys = random.Random(42).sample(sorted(answer_keys), ANNOTATION_SIZE)
annotation_records = [{"id": f"{name}|{qid}", "question": gold_queries[qid],
                       "passages": context_of((name, qid)), "answer": answers[(name, qid)]}
                      for name, qid in annotation_keys]
annotation.write_sample(ROOT / "gold" / "kt5_annotation_sample.json", annotation_records)
pd.Series([key[0] for key in annotation_keys]).value_counts().rename("ответов в выборке")

# %% [markdown]
# ## 4. LLM-судья: итерации промпта и детерминизм
#
# Каждая версия промпта (`judge.PROMPTS`) оценивает 30 ответов выборки разметки дважды при t = 0: второй прогон —
# с отдельным кэшем (`llm_cache/kt5_repeat`), модель отвечает заново. Agreement — доля ответов с тем же баллом;
# порог задания — 85%. Шкалы `judge.SCALES` во всех версиях одинаковые — те же, что видит человек.
#
# - **v1** — шкала и короткое обоснование перед баллом.
# - **v2** — та же шкала, но сначала свободный разбор (части вопроса, утверждения, нужные факты) и правила подсчёта
#   для пограничных случаев v1; при колебании — меньший балл.
# - **v3** — вместо свободного разбора короткие структурированные поля (неподтверждённые утверждения, части вопроса
#   с отметкой «отвечена», нужные и пропущенные факты, есть ли ответ в контексте) и балл по правилу из них.

# %%
annotation_cases = {key: (gold_queries[key[1]], context_of(key), answers[key]) for key in annotation_keys}
runs = {version: (judge.judge_answers(chat, annotation_cases, version),
                  judge.judge_answers(repeat_chat, annotation_cases, version)) for version in judge.PROMPTS}


def scores(verdicts, criterion: str, keys) -> list:
    return [verdicts[(criterion, key)].score for key in keys]


determinism = pd.DataFrame({
    f"v{version}": {**{criterion: 100 * agreement_rate(scores(first, criterion, annotation_keys),
                                                       scores(second, criterion, annotation_keys))
                       for criterion in judge.CRITERIA},
                    "ответов не по формату": sum(v.score is None for run in (first, second) for v in run.values())}
    for version, (first, second) in runs.items()}).T
determinism.round(1)

# %% [markdown]
# ### Расхождения между прогонами
#
# Все пары, где два прогона одной версии поставили разный балл, — с обоснованиями обоих прогонов.

# %%
disagreements = pd.DataFrame([
    {"версия": f"v{version}", "критерий": criterion, "ответ": f"{key[1]} ({key[0]})",
     "баллы": f"{first[(criterion, key)].score} / {second[(criterion, key)].score}",
     "прогон 1": first[(criterion, key)].rationale[:300], "прогон 2": second[(criterion, key)].rationale[:300]}
    for version, (first, second) in runs.items() for criterion in judge.CRITERIA for key in annotation_keys
    if first[(criterion, key)].score != second[(criterion, key)].score])
print(disagreements.groupby(["версия", "критерий"]).size().unstack(fill_value=0))
disagreements

# %% [markdown]
# ## 5. Валидация человеком: weighted Cohen's kappa
#
# Человек разметил те же 30 ответов по тем же шкалам (`gold/kt5_human_labels.json`), не видя оценок судьи,
# конфигурации и эталона. Первый проход (`gold/kt5_human_labels_pass1.json`) сделан до того, как правила для
# отказа и «достоверность — только относительно фрагментов» вынесли на страницу разметки отдельным блоком: в нём
# отказы получали низкую достоверность вопреки шкале. Решения — по второму проходу, первый — для сравнения.
#
# Kappa — с квадратичными весами (`stats.weighted_kappa`), ответы судьи не по формату исключаются из пары.
# Версия судьи для КТ 5 — среди прошедших порог детерминизма (≥ 85% по всем критериям, без ответов не по формату)
# та, у которой выше средняя kappa по второму проходу. Kappa посчитана на тех же 30 ответах, по которым выбрана
# версия, — это оптимистичная оценка.

# %%
human = annotation.read_labels(ROOT / "gold" / "kt5_human_labels.json")
human_pass1 = annotation.read_labels(ROOT / "gold" / "kt5_human_labels_pass1.json")
assert all(annotation.is_complete(human, r["id"]) for r in annotation_records), "разметка второго прохода неполная"


def kappa(verdicts, labels, criterion: str) -> float:
    pairs = [(verdicts[(criterion, key)].score, labels[f"{key[0]}|{key[1]}"][criterion]) for key in annotation_keys]
    pairs = [(j, h) for j, h in pairs if j is not None]
    return weighted_kappa([j for j, _ in pairs], [h for _, h in pairs], judge.SCORES)


kappas = pd.DataFrame({(f"v{version}", source): {c: kappa(first, labels, c) for c in judge.CRITERIA}
                       for version, (first, _) in runs.items()
                       for source, labels in (("проход 2", human), ("проход 1", human_pass1))}).T


def closeness(verdicts, labels, tolerance: int) -> dict[str, float]:
    """Доля ответов, где судья и человек разошлись не больше чем на `tolerance` баллов, %. Kappa при почти
    постоянной оценке одного из оценщиков ≈ 0 даже при почти полном совпадении — доли показывают это рядом."""
    return {c: 100 * np.mean([abs(verdicts[(c, key)].score - labels[f"{key[0]}|{key[1]}"][c]) <= tolerance
                              for key in annotation_keys if verdicts[(c, key)].score is not None])
            for c in judge.CRITERIA}


closeness_table = pd.DataFrame({(f"v{version}", what): closeness(first, human, tolerance)
                                for version, (first, _) in runs.items()
                                for what, tolerance in (("точно, %", 0), ("±1, %", 1))}).T
stable = [v for v in judge.PROMPTS if determinism.loc[f"v{v}", list(judge.CRITERIA)].min() >= 85
          and determinism.loc[f"v{v}", "ответов не по формату"] == 0]
JUDGE_VERSION = max(stable, key=lambda v: kappas.loc[(f"v{v}", "проход 2")].mean())
print("версия судьи для КТ 5:", f"v{JUDGE_VERSION}")
display(kappas.round(3))
closeness_table.round(1)

# %%
human_table = pd.DataFrame([human[f"{key[0]}|{key[1]}"] for key in annotation_keys])
judge_table = pd.DataFrame([{c: runs[JUDGE_VERSION][0][(c, key)].score for c in judge.CRITERIA}
                            for key in annotation_keys])
display(pd.DataFrame({"человек": human_table.mean(), "судья": judge_table.mean()}).T.round(2))
for criterion in judge.CRITERIA:  # матрица «человек × судья»: где именно расходятся
    display(pd.crosstab(human_table[criterion], judge_table[criterion], rownames=[f"{criterion}: человек"],
                        colnames=["судья"]))

# %% [markdown]
# ## 6. Оценка всех ответов финальным судьёй
#
# 100 ответов (50 вопросов × 2 конфигурации) × 3 критерия. Отдельно — поведение на отказах: доля «не знаю»
# на негативных (должна быть 100%) и ложные отказы на вопросах, у которых ответ есть.

# %%
all_cases = {key: (gold_queries[key[1]], context_of(key), answers[key]) for key in answer_keys}
final = judge.judge_answers(chat, all_cases, JUDGE_VERSION)
judged = answer_table.assign(**{c: [final[(c, key)].score for key in answer_keys] for c in judge.CRITERIA})
print("ответов судьи не по формату:", int(judged[list(judge.CRITERIA)].isna().sum().sum()))
display(judged.groupby(["config", "split"], sort=False)[list(judge.CRITERIA)].mean().round(2))
judged[judged["split"] == "validation"].pivot_table(index="config", columns="qtype", values=list(judge.CRITERIA),
                                                    sort=False).round(2)

# %%
abstentions = judged.assign(negative=judged["qtype"] == "negative").groupby(["config", "negative"], sort=False)[
    "отказ"].mean().mul(100).unstack().rename(columns={True: "отказ на негативных, %",
                                                       False: "ложный отказ на остальных, %"})
abstentions.round(1)

# %% [markdown]
# ## 7. Итоговая таблица
#
# Поиск — R@5 и NDCG@5 (как в КТ 3–4), генерация — средние баллы судьи, отказы — по всем 50 вопросам.
# Задержка — медиана поиска (с реранкингом) и медиана ответа генератора по 50 вопросам.

# %%
qrels = build_qrels(items, corpus.chunks, corpus.clean.aliases)
retrieval = pd.concat([metrics.per_question({qid: ranked_ids(t.ranking) for qid, t in traces[name].items()}, qrels,
                                            items).assign(config=name) for name in (BASELINE, BEST)])
summary = pd.DataFrame({
    name: {
        "R@5": 100 * retrieval.query("config == @name and split == 'validation'")["R@5"].mean(),
        "NDCG@5": 100 * retrieval.query("config == @name and split == 'validation'")["NDCG@5"].mean(),
        **{c: judged.query("config == @name and split == 'validation'")[c].mean() for c in judge.CRITERIA},
        "отказ на негативных, %": abstentions.loc[name, "отказ на негативных, %"],
        "ложный отказ, %": abstentions.loc[name, "ложный отказ на остальных, %"],
        "поиск, мс (медиана)": 1000 * np.median([t.total_seconds for t in traces[name].values()]),
        "генерация, мс (медиана)": 1000 * np.median([responses[(name, q)].seconds for q in gold_queries]),
    } for name in (BASELINE, BEST)}).T
summary.round(2)

# %% [markdown]
# ## 8. Выводы
#
# Метрики — validation (29 вопросов с доказательствами и 7 негативных), если не сказано иное.
#
# **Генерация.** Промпт «только по фрагментам, иначе „Не знаю“» отказывается на всех 10 негативных вопросах
# в обеих конфигурациях. Достоверность почти максимальная (4,94 из 5): генератор не выдумывает. Слабое место —
# абстрактные вопросы: ложные отказы — 50% у baseline и 40% у best retrieval, использование контекста — 2,71 и 3,29.
#
# **Судья.** v1 и v3 проходят порог детерминизма (≥ 85% по каждому критерию), v2 — нет: свободный разбор перед
# баллом увеличил разброс (релевантность 80%, два ответа не по формату). v3 заменила разбор короткими полями
# и правилами для пограничных случаев v1. Против человека (второй проход) v3: κ релевантности 0,81,
# использования контекста 0,91 — судья надёжен. Достоверность: человек поставил 5 всем 30 ответам, судья — 26 раз 5
# и 4 раза 4, κ = 0 — парадокс kappa при постоянной оценке одного оценщика; точное совпадение — 86,7%, ±1 — 100%.
# Первый проход разметки — до того, как правила для отказа вынесли на страницу отдельным блоком, — давал κ 0,25–0,35:
# расхождение было в понимании шкалы, а не в судье.
#
# **Вклад поиска.** Best retrieval поднимает R@5 66,0 → 71,8 и NDCG@5 60,2 → 67,4, использование контекста —
# 4,47 → 4,56, но релевантность ответов падает 4,19 → 3,94, и всё падение — на абстрактных (3,57 → 2,14). Причина —
# в судье без эталона и в генераторе: лучший поиск чаще приносит частично полезный контекст, а генератор всё равно
# отказывается или отвечает мимо, и такой отказ получает 1 вместо 5 при пустом контексте. Это узкое место
# генерации, а не поиска: на абстрактных вопросах модель не собирает ответ из разрозненных фрагментов.
