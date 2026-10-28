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

from gigarag import annotation, baseline, generation, judge, metadata, metrics, perturbation, tokens
from gigarag.atomic import atomic_write_jsonl
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
from gigarag.stats import agreement_rate, weighted_kappa, within_tolerance

pd.set_option("display.max_colwidth", 160)
pd.set_option("display.width", 220)

ROOT = Path.cwd().parent  # ноутбук исполняется из notebooks/
DATA = ROOT / "data"
rows, CORPUS_SHA = read_jsonl(DATA / "corpus.jsonl")
items = load_gold(ROOT / "gold" / "gold.jsonl")
cache = CacheStore(DATA / "cache" / "kt4", CORPUS_SHA)  # чанки и векторы — те же, что в КТ 4
client = OpenAIChat.from_env(ROOT / ".env", PROVIDER, PROVIDER_BASE_URL)
rerank_chat = CachedChat(client, DATA / "llm_cache")  # реранкер best retrieval — те же ответы, что в КТ 4
# Генерация и судья КТ 5 — в своём кэше: все ответы получены одним прогоном и хранят идентичность модели
# (model, system_fingerprint, id ответа API); второй прогон судьи для детерминизма — в отдельном кэше.
chat = CachedChat(client, DATA / "llm_cache" / "kt5")
repeat_chat = CachedChat(client, DATA / "llm_cache" / "kt5" / "repeat")

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
                            rerank=ListwiseLlmReranker(rerank_chat, passages, LLM_POOL, GRADES_PROMPT))}
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
# ## 3. Выборки для ручной разметки
#
# Выборки делятся по вопросам, а не по парам «конфигурация, вопрос»: обе конфигурации одного вопроса часто дают
# тот же контекст и тот же ответ, и пара в dev и holdout была бы дубликатом. На каждый вопрос — один ответ,
# конфигурация выбирается случайно (seed 42); вопросы двух выборок не пересекаются.
#
# - **holdout** — 20 вопросов из тех, что не участвовали в подборе промпта судьи в предыдущем прогоне КТ 5
#   (`gold/kt5_human_labels_pass1.json` — та выборка). Размечен после заморозки версии; итоговая kappa — по нему.
# - **dev** — остальные 30 вопросов: по ним итерируется промпт и выбирается версия.
#
# Разметчик видит то же, что судья: вопрос, пять фрагментов и ответ; конфигурация, эталон gold и оценки судьи
# скрыты. Оценка хранится с хэшем этого содержимого (`annotation.content_digest`): оценки прежнего прогона
# перенесены только на ответы, которые совпали дословно, остальные размечены заново
# (`scripts/kt5_annotate.py dev|holdout`).

# %%
HOLDOUT_SIZE = 20
previous_dev = annotation.read_labels(ROOT / "gold" / "kt5_human_labels_pass1.json")  # dev прошлого прогона
tuned_qids = {label_id.split("|")[1] for label_id in previous_dev}
qids = sorted(gold_queries)
holdout_qids = sorted(random.Random(42).sample(sorted(set(qids) - tuned_qids), HOLDOUT_SIZE))
dev_qids = sorted(set(qids) - set(holdout_qids))
config_rng = random.Random(42)
config_of = {qid: config_rng.choice((BASELINE, BEST)) for qid in qids}
annotation_keys = [(config_of[qid], qid) for qid in dev_qids]
holdout_keys = [(config_of[qid], qid) for qid in holdout_qids]


def annotation_record(key: tuple[str, str]) -> dict:
    name, qid = key
    return {"id": f"{name}|{qid}", "question": gold_queries[qid], "passages": context_of(key), "answer": answers[key]}


annotation_records = [annotation_record(key) for key in annotation_keys]
holdout_records = [annotation_record(key) for key in holdout_keys]
assert not ({annotation.content_digest(r) for r in annotation_records}
            & {annotation.content_digest(r) for r in holdout_records}), "одинаковое содержимое в dev и holdout"
# в data/, а не в gold/: в выборках тексты чанков корпуса; в git — только оценки по id ответа и хэшу
annotation.write_sample(DATA / "kt5" / "annotation_dev.json", annotation_records)
annotation.write_sample(DATA / "kt5" / "annotation_holdout.json", holdout_records)
pd.DataFrame({"dev": pd.Series([k[0] for k in annotation_keys]).value_counts(),
              "holdout": pd.Series([k[0] for k in holdout_keys]).value_counts()}).rename_axis("ответов в выборке")

# %% [markdown]
# ## 4. LLM-судья: итерации промпта и детерминизм
#
# Каждая версия промпта (`judge.PROMPTS`) оценивает 30 ответов выборки dev дважды при t = 0: второй прогон —
# с отдельным кэшем (`llm_cache/kt5_repeat`), модель отвечает заново. Agreement — доля ответов с тем же баллом;
# порог задания — 85%. Шкалы `judge.SCALES` во всех версиях одинаковые — те же, что видит человек.
#
# - **v1** — шкала и короткое обоснование перед баллом.
# - **v2** — та же шкала, но сначала свободный разбор (части вопроса, утверждения, нужные факты) и правила подсчёта
#   для пограничных случаев v1; при колебании — меньший балл.
# - **v3** — вместо свободного разбора короткие структурированные поля (неподтверждённые утверждения, части вопроса
#   с отметкой «отвечена», нужные и пропущенные факты, есть ли ответ в контексте) и балл по правилу из них.
# - **v4** — v3, в которой изменена только достоверность: контрольная выборка (раздел 5) показала, что v3 считает
#   неверное число «неточностью пересказа»; противоречие фрагментам выделено в отдельное поле и правило.

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
# ## 5. Контрольная выборка достоверности: ловит ли судья выдумку
#
# Человек поставил достоверность 5 всем ответам dev — на такой выборке судья, всегда ставящий 5, неотличим
# от хорошего. Контроль: ответы, которые человек признал достоверными (≥ 4, не отказы), и две их искажённые копии
# (`gigarag.perturbation`): одно число заменено значением, которого нет во фрагментах, и добавлено выдуманное
# утверждение со ссылкой [1] и числом, которого нет во фрагментах. Метка — по построению: исходный ответ
# достоверен, искажённый — нет (по шкале — не выше 3). «Поймал» — балл ≤ 3; kappa — между меткой и «балл ≤ 3».
#
# Контроль строится на dev (подбор версии) и отдельно на holdout (проверка замороженной версии).

# %%
# только оценки, сделанные для текущего содержимого ответа (хэш совпадает); неполная разметка останавливает расчёт
human = annotation.valid_labels(annotation.read_labels(ROOT / "gold" / "kt5_labels_dev.json"), annotation_records)
human_holdout = annotation.valid_labels(annotation.read_labels(ROOT / "gold" / "kt5_labels_holdout.json"),
                                        holdout_records)
for name, labels, records in (("dev", human, annotation_records), ("holdout", human_holdout, holdout_records)):
    missing = [r["id"] for r in records if set(labels.get(r["id"], {})) != set(judge.CRITERIA)]
    assert not missing, f"разметка {name} неполная или сделана для другого текста: {missing}"
ORIGINAL, SWAPPED, CLAIMED = "исходный", "число заменено", "добавлен факт"


def control_cases(keys, labels) -> dict:
    """(ключ ответа, вид) → (вопрос, фрагменты, текст ответа) для подтверждённых человеком ответов и их искажений."""
    rng = random.Random(42)
    cases = {}
    for key in keys:
        if is_abstention(answers[key]) or labels[f"{key[0]}|{key[1]}"]["faithfulness"] < 4:
            continue
        variants = {ORIGINAL: answers[key], SWAPPED: perturbation.swap_value(answers[key], context_of(key), rng),
                    CLAIMED: perturbation.add_invented_claim(answers[key], context_of(key), rng)}
        cases |= {(key, kind): (gold_queries[key[1]], context_of(key), text)
                  for kind, text in variants.items() if text is not None}
    return cases


def control_scores(cases, version: int) -> dict:
    responses = parallel_map(lambda case: chat.complete(judge.judge_request("faithfulness", *cases[case], version)),
                             list(cases))
    return {case: judge.parse_verdict(r.content).score for case, r in zip(cases, responses, strict=True)}


def control_summary(cases, scored) -> dict:
    detection = perturbation.detection(
        [(kind, scored[(key, kind)]) for key, kind in cases], original=ORIGINAL)
    return {"ложная тревога на исходных, %": 100 * detection.rates[ORIGINAL],
            f"поймано: {SWAPPED}, %": 100 * detection.rates[SWAPPED],
            f"поймано: {CLAIMED}, %": 100 * detection.rates[CLAIMED], "κ (метка, балл ≤ 3)": detection.kappa}


control_dev = control_cases(annotation_keys, human)
control_dev_scores = {version: control_scores(control_dev, version) for version in judge.PROMPTS}
print("контроль dev:", pd.Series([kind for _, kind in control_dev]).value_counts().to_dict())
pd.DataFrame({f"v{v}": control_summary(control_dev, control_dev_scores[v]) for v in judge.PROMPTS}).T.round(2)

# %% [markdown]
# ## 6. Валидация человеком: weighted Cohen's kappa
#
# Человек разметил ответы по тем же шкалам, не видя оценок судьи, конфигурации и эталона.
#
# - **dev** (30 вопросов, `gold/kt5_labels_dev.json`) — подбор версии судьи;
# - **holdout** (20 других вопросов, `gold/kt5_labels_holdout.json`) — итоговая kappa и итоговый контроль
#   достоверности замороженной версии.
#
# Kappa — с квадратичными весами (`stats.weighted_kappa`), ответы судьи не по формату исключаются из пары.
# **Версия судьи** — среди прошедших порог детерминизма (≥ 85% по всем критериям, без ответов не по формату)
# та, у которой выше средняя kappa на dev по трём критериям: релевантность и использование — против человека,
# достоверность — против меток контрольной выборки dev (у человека на dev все пятёрки, и kappa с ним не определена
# по смыслу).

# %%
def kappa(verdicts, labels, criterion: str, keys) -> float:
    pairs = [(verdicts[(criterion, key)].score, labels[f"{key[0]}|{key[1]}"][criterion]) for key in keys]
    pairs = [(j, h) for j, h in pairs if j is not None]
    return weighted_kappa([j for j, _ in pairs], [h for _, h in pairs], judge.SCORES)


def closeness(verdicts, labels, keys, tolerance: int) -> dict[str, float]:
    """Доля ответов, где судья и человек разошлись не больше чем на `tolerance` баллов, % (по критериям)."""
    def share(criterion: str) -> float:
        pairs = [(verdicts[(criterion, key)].score, labels[f"{key[0]}|{key[1]}"][criterion]) for key in keys]
        pairs = [(j, h) for j, h in pairs if j is not None]
        return 100 * within_tolerance([j for j, _ in pairs], [h for _, h in pairs], tolerance)
    return {c: share(c) for c in judge.CRITERIA}


kappas = pd.DataFrame({f"v{version}": {c: kappa(first, human, c, annotation_keys) for c in judge.CRITERIA}
                       for version, (first, _) in runs.items()}).T
selection = pd.DataFrame({f"v{v}": {"κ relevance": kappas.loc[f"v{v}", "relevance"],
                                    "κ utilization": kappas.loc[f"v{v}", "utilization"],
                                    "κ faithfulness (контроль)": control_summary(
                                        control_dev, control_dev_scores[v])["κ (метка, балл ≤ 3)"]}
                          for v in judge.PROMPTS}).T
selection["среднее"] = selection.mean(axis=1)
stable = [v for v in judge.PROMPTS if determinism.loc[f"v{v}", list(judge.CRITERIA)].min() >= 85
          and determinism.loc[f"v{v}", "ответов не по формату"] == 0]
JUDGE_VERSION = max(stable, key=lambda v: selection.loc[f"v{v}", "среднее"])
print("прошли порог детерминизма:", [f"v{v}" for v in stable], "→ версия судьи для КТ 5:", f"v{JUDGE_VERSION}")
display(kappas.round(3))
selection.round(3)

# %% [markdown]
# ### Итоговая проверка замороженной версии на holdout

# %%
holdout_cases = {key: (gold_queries[key[1]], context_of(key), answers[key]) for key in holdout_keys}
holdout_verdicts = judge.judge_answers(chat, holdout_cases, JUDGE_VERSION)
control_holdout = control_cases(holdout_keys, human_holdout)
print("контроль holdout:", pd.Series([kind for _, kind in control_holdout]).value_counts().to_dict())
holdout_table = pd.DataFrame({
    "κ с человеком": {c: kappa(holdout_verdicts, human_holdout, c, holdout_keys) for c in judge.CRITERIA},
    "точно, %": closeness(holdout_verdicts, human_holdout, holdout_keys, 0),
    "±1, %": closeness(holdout_verdicts, human_holdout, holdout_keys, 1)})
display(holdout_table.round(3))
pd.Series(control_summary(control_holdout, control_scores(control_holdout, JUDGE_VERSION))).round(2).to_frame(
    f"контроль достоверности, holdout, v{JUDGE_VERSION}")

# %%
human_table = pd.DataFrame([human_holdout[f"{key[0]}|{key[1]}"] for key in holdout_keys])
judge_table = pd.DataFrame([{c: holdout_verdicts[(c, key)].score for c in judge.CRITERIA} for key in holdout_keys])
display(pd.DataFrame({"человек": human_table.mean(), "судья": judge_table.mean()}).T.round(2))
for criterion in judge.CRITERIA:  # матрица «человек × судья» на holdout: где именно расходятся
    display(pd.crosstab(human_table[criterion], judge_table[criterion], rownames=[f"{criterion}: человек"],
                        colnames=["судья"]))

# %% [markdown]
# ## 7. Оценка всех ответов финальным судьёй
#
# 100 ответов (50 вопросов × 2 конфигурации) × 3 критерия. Релевантность и использование — средним баллом 1–5,
# достоверность — долей достоверных ответов (балл ≥ 4, `judge.FAITHFUL_MIN`): контроль на искажениях проверяет
# только эту границу, а не калибровку шкалы внутри ошибок (q08 прошлого прогона: судья поставил 1 там, где по
# шкале 3), поэтому средний балл достоверности не публикуется. Отдельно — поведение на отказах: доля «не знаю»
# на негативных (должна быть 100%) и ложные отказы на вопросах, у которых ответ есть.

# %%
all_cases = {key: (gold_queries[key[1]], context_of(key), answers[key]) for key in answer_keys}
final = judge.judge_answers(chat, all_cases, JUDGE_VERSION)
judged = answer_table.assign(**{c: [final[(c, key)].score for key in answer_keys] for c in judge.CRITERIA})
print("ответов судьи не по формату:", int(judged[list(judge.CRITERIA)].isna().sum().sum()))
FAITHFUL = "достоверных, %"
judged[FAITHFUL] = 100 * (judged["faithfulness"] >= judge.FAITHFUL_MIN)
GENERATION_METRICS = [FAITHFUL, *judge.SCALED]
display(judged.groupby(["config", "split"], sort=False)[GENERATION_METRICS].mean().round(2))
display(judged[judged["faithfulness"] < judge.FAITHFUL_MIN][["config", "qid", "split", "faithfulness"]])
judged[judged["split"] == "validation"].pivot_table(index="config", columns="qtype", values=GENERATION_METRICS,
                                                    sort=False).round(2)

# %%
abstentions = judged.assign(negative=judged["qtype"] == "negative").groupby(["config", "negative"], sort=False)[
    "отказ"].mean().mul(100).unstack().rename(columns={True: "отказ на негативных, %",
                                                       False: "ложный отказ на остальных, %"})
abstentions.round(1)

# %% [markdown]
# ## 8. Итоговая таблица
#
# Поиск — R@5 и NDCG@5 (как в КТ 3–4), генерация — доля достоверных ответов и средние баллы релевантности
# и использования, отказы — по всем 50 вопросам.
# Задержка — медиана поиска (с реранкингом) и медиана ответа генератора по 50 вопросам.

# %%
qrels = build_qrels(items, corpus.chunks, corpus.clean.aliases)
retrieval = pd.concat([metrics.per_question({qid: ranked_ids(t.ranking) for qid, t in traces[name].items()}, qrels,
                                            items).assign(config=name) for name in (BASELINE, BEST)])
summary = pd.DataFrame({
    name: {
        "R@5": 100 * retrieval.query("config == @name and split == 'validation'")["R@5"].mean(),
        "NDCG@5": 100 * retrieval.query("config == @name and split == 'validation'")["NDCG@5"].mean(),
        **{c: judged.query("config == @name and split == 'validation'")[c].mean() for c in GENERATION_METRICS},
        "отказ на негативных, %": abstentions.loc[name, "отказ на негативных, %"],
        "ложный отказ, %": abstentions.loc[name, "ложный отказ на остальных, %"],
        "поиск, мс (медиана)": 1000 * np.median([t.total_seconds for t in traces[name].values()]),
        "генерация, мс (медиана)": 1000 * np.median([responses[(name, q)].seconds for q in gold_queries]),
    } for name in (BASELINE, BEST)}).T
summary.round(2)

# %% [markdown]
# ### Артефакты эксперимента и идентичность модели
#
# У DeepSeek нет неизменяемого ID версии модели: алиас `deepseek-flash` (DeepSeek-V4.1-Flash) обновляется
# на сервере. Поэтому результаты, на которых построены выводы, сохраняются в git: ответы генератора
# (`gold/kt5_answers.jsonl`) и все вердикты финального судьи (`gold/kt5_verdicts.jsonl`) — с `system_fingerprint`
# и id ответа API там, где они записаны (поле появилось в кэше после генерации ответов и первых версий судьи);
# `requested_model` — алиас из запроса, `model` — что вернул API (пусто, если не записано).
# Повторный запуск после обновления модели можно сравнить с этими файлами построчно.

# %%
atomic_write_jsonl(ROOT / "gold" / "kt5_answers.jsonl", [
    {"config": name, "qid": qid, "answer": answers[(name, qid)], "context": contexts[name][qid],
     "requested_model": generation.MODEL, "model": responses[(name, qid)].model,
     "fingerprint": responses[(name, qid)].fingerprint,
     "response_id": responses[(name, qid)].response_id, "prompt_version": generation.PROMPT_VERSION}
    for name, qid in answer_keys])
atomic_write_jsonl(ROOT / "gold" / "kt5_verdicts.jsonl", [
    {"config": key[0], "qid": key[1], "criterion": criterion, "judge_version": JUDGE_VERSION,
     "score": verdict.score, "rationale": verdict.rationale, "requested_model": judge.MODEL, "model": verdict.model,
     "fingerprint": verdict.fingerprint, "response_id": verdict.response_id}
    for (criterion, key), verdict in final.items()])
fingerprints = pd.Series([r.fingerprint or "не записан" for r in responses.values()] +
                         [v.fingerprint or "не записан" for v in final.values()],
                         index=["генерация"] * len(responses) + ["судья"] * len(final))
fingerprints.groupby(level=0).value_counts().rename("ответов API").to_frame()

# %% [markdown]
# ## 9. Выводы
#
# Метрики — validation (29 вопросов с доказательствами и 7 негативных), если не сказано иное.
#
# **Генерация.** Промпт «только по фрагментам, иначе „Не знаю“» отказывается на всех 10 негативных вопросах
# в обеих конфигурациях. Достоверность по финальному судье — 5,0 на validation; на test у best retrieval 4,71:
# один ответ (q08) приписал фрагменту [2] правку, которой там нет. Слабое место — абстрактные вопросы: ложные
# отказы — 50% у baseline и 40% у best retrieval, использование контекста — 2,71 и 3,29.
#
# **Судья.** Порог детерминизма проходят v1, v3 и v4; v2 — нет: свободный разбор перед баллом увеличил разброс.
# v3 заменила разбор короткими полями, v4 — v3 с новым правилом достоверности. Его потребовала контрольная выборка:
# v3 ловила заменённое число лишь в 33% случаев, считая его «неточностью пересказа»; v4 — в 95%, выдуманный факт —
# в 100%, ложная тревога — 1 из 23 (q08: ошибку в ответе судья нашёл, но завысил её тяжесть до 1 вместо 3).
# Версия выбрана на dev (30 ответов + контроль dev) и проверена на holdout, который размечен после заморозки:
# κ релевантности 0,72, использования контекста 0,93 — судья надёжен; контроль достоверности на holdout —
# 22 искажения из 22 пойманы, ложных тревог нет, κ = 1,0. Kappa достоверности против человека не определена:
# у человека и судьи на holdout все пятёрки (совпадение 100%) — проверку этого критерия даёт контроль.
#
# **Вклад поиска.** Best retrieval поднимает R@5 66,0 → 71,8 и NDCG@5 60,2 → 67,4, использование контекста —
# 4,47 → 4,56, но релевантность ответов падает 4,19 → 3,94, и всё падение — на абстрактных (3,57 → 2,14). Причина —
# в судье без эталона и в генераторе: лучший поиск чаще приносит частично полезный контекст, а генератор всё равно
# отказывается или отвечает мимо, и такой отказ получает 1 вместо 5 при пустом контексте. Это узкое место
# генерации, а не поиска: на абстрактных вопросах модель не собирает ответ из разрозненных фрагментов.
