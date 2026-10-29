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
# # Финальный сквозной отчёт: система целиком
#
# Выводы — в `reports/final.md`, здесь — цифры. Ключевые конфигурации поиска КТ 3–4 проходят весь путь системы
# КТ 5: поиск → топ-5 → генерация → судья. Так их можно сравнить не только по поиску, но и по ответу.
#
# - **Поиск** — как в КТ 4: ответы multi-query и LLM-реранкера — из кэша КТ 4, cross-encoder — `mmarco-mMiniLMv2`.
# - **Генерация и судья** — как в КТ 5: `deepseek-flash` без рассуждения, промпт генерации версии 1, судья v5
#   (заморожен в КТ 5). Кэш КТ 5: для baseline КТ 3 и best КТ 4 это те же ответы и вердикты, что в отчёте КТ 5.
# - **Метрики ответа.** Достоверность — доля ответов с баллом ≥ 4: проверена контролем на искажениях (κ = 1,0
#   на holdout КТ 5). Релевантность — средний балл 1–5, **не валидирована** (κ 0,29 на holdout КТ 5): приведена,
#   но в выводах о сравнении не используется. Отказы — по всем 50 вопросам.
# - **Метрики поиска и ответа** — validation (29 вопросов с доказательствами, 7 негативных).

# %%
from pathlib import Path

import pandas as pd

from gigarag import answer_eval, baseline, generation, judge, metadata, metrics
from gigarag.cache import CacheStore
from gigarag.corpus import Document, read_jsonl
from gigarag.enrich import PROVIDER, PROVIDER_BASE_URL
from gigarag.gold import load_gold
from gigarag.llm import CachedChat, OpenAIChat
from gigarag.llm_rerank import GRADES_PROMPT, ListwiseLlmReranker
from gigarag.parallel import parallel_map
from gigarag.patterns import is_abstention
from gigarag.pipeline import DENSE, Pipeline, run_all
from gigarag.query_expansion import multi_query_expander
from gigarag.relevance import build_qrels
from gigarag.rerank import CrossEncoderReranker, CrossEncoderSpec, load_cross_encoder
from gigarag.search import ranked_ids

pd.set_option("display.max_colwidth", 120)
pd.set_option("display.width", 220)

ROOT = Path.cwd().parent  # ноутбук исполняется из notebooks/
DATA = ROOT / "data"
rows, CORPUS_SHA = read_jsonl(DATA / "corpus.jsonl")
items = load_gold(ROOT / "gold" / "gold.jsonl")
by_qid = {i.qid: i for i in items}
gold_queries = {i.qid: i.question for i in items}
client = OpenAIChat.from_env(ROOT / ".env", PROVIDER, PROVIDER_BASE_URL)
kt4_chat = CachedChat(client, DATA / "llm_cache")         # multi-query и реранкер — ответы КТ 4
chat = CachedChat(client, DATA / "llm_cache" / "kt5")     # генерация и судья — кэш КТ 5
JUDGE_VERSION = 5  # заморожен в КТ 5 (коммит 69b9fd5) до разметки holdout

stack = baseline.load_search_stack(tuple(Document(**r) for r in rows), CacheStore(DATA / "cache" / "kt4", CORPUS_SHA))
corpus, passages = stack.corpus, stack.corpus.dense_texts
qrels = build_qrels(items, corpus.chunks, corpus.clean.aliases)
patch_masks = metadata.filter_masks(metadata.patch_filter(corpus.meta, corpus.texts), corpus.meta, gold_queries)
stack.encode_queries(["прогрев"])  # первая инициализация CUDA не в счёт
print(f"{len(corpus.chunks)} чанков, {len(items)} вопросов gold")

# %% [markdown]
# ## 1. Конфигурации: от baseline КТ 3 до финальной системы КТ 5

# %%
MMARCO = CrossEncoderSpec("cross-encoder/mmarco-mMiniLMv2-L12-H384-v1", "1427fd652930e4ba29e8149678df786c240d8825", 118)
CE_POOL, LLM_POOL = 20, 20  # лучшие пулы КТ 4
expand = multi_query_expander(kt4_chat)
llm_rerank = ListwiseLlmReranker(kt4_chat, passages, LLM_POOL, GRADES_PROMPT)
ce_rerank = CrossEncoderReranker(load_cross_encoder(MMARCO, half=True), passages, CE_POOL)
CONFIGS = {
    "КТ 3: dense BGE-M3": Pipeline(stack.retriever, stack.encode_queries, mode=DENSE),
    "КТ 3: hybrid + фильтр по патчу (baseline)": Pipeline(stack.retriever, stack.encode_queries),
    "КТ 4: + multi-query": Pipeline(stack.retriever, stack.encode_queries, expand=expand),
    "КТ 4: + cross-encoder mmarco": Pipeline(stack.retriever, stack.encode_queries, rerank=ce_rerank),
    "КТ 4: + LLM-реранкер (финальная система КТ 5)": Pipeline(stack.retriever, stack.encode_queries,
                                                              rerank=llm_rerank),
    "КТ 4: + multi-query + LLM-реранкер": Pipeline(stack.retriever, stack.encode_queries, expand=expand,
                                                   rerank=llm_rerank),
}
traces = {name: run_all(pipeline, gold_queries, patch_masks) for name, pipeline in CONFIGS.items()}
contexts = {name: {qid: [doc for doc, _ in t.ranking[:generation.CONTEXT_K]] for qid, t in runs.items()}
            for name, runs in traces.items()}
retrieval = pd.concat([metrics.per_question({q: ranked_ids(t.ranking) for q, t in traces[name].items()}, qrels,
                                            items).assign(config=name) for name in CONFIGS])
retrieval.query("split == 'validation'").groupby("config", sort=False)[["R@5", "NDCG@5"]].mean().mul(100).round(1)

# %% [markdown]
# ## 2. Генерация и судья для каждой конфигурации

# %%
keys = [(name, qid) for name in CONFIGS for qid in gold_queries]
cases = {key: (gold_queries[key[1]], [passages[d] for d in contexts[key[0]][key[1]]]) for key in keys}
responses = dict(zip(keys, parallel_map(lambda key: chat.complete(generation.answer_request(*cases[key])), keys),
                     strict=True))
verdicts = judge.judge_answers(chat, {key: (*cases[key], responses[key].content) for key in keys}, JUDGE_VERSION)
answers = pd.DataFrame([{
    "config": name, "qid": qid, "qtype": by_qid[qid].qtype, "split": by_qid[qid].split,
    "отказ": is_abstention(responses[(name, qid)].content),
    "достоверен": (verdicts[("faithfulness", (name, qid))].score or 0) >= judge.FAITHFUL_MIN,
    "relevance": verdicts[("relevance", (name, qid))].score,
    "поиск, с": traces[name][qid].total_seconds, "генерация, с": responses[(name, qid)].seconds,
} for name, qid in keys])
print("ответов судьи не по формату:", sum(v.score is None for v in verdicts.values()),
      "| fingerprint модели:", sorted({r.fingerprint for r in responses.values()}))

# %% [markdown]
# ## 3. Сводная таблица

# %%
validation = answers.query("split == 'validation'")
negative = answers["qtype"] == answer_eval.NEGATIVE
summary = pd.DataFrame({
    "R@5": retrieval.query("split == 'validation'").groupby("config", sort=False)["R@5"].mean() * 100,
    "NDCG@5": retrieval.query("split == 'validation'").groupby("config", sort=False)["NDCG@5"].mean() * 100,
    "достоверных, %": validation.groupby("config", sort=False)["достоверен"].mean() * 100,
    "relevance (не валидировано)": validation.groupby("config", sort=False)["relevance"].mean(),
    "отказ на негативных, %": answers[negative].groupby("config", sort=False)["отказ"].mean() * 100,
    "ложный отказ, %": answers[~negative].groupby("config", sort=False)["отказ"].mean() * 100,
    "поиск, мс": answers.groupby("config", sort=False)["поиск, с"].median() * 1000,
    "end-to-end, мс": (answers["поиск, с"] + answers["генерация, с"]).groupby(answers["config"], sort=False).median()
    * 1000,
}).loc[list(CONFIGS)]
summary.round(1)

# %% [markdown]
# ## 4. Где система теряет качество: поиск или генерация
#
# Для каждого вопроса validation с доказательствами (`answer_eval.loss_stage`): потеря на поиске — в топ-5 нет
# ни одного чанка с доказательством; потеря на генерации — доказательство в топ-5 есть, но модель отказалась или
# ответ недостоверен; ответ — доказательство найдено и ответ содержательный и достоверный. Полнота ответа
# не проверяется: судья релевантности не валидирован.

# %%
stages = retrieval.query("split == 'validation'")[["config", "qid", "qtype", "R@5"]].merge(
    answers[["config", "qid", "отказ", "достоверен"]], on=["config", "qid"])
stages["этап"] = [answer_eval.loss_stage(r, a, f) for r, a, f in stages[["R@5", "отказ", "достоверен"]].itertuples(
    index=False)]
display(pd.crosstab(stages["config"], stages["этап"]).loc[list(CONFIGS)])
FINAL = "КТ 4: + LLM-реранкер (финальная система КТ 5)"
final_stages = stages[stages["config"] == FINAL]
pd.crosstab(final_stages["qtype"], final_stages["этап"], margins=True, margins_name="всего")

# %% [markdown]
# ### Какие типы вопросов не решены
#
# Финальная система по типам вопросов validation: полнота поиска, доля отказов, доля вопросов, где найдено
# доказательство и дан достоверный ответ.

# %%
by_type = pd.DataFrame({
    "R@5": final_stages.groupby("qtype")["R@5"].mean() * 100,
    "отказов, %": final_stages.groupby("qtype")["отказ"].mean() * 100,
    "отвечено, %": final_stages.groupby("qtype")["этап"].apply(lambda s: (s == answer_eval.ANSWERED).mean()) * 100,
})
negatives = answers[(answers["config"] == FINAL) & negative & (answers["split"] == "validation")]
by_type.loc[answer_eval.NEGATIVE, "отказов, %"] = negatives["отказ"].mean() * 100
by_type.round(1)

# %% [markdown]
# ## 5. Прирост на единицу усилий
#
# Шаги, которые меняли поиск, — по исполненным ноутбукам КТ 2–4 (`reports/kt2.md` … `kt4.md`), прирост — на
# validation. Шаги КТ 2 меряются на dense до hybrid, поэтому сравнимы между собой, а не с шагами КТ 3–4.

# %%
pd.DataFrame([
    ("КТ 2", "чистка корпуса (все шаги, кроме подписей Telegram)", "dense Hit@5", 72.4, 79.3, "несколько правил, 0 мс"),
    ("КТ 2", "префикс метаданных к чанку", "dense R@5", 48.8, 53.3, "одна строка, 0 мс"),
    ("КТ 2", "фильтр по патчу из запроса", "dense R@5", 48.8, 53.9, "регулярка, 0 мс"),
    ("КТ 2", "LLM: вопросы к чанку вместо текста", "dense R@5", 48.8, 45.2, "$22,7 и 54 мин (резюме + вопросы)"),
    ("КТ 3", "hybrid BM25 + dense, alpha 0,6, фильтр по патчу", "R@5", 53.3, 66.0, "+21 мс на запрос"),
    ("КТ 4", "multi-query + RRF", "R@5", 66.0, 68.0, "+1,4 с, $0,22 / 1000"),
    ("КТ 4", "cross-encoder mmarco, пул 20", "R@5", 66.0, 62.1, "+25 мс"),
    ("КТ 4", "LLM-реранкер listwise, пул 20", "R@5", 66.0, 71.8, "+1,2 с, $1,41 / 1000"),
], columns=["КТ", "шаг", "метрика", "до", "после", "цена"]).assign(
    прирост=lambda t: (t["после"] - t["до"]).round(1))
