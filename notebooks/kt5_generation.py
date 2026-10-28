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

from gigarag import annotation, baseline, generation, judge, metadata, tokens
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
from gigarag.retrieval import Bm25Index
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
