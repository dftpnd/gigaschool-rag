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
# # КТ 1 — золотой набор и baseline без RAG
#
# Набор построен скриптом `scripts/build_gold.py` (стадии generate → stylize → label → review → finalize),
# baseline — скриптом `scripts/baseline_no_rag.py`. Методика описана в `README.md`; здесь — результат:
# 1. Состав набора и баланс разбиения validation/test.
# 2. Лексическое разнообразие формулировок.
# 3. Покрытие корпуса: источники, патчи, число релевантных документов на вопрос.
# 4. Ревью кандидатов: что отклонено и почему.
# 5. Baseline: как отвечает LLM без контекста — нужен ли RAG.

# %%
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from gigarag import gold_coverage as coverage
from gigarag.answer_eval import VERDICTS, load_verdicts, summary_by_type, verdict_table
from gigarag.corpus import read_jsonl
from gigarag.gold import QUESTION_TYPES, load_gold

pd.set_option("display.max_colwidth", 160)
plt.rcParams["figure.dpi"] = 110

DATA = Path.cwd() / "data"
GOLD = Path.cwd() / "gold"
rows, CORPUS_SHA = read_jsonl(DATA / "corpus.jsonl")
sources = {r["doc_id"]: r["source"] for r in rows}
patches = {r["doc_id"]: r["patch"] for r in rows}
corpus_patches = {r["patch"] for r in rows if r["source"] == "patchnotes"}

items = load_gold(GOLD / "gold.jsonl")
questions = coverage.question_frame(items)
evidence = coverage.evidence_frame(items, sources, patches)
print(len(items), "вопросов; sha256 корпуса:", CORPUS_SHA[:16])

# %% [markdown]
# ## 1. Состав и разбиение
#
# Разбиение стратифицировано по типу вопроса (`gigarag.gold.stratified_split`, seed 42): доля test ≈ 30%
# внутри каждого типа.

# %%
coverage.split_balance(questions)

# %% [markdown]
# Примеры — по одному вопросу каждого типа:

# %%
examples = [next(i for i in items if i.qtype == t) for t in QUESTION_TYPES]
pd.DataFrame([{"qid": i.qid, "qtype": i.qtype, "question": i.question, "answer": i.answer} for i in examples])

# %% [markdown]
# ## 2. Лексическое разнообразие
#
# Часть вопросов переформулирована отдельным запросом к LLM: опечатки, разговорный стиль, сленговые
# и сокращённые имена героев/предметов, короткий поисковый запрос, развёрнутый вопрос. Остальные
# оставлены в нейтральной формулировке генератора.

# %%
by_style = questions.groupby("style").agg(questions=("qid", "size"), median_words=("words", "median"))
by_style["share_%"] = (by_style["questions"] / len(questions) * 100).round(1)
by_style.sort_values("questions", ascending=False)

# %%
pd.crosstab(questions["style"], questions["split"])

# %% [markdown]
# Как стилизация изменила формулировку (по одному примеру на стиль):

# %%
styled = {i.styles[0]: i for i in reversed(items) if i.styles}  # reversed: остаётся первый вопрос стиля
pd.DataFrame([{"style": s, "original": i.original_question, "styled": i.question} for s, i in styled.items()])

# %%
fig, ax = plt.subplots(figsize=(7, 3))
ax.hist(questions["words"], bins=20, color="#4C72B0")
ax.set_xlabel("длина вопроса, слов")
ax.set_ylabel("вопросов")
ax.set_title("Длины вопросов")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 3. Покрытие корпуса
#
# Оценка 2 — отрывок содержит ответ или его необходимую часть, 1 — по теме, но без ответа.
# Ретривер в КТ 3 оценивается по документам с оценкой 2; оценка 1 нужна для NDCG.

# %%
questions.groupby("qtype")[["relevant_docs", "related_docs"]].describe().round(1).reindex(QUESTION_TYPES)

# %% [markdown]
# Из каких источников берутся ответы — доля вопросов типа (в %), у которых среди документов с ответом
# есть документ источника:

# %%
coverage.source_coverage(evidence, questions)

# %%
questions["answer_sources"] = coverage.sources_per_question(evidence, questions).to_numpy()
pd.crosstab(questions["qtype"], questions["answer_sources"]).reindex(QUESTION_TYPES)

# %% [markdown]
# Какие патчи покрыты: число вопросов, ответ на которые опирается на документ патча (патчноут, статью
# dota2.ru или комментарий к ней). Посты Telegram к патчу не привязаны и здесь не учитываются.

# %%
by_patch = coverage.patch_coverage(evidence, corpus_patches)
print("патчей в корпусе:", len(by_patch), "покрыто хотя бы одним вопросом:", (by_patch["questions"] > 0).sum())
fig, ax = plt.subplots(figsize=(10, 3))
ax.bar(by_patch.index, by_patch["questions"], color="#4C72B0")
ax.set_ylabel("вопросов")
ax.set_title("Покрытие патчей золотым набором")
ax.tick_params(axis="x", rotation=90)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 4. Ревью кандидатов
#
# Каждый кандидат прошёл автоматические проверки (цитаты генератора есть в документе, судья подтвердил
# затравку, нет дублей) и ручное ревью: `accept` — в набор, `reserve` — годен, но сверх квоты типа,
# `reject` — отклонён.

# %%
records = {r["cid"]: r for r in map(json.loads, (DATA / "gold" / "labeled.jsonl").read_text(encoding="utf-8")
                                    .splitlines())}
review = coverage.review_frame(json.loads((GOLD / "review.json").read_text(encoding="utf-8")), records)
pd.crosstab(review["qtype"], review["decision"], margins=True, margins_name="ВСЕГО")

# %%
review.loc[review["decision"] == "reject", ["cid", "qtype", "question", "reason"]]

# %% [markdown]
# ## 5. Baseline без RAG
#
# Две модели отвечают на вопросы validation без контекста (t = 0, системный промпт разрешает «Не знаю»),
# судья `gpt-4.1` сравнивает ответ с эталоном. Балл: correct — 1, partial — 0,5; `hallucinated` —
# содержательный неверный ответ (не отказ).

# %%
verdicts = load_verdicts(GOLD / "baseline_no_rag.jsonl")
summary = summary_by_type(verdicts)
summary

# %%
table = verdict_table(verdicts)
shares = (pd.crosstab([table["model"], table["qtype"]], table["verdict"], normalize="index")
          .reindex(columns=list(VERDICTS), fill_value=0) * 100)
colors = {"correct": "#55A868", "partial": "#DD8452", "incorrect": "#C44E52", "abstained": "#8C8C8C"}
fig, axes = plt.subplots(1, 2, figsize=(11, 3.5), sharey=True)
for ax, model in zip(axes, shares.index.get_level_values("model").unique(), strict=True):
    part = shares.loc[model].reindex(QUESTION_TYPES)
    part.plot.barh(stacked=True, ax=ax, color=[colors[v] for v in VERDICTS], legend=False, width=0.7)
    ax.set_title(model)
    ax.set_xlabel("% вопросов")
    ax.invert_yaxis()
axes[0].set_ylabel("")
fig.legend(VERDICTS, loc="lower center", ncol=len(VERDICTS), fontsize=8)
plt.tight_layout(rect=(0, 0.08, 1, 1))
plt.show()

# %% [markdown]
# Ответы без контекста на негативные вопросы (в корпусе ответа нет) — проверка, умеет ли модель
# признать незнание:

# %%
negative = table[table["qtype"] == "negative"]
negative[["qid", "model", "verdict", "answer"]].sort_values(["qid", "model"])

# %% [markdown]
# ## Выводы
#
# **Состав.** 50 вопросов: 15 фактических, 15 составных, 10 абстрактных, 10 негативных. Разбиение
# 36 validation / 14 test (28% test), в каждом типе в test 27–30% вопросов.
#
# **Лексическое разнообразие.** 70% вопросов переформулированы: опечатки — 18% (цель ~15%),
# разговорный стиль — 16%, короткий запрос — 14% (медиана 7 слов), развёрнутый вопрос — 12%
# (медиана 55 слов), сленговые имена — 10%. Все стили есть в обеих частях разбиения; опечатки и сленговые
# имена попали в test по одному разу — для сравнения по стилям надёжнее только validation.
#
# **Покрытие.** Число документов с ответом растёт с типом вопроса: в среднем 2,1 у фактических, 3,9 у
# составных, 8,3 у абстрактных. Это задумано: составному вопросу нужно собрать ответ из нескольких
# документов, абстрактному — из обсуждений. У 10 из 15 составных ответ лежит в 3–4 источниках,
# у фактических чаще в 1–2: правка из патчноута почти всегда продублирована в статье dota2.ru.
# Патчноуты — основа фактических и составных вопросов (73% и 100%). Абстрактные вопросы опираются на
# Telegram (80% @Dota2, 50% @gabenews) и комментарии — «грязная» часть корпуса тоже входит в набор.
# Вопросы затрагивают 27 из 36 патчей окна корпуса; непокрытые 9 — мелкие буквенные хотфиксы
# (7.34c–e, 7.35d, 7.37e, 7.38b, 7.39e, 7.41a, 7.41c). Крупные патчи 7.38 и 7.41 покрыты сильнее всех.
#
# **Ревью.** Из 74 кандидатов отклонены 12. Причины: искусственная связка двух несвязанных тем
# в составном вопросе (4), неподтверждённый или неустойчивый эталон (3), вопрос без контекста или
# неоднозначный (3), стилизация исказила вопрос (2). Ещё 12 годных кандидатов — в резерве сверх квоты.
#
# **Baseline без RAG.** Без контекста обе модели набирают около 30% балла (gpt-4.1-mini — 30,6%,
# gpt-4.1 — 29,2%) и выдумывают в 53–58% ответов. Отказываются отвечать они редко (3–6%), хотя
# системный промпт разрешает «Не знаю». Хуже всего фактические и составные вопросы (балл 14–23%,
# выдумки ~70%): конкретные числа правок по патчам модель не помнит, а уверенно сочиняет.
# Абстрактные проходят частично (43–50%): общие тезисы про «мету» модель знает из предобучения.
# Негативные вопросы делятся на два вида. О ещё не вышедшем патче 7.42 и о несуществующем герое
# Frost Warden обе модели честно отвечают «не знаю». О патчах до окна корпуса (7.29, 7.32) они отвечают
# по памяти: такой ответ не опирается на корпус и засчитывается как выдумка, даже если он случайно верен.
# Для несуществующего «Phantom Golem» обе модели сочинили список изменений. Большая модель не лучше малой.
# Значит, RAG нужен в первую очередь ради точных фактов по версиям и ради отказа там, где в корпусе
# ответа нет. Генератор для КТ 5 — gpt-4.1-mini: качество без контекста то же, стоимость ниже.
