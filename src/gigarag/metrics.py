"""Метрики ретрива по чанкам: Hit@k, Recall@k, Precision@k, MRR.

Ранжирование — список индексов чанков по убыванию score. Релевантные — чанки с доказательством оценки 2.
- Hit@k — в топ-k есть хотя бы одно доказательство вопроса.
- Recall@k — доля доказательств вопроса, найденных в топ-k (доказательство найдено, если найден
  хоть один содержащий его чанк). Единица — доказательство, а не чанк: длинный документ, порезанный
  на много чанков, не раздувает знаменатель.
- Precision@k — доля релевантных чанков в топ-k. Разметка gold неполна (судья видел только кандидатов
  трёх поисковиков), поэтому это оценка снизу.
- MRR — 1 / место первого релевантного чанка (0, если его нет в выдаче).
Негативные вопросы доказательств не имеют и в метриках ретрива не участвуют.
"""

from collections.abc import Mapping, Sequence

import pandas as pd

from gigarag.gold import RELEVANT, GoldItem
from gigarag.relevance import EvidenceChunks, Qrels, relevant_chunk_ids

K_VALUES = (1, 5, 10)


def hit_at_k(ranking: Sequence[int], evidence: Sequence[EvidenceChunks], k: int) -> float:
    top = set(ranking[:k])
    return float(any(e.chunks & top for e in evidence if e.grade >= RELEVANT))


def recall_at_k(ranking: Sequence[int], evidence: Sequence[EvidenceChunks], k: int) -> float:
    """|найденные доказательства| / |доказательства с оценкой 2|."""
    relevant = [e for e in evidence if e.grade >= RELEVANT]
    top = set(ranking[:k])
    return sum(bool(e.chunks & top) for e in relevant) / len(relevant) if relevant else 0.0


def precision_at_k(ranking: Sequence[int], evidence: Sequence[EvidenceChunks], k: int) -> float:
    """|релевантные чанки в топ-k| / k."""
    relevant = relevant_chunk_ids(evidence)
    return sum(i in relevant for i in ranking[:k]) / k


def reciprocal_rank(ranking: Sequence[int], evidence: Sequence[EvidenceChunks]) -> float:
    relevant = relevant_chunk_ids(evidence)
    return next((1 / rank for rank, i in enumerate(ranking, 1) if i in relevant), 0.0)


def has_evidence(evidence: Sequence[EvidenceChunks]) -> bool:
    return any(e.grade >= RELEVANT for e in evidence)


CONTEXT_K = 5


def per_question(rankings: Mapping[str, Sequence[int]], qrels: Qrels, items: Sequence[GoldItem],
                 k_values: Sequence[int] = K_VALUES, chunk_tokens: Sequence[int] | None = None) -> pd.DataFrame:
    """Метрики каждого вопроса с доказательствами: qid, qtype, split и метрики при каждом k.
    С `chunk_tokens` — ещё `tokens@5`: сколько токенов контекста занимают топ-5 (цена выдачи для генератора)."""
    rows = []
    for item in items:
        evidence = qrels[item.qid]
        if not has_evidence(evidence):
            continue
        ranking = rankings[item.qid]
        row = {"qid": item.qid, "qtype": item.qtype, "split": item.split, "MRR": reciprocal_rank(ranking, evidence)}
        for k in k_values:
            row |= {f"Hit@{k}": hit_at_k(ranking, evidence, k), f"R@{k}": recall_at_k(ranking, evidence, k),
                    f"P@{k}": precision_at_k(ranking, evidence, k)}
        if chunk_tokens is not None:
            row[f"tokens@{CONTEXT_K}"] = sum(chunk_tokens[i] for i in ranking[:CONTEXT_K])
        rows.append(row)
    return pd.DataFrame(rows)


METRIC_COLUMNS = ("Hit@5", "R@5", "R@10", "P@5", "MRR")


def summarize(per_q: pd.DataFrame, by: Sequence[str] = ("split",), columns: Sequence[str] = METRIC_COLUMNS
              ) -> pd.DataFrame:
    """Средние метрики по группам, в процентах (MRR — тоже ×100 для единообразия таблиц)."""
    return (per_q.groupby(list(by))[list(columns)].mean() * 100).round(1)


TABLE_COLUMNS = ("Hit@5", "R@5", "R@10", "MRR")


def comparison_table(per_q: pd.DataFrame, split: str = "validation", columns: Sequence[str] = TABLE_COLUMNS,
                     by: Sequence[str] = ("config",)) -> pd.DataFrame:
    """Строка — конфигурация (в порядке появления), столбцы — (поисковик, метрика), значения в процентах."""
    subset = per_q[per_q["split"] == split]
    table = subset.groupby([*by, "retriever"], sort=False)[list(columns)].mean().mul(100).round(1)
    return table.unstack("retriever").swaplevel(axis=1).sort_index(axis=1, level=0, sort_remaining=False)


def paired_changes(per_q: pd.DataFrame, base: str, other: str, metric: str = "R@5",
                   split: str = "validation") -> pd.DataFrame:
    """Сколько вопросов `other` улучшила, ухудшила и не изменила по `metric` относительно `base` —
    для каждого поисковика. На 29 вопросах средняя разница в пару пунктов может быть шумом, а счёт
    «лучше/хуже» показывает, за счёт скольких вопросов она получена."""
    subset = per_q[per_q["split"] == split]
    wide = subset.pivot_table(index=["retriever", "qid"], columns="config", values=metric)
    delta = wide[other] - wide[base]
    return pd.DataFrame({"лучше": (delta > 0).groupby(level=0).sum(), "хуже": (delta < 0).groupby(level=0).sum(),
                         "без изменений": (delta == 0).groupby(level=0).sum()})
