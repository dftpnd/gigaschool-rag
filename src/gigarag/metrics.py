"""Метрики ретрива по чанкам: Hit@k, Recall@k, Precision@k, NDCG@k, MAP, MRR.

Ранжирование — список индексов чанков по убыванию score. Релевантные — чанки с доказательством оценки 2.
- Hit@k — в топ-k есть хотя бы одно доказательство вопроса.
- Recall@k — доля доказательств вопроса, найденных в топ-k (доказательство найдено, если найден
  хоть один содержащий его чанк). Единица — доказательство, а не чанк: длинный документ, порезанный
  на много чанков, не раздувает знаменатель.
- Precision@k — доля релевантных чанков в топ-k. Разметка gold неполна (судья видел только кандидатов
  трёх поисковиков), поэтому это оценка снизу.
- MRR — 1 / место первого релевантного чанка (0, если его нет в выдаче).
- NDCG@k — градуированная: выигрыш чанка 2^оценка − 1 (оценка 2 — ответ, 1 — по теме без ответа).
- MAP — среднее по вопросам AP@10 по релевантным чанкам.
Негативные вопросы доказательств не имеют и в метриках ретрива не участвуют; для них считается доля
«похожих, но не ответ» в выдаче (`negative_per_question`).
"""

import math
from collections.abc import Mapping, Sequence

import pandas as pd

from gigarag.gold import RELEVANT, GoldItem
from gigarag.relevance import EvidenceChunks, Qrels, chunk_grades, relevant_chunk_ids
from gigarag.retrieval import Ranking

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


def dcg(gains: Sequence[float]) -> float:
    """DCG = Σ_i gain_i / log2(i + 1), i — место с единицы."""
    return sum(g / math.log2(rank + 1) for rank, g in enumerate(gains, 1))


def ndcg_at_k(ranking: Sequence[int], evidence: Sequence[EvidenceChunks], k: int) -> float:
    """NDCG@k = DCG выдачи / DCG идеальной выдачи; выигрыш чанка — 2^оценка − 1."""
    grades = chunk_grades(evidence)
    gains = [2 ** grades.get(i, 0) - 1 for i in ranking[:k]]
    ideal = dcg(sorted((2 ** g - 1 for g in grades.values()), reverse=True)[:k])
    return dcg(gains) / ideal if ideal else 0.0


def average_precision(ranking: Sequence[int], evidence: Sequence[EvidenceChunks], k: int) -> float:
    """AP@k = Σ P@i по местам i ≤ k с релевантным чанком / min(|релевантные|, k)."""
    relevant = relevant_chunk_ids(evidence)
    hits = [rank for rank, i in enumerate(ranking[:k], 1) if i in relevant]
    precisions = (n / rank for n, rank in enumerate(hits, 1))
    return sum(precisions) / min(len(relevant), k) if relevant else 0.0


def overlap_at_k(ranking: Sequence[int], reference: Sequence[int], k: int) -> float:
    """|топ-k ∩ эталонный топ-k| / k — ANN recall: какую долю точной выдачи нашёл приближённый индекс."""
    return len(set(ranking[:k]) & set(reference[:k])) / k


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
        row = {"qid": item.qid, "qtype": item.qtype, "split": item.split, "MRR": reciprocal_rank(ranking, evidence),
               "MAP": average_precision(ranking, evidence, max(k_values))}
        for k in k_values:
            row |= {f"Hit@{k}": hit_at_k(ranking, evidence, k), f"R@{k}": recall_at_k(ranking, evidence, k),
                    f"P@{k}": precision_at_k(ranking, evidence, k), f"NDCG@{k}": ndcg_at_k(ranking, evidence, k)}
        if chunk_tokens is not None:
            row[f"tokens@{CONTEXT_K}"] = sum(chunk_tokens[i] for i in ranking[:CONTEXT_K])
        rows.append(row)
    return pd.DataFrame(rows)


def negative_per_question(rankings: Mapping[str, Sequence[int]], qrels: Qrels, items: Sequence[GoldItem],
                          k: int = 5) -> pd.DataFrame:
    """Для вопросов без ответа в корпусе — доля топ-k, занятая чанками «по теме, но без ответа» (оценка 1):
    чем она выше, тем правдоподобнее контекст, на котором генератор может выдумать ответ."""
    rows = [{"qid": item.qid, "qtype": item.qtype, "split": item.split,
             f"near@{k}": sum(i in chunk_grades(qrels[item.qid]) for i in rankings[item.qid][:k]) / k}
            for item in items if not has_evidence(qrels[item.qid])]
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


def threshold_curve(rankings: Mapping[str, Ranking], qrels: Qrels, items: Sequence[GoldItem],
                    thresholds: Sequence[float], k: int = CONTEXT_K) -> pd.DataFrame:
    """Компромисс precision/recall порога по score реранкера: в контекст идут чанки топ-k со score ≥ порога.

    - precision — доля релевантных среди всех оставленных чанков (микро-среднее по вопросам с доказательствами);
    - recall — средний R@k по вопросам с доказательствами после отсечения;
    - F1 = 2PR / (P + R);
    - «негативные без контекста» — доля негативных вопросов, у которых не осталось ни одного чанка:
      генератор получит пустой контекст и должен ответить «не знаю».
    """
    answered = [i for i in items if has_evidence(qrels[i.qid])]
    negatives = [i for i in items if not has_evidence(qrels[i.qid])]

    def point(threshold: float) -> dict:
        kept = {i.qid: [doc for doc, s in rankings[i.qid][:k] if s >= threshold] for i in items}
        relevant_kept = sum(len(set(kept[i.qid]) & relevant_chunk_ids(qrels[i.qid])) for i in answered)
        total_kept = sum(len(kept[i.qid]) for i in answered)
        precision = relevant_kept / total_kept if total_kept else 0.0
        recall = sum(recall_at_k(kept[i.qid], qrels[i.qid], k) for i in answered) / len(answered) if answered else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        return {"порог": threshold, "precision": precision, "recall": recall, "F1": f1,
                "чанков в контексте": total_kept / len(answered) if answered else 0.0,
                "негативные без контекста": (sum(not kept[i.qid] for i in negatives) / len(negatives)
                                             if negatives else 0.0)}

    return pd.DataFrame([point(t) for t in thresholds])
