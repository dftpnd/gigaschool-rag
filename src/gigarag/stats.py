"""Описательная статистика по группам."""

from collections.abc import Sequence

import pandas as pd

_DESCRIBE_COLS = ["count", "mean", "std", "min", "10%", "50%", "90%", "max"]
_PERCENTILES = [0.1, 0.5, 0.9]


def percent_where(values: pd.Series, condition) -> float:
    """Процент элементов, для которых выполняется `condition` (функция Series → bool Series)."""
    return float(condition(values).mean() * 100)


def length_summary(df: pd.DataFrame, column: str, by: str, short: int, long: int) -> pd.DataFrame:
    """Распределение длин по группам и в целом: перцентили и доли коротких (< short) и длинных (> long)."""
    def summarize(values: pd.Series) -> pd.Series:
        stats = values.describe(percentiles=_PERCENTILES)[_DESCRIBE_COLS]
        stats[f"<{short}_%"] = percent_where(values, lambda v: v < short)
        stats[f">{long}_%"] = percent_where(values, lambda v: v > long)
        return stats

    groups = {name: summarize(values) for name, values in df.groupby(by)[column]}
    groups["ВСЕГО"] = summarize(df[column])
    return pd.DataFrame(groups).T.rename_axis(by).round(1)


def share_percent(counts: pd.Series) -> pd.Series:
    return counts / counts.sum() * 100


def agreement_rate(first: Sequence, second: Sequence) -> float:
    """Доля совпавших ответов двух прогонов: Σ [a_i = b_i] / n — проверка детерминизма LLM при t = 0."""
    return sum(a == b for a, b in zip(first, second, strict=True)) / len(first) if first else 1.0


def pareto_front(costs: Sequence[float], values: Sequence[float]) -> list[bool]:
    """Отметки точек парето-фронта «меньше стоимость — больше качество»: точка i на фронте, если нет j
    с costs_j ≤ costs_i и values_j ≥ values_i, где хотя бы одно неравенство строгое."""
    points = list(zip(costs, values, strict=True))
    return [not any(c <= cost and v >= value and (c, v) != (cost, value) for c, v in points)
            for cost, value in points]


def weighted_kappa(first: Sequence[int], second: Sequence[int], categories: Sequence[int]) -> float:
    """Weighted Cohen's kappa с квадратичными весами — согласие двух оценщиков на порядковой шкале:

    κ_w = 1 − Σ w_ij · O_ij / Σ w_ij · E_ij,  w_ij = (i − j)² / (k − 1)²,

    O — доли пар (оценка первого = i, второго = j), E — то же при независимости оценщиков (произведение
    маргиналов), k — число категорий. Расхождение 1 против 5 штрафуется в 16 раз сильнее, чем 4 против 5.
    Если оба ставят одну и ту же категорию всем, ожидаемое расхождение нулевое и κ не определена — NaN.
    """
    index = {c: i for i, c in enumerate(categories)}
    k, n = len(categories), len(first)
    observed = [[0.0] * k for _ in range(k)]
    for a, b in zip(first, second, strict=True):
        observed[index[a]][index[b]] += 1 / n
    rows = [sum(row) for row in observed]
    columns = [sum(observed[i][j] for i in range(k)) for j in range(k)]
    weight = [[(i - j) ** 2 / (k - 1) ** 2 for j in range(k)] for i in range(k)]
    disagreement = sum(weight[i][j] * observed[i][j] for i in range(k) for j in range(k))
    expected = sum(weight[i][j] * rows[i] * columns[j] for i in range(k) for j in range(k))
    return 1 - disagreement / expected if expected else float("nan")
