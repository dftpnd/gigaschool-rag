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
