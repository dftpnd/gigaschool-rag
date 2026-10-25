"""Оценка ответов системы по эталону: вердикт судьи на каждый вопрос и сводка по типам вопросов.

Вердикты:
- correct — ответ верен и полон (для негативного вопроса — система признала, что не знает);
- partial — верна только часть ответа;
- incorrect — ответ неверен или выдуман (для негативного вопроса — любой содержательный ответ);
- abstained — система отказалась отвечать на вопрос, у которого ответ есть.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

VERDICTS = ("correct", "partial", "incorrect", "abstained")


@dataclass(frozen=True)
class AnswerVerdict:
    qid: str
    qtype: str
    model: str
    answer: str
    verdict: str
    rationale: str = ""

    @property
    def score(self) -> float:
        """Балл ответа: верный — 1, частично верный — 0,5, иначе 0."""
        return {"correct": 1.0, "partial": 0.5}.get(self.verdict, 0.0)

    @property
    def hallucinated(self) -> bool:
        """Содержательный неверный ответ (а не отказ) — выдумка модели."""
        return self.verdict == "incorrect"


def load_verdicts(path: Path) -> list[AnswerVerdict]:
    return [AnswerVerdict(**json.loads(line)) for line in path.read_text(encoding="utf-8").splitlines()]


def verdict_table(verdicts: list[AnswerVerdict]) -> pd.DataFrame:
    return pd.DataFrame([{**v.__dict__, "score": v.score, "hallucinated": v.hallucinated} for v in verdicts])


def summary_by_type(verdicts: list[AnswerVerdict]) -> pd.DataFrame:
    """Доли вердиктов (в %), средний балл и доля выдумок по типам вопросов и в целом — для каждой модели."""
    table = verdict_table(verdicts)

    def summarize(group: pd.DataFrame) -> pd.Series:
        shares = group["verdict"].value_counts(normalize=True).reindex(VERDICTS, fill_value=0) * 100
        return pd.concat([pd.Series({"n": len(group)}), shares,
                          pd.Series({"score_%": group["score"].mean() * 100,
                                     "hallucinated_%": group["hallucinated"].mean() * 100})])

    per_type = table.groupby(["model", "qtype"]).apply(summarize, include_groups=False)
    overall = table.groupby("model").apply(summarize, include_groups=False)
    overall.index = pd.MultiIndex.from_tuples([(m, "ВСЕГО") for m in overall.index], names=["model", "qtype"])
    return pd.concat([per_type, overall]).sort_index().round(1)
