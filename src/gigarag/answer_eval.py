"""Оценка ответов системы по эталону: вердикт судьи на каждый вопрос и сводка по типам вопросов.

Вердикты:
- correct — ответ верен и полон (для негативного вопроса — система признала, что не знает);
- partial — верна только часть ответа;
- incorrect — ответ неверен или выдуман (для негативного вопроса — любой содержательный ответ);
- abstained — система отказалась отвечать на вопрос, у которого ответ есть.

Две метрики ошибок не смешиваются: `hallucinated` — неверный содержательный ответ на вопрос, у которого
ответ в корпусе есть; `missed_abstention` — содержательный ответ на негативный вопрос вместо отказа.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

VERDICTS = ("correct", "partial", "incorrect", "abstained")
NEGATIVE = "negative"


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
        """Неверный содержательный ответ на вопрос, у которого есть эталон, — выдумка модели."""
        return self.verdict == "incorrect" and self.qtype != NEGATIVE

    @property
    def missed_abstention(self) -> bool:
        """Содержательный ответ на негативный вопрос: ответа в корпусе нет, а модель не отказалась."""
        return self.verdict == "incorrect" and self.qtype == NEGATIVE


def abstention_verdict(qtype: str) -> str:
    """Вердикт отказа без судьи: для негативного вопроса отказ верен, для остальных — `abstained`."""
    return "correct" if qtype == NEGATIVE else "abstained"


def load_verdicts(path: Path) -> list[AnswerVerdict]:
    return [AnswerVerdict(**json.loads(line)) for line in path.read_text(encoding="utf-8").splitlines()]


def verdict_table(verdicts: list[AnswerVerdict]) -> pd.DataFrame:
    return pd.DataFrame([{**v.__dict__, "score": v.score, "hallucinated": v.hallucinated,
                          "missed_abstention": v.missed_abstention} for v in verdicts])


def summary_by_type(verdicts: list[AnswerVerdict]) -> pd.DataFrame:
    """Доли вердиктов (в %), средний балл, доли выдумок и пропущенных отказов по типам вопросов и в целом —
    для каждой модели. Доли ошибок — от всех вопросов группы."""
    table = verdict_table(verdicts)

    def summarize(group: pd.DataFrame) -> pd.Series:
        shares = group["verdict"].value_counts(normalize=True).reindex(VERDICTS, fill_value=0) * 100
        return pd.concat([pd.Series({"n": len(group)}), shares,
                          pd.Series({"score_%": group["score"].mean() * 100,
                                     "hallucinated_%": group["hallucinated"].mean() * 100,
                                     "missed_abstention_%": group["missed_abstention"].mean() * 100})])

    per_type = table.groupby(["model", "qtype"]).apply(summarize, include_groups=False)
    overall = table.groupby("model").apply(summarize, include_groups=False)
    overall.index = pd.MultiIndex.from_tuples([(m, "ВСЕГО") for m in overall.index], names=["model", "qtype"])
    return pd.concat([per_type, overall]).sort_index().round(1)


RETRIEVAL_LOSS, GENERATION_LOSS, ANSWERED = "потеря на поиске", "потеря на генерации", "ответ"


def loss_stage(recall_at_5: float, abstained: bool, faithful: bool) -> str:
    """Где система теряет вопрос, у которого ответ в корпусе есть:
    - на поиске — в топ-5 нет ни одного чанка с доказательством (R@5 = 0): генератору не из чего ответить;
    - на генерации — доказательство в топ-5 есть, но модель отказалась или ответ недостоверен;
    - ответ — доказательство найдено, ответ содержательный и достоверный (полнота ответа не проверяется:
      судья релевантности не валидирован)."""
    if recall_at_5 == 0:
        return RETRIEVAL_LOSS
    return GENERATION_LOSS if abstained or not faithful else ANSWERED
