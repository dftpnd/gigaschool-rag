"""Описание золотого набора: состав, баланс разбиения, лексическое разнообразие и покрытие корпуса.

Функции принимают элементы набора и справочники корпуса (doc_id → источник, doc_id → патч)
и возвращают таблицы pandas; ноутбук только показывает их.
"""

from collections.abc import Iterable, Mapping

import pandas as pd

from gigarag.gold import QUESTION_TYPES, RELATED, RELEVANT, GoldItem
from gigarag.patterns import parse_patch_version, word_tokens

NO_STYLE = "без изменений"


def question_frame(items: Iterable[GoldItem]) -> pd.DataFrame:
    """Строка на вопрос: тип, часть разбиения, стиль формулировки, длина в словах, число документов
    с оценкой 2 (содержат ответ) и только 1 (по теме)."""
    return pd.DataFrame([{
        "qid": i.qid, "qtype": i.qtype, "split": i.split,
        "style": "+".join(i.styles) or NO_STYLE,
        "words": len(word_tokens(i.question)),
        "relevant_docs": len(i.relevant_docs()),
        "related_docs": len(i.relevant_docs(RELATED)) - len(i.relevant_docs()),
    } for i in items])


def evidence_frame(items: Iterable[GoldItem], sources: Mapping[str, str],
                   patches: Mapping[str, str | None]) -> pd.DataFrame:
    """Строка на размеченный отрывок: вопрос, оценка, документ, источник документа и его патч."""
    return pd.DataFrame([{
        "qid": i.qid, "qtype": i.qtype, "split": i.split, "grade": e.grade,
        "doc_id": e.doc_id, "source": sources[e.doc_id], "patch": patches[e.doc_id],
    } for i in items for e in i.evidence],
        columns=["qid", "qtype", "split", "grade", "doc_id", "source", "patch"])


def split_balance(questions: pd.DataFrame) -> pd.DataFrame:
    """Число вопросов по типам в validation и test и доля test (в %) внутри типа."""
    table = pd.crosstab(questions["qtype"], questions["split"]).reindex(QUESTION_TYPES)
    table.loc["ВСЕГО"] = table.sum()
    table["test_%"] = (table["test"] / table.sum(axis=1) * 100).round(1)
    return table


def source_coverage(evidence: pd.DataFrame, questions: pd.DataFrame) -> pd.DataFrame:
    """Доля вопросов (в %) каждого типа, у которых среди документов с ответом (оценка 2) есть
    документ данного источника. Сумма по строке больше 100%, если ответ собран из нескольких источников."""
    relevant = evidence[evidence["grade"] >= RELEVANT].drop_duplicates(["qid", "source"])
    hits = pd.crosstab(relevant["qtype"], relevant["source"])
    totals = questions["qtype"].value_counts()
    return (hits.div(totals, axis=0) * 100).reindex(QUESTION_TYPES).fillna(0).round(1)


def sources_per_question(evidence: pd.DataFrame, questions: pd.DataFrame) -> pd.Series:
    """qid → число разных источников среди документов с ответом (0 у негативных вопросов)."""
    relevant = evidence[evidence["grade"] >= RELEVANT]
    return relevant.groupby("qid")["source"].nunique().reindex(questions["qid"], fill_value=0)


def patch_coverage(evidence: pd.DataFrame, all_patches: Iterable[str]) -> pd.DataFrame:
    """Патч (в хронологическом порядке) → число вопросов, ответ на которые опирается на документ этого патча;
    патчи корпуса без единого вопроса тоже в таблице (с нулём) — видны пробелы покрытия."""
    relevant = evidence[(evidence["grade"] >= RELEVANT) & evidence["patch"].notna()]
    counts = relevant.drop_duplicates(["qid", "patch"])["patch"].value_counts()
    chronological = sorted(set(all_patches), key=parse_patch_version)
    return counts.reindex(chronological, fill_value=0).rename("questions").to_frame()


def review_frame(decisions: Mapping[str, dict], records: Mapping[str, dict]) -> pd.DataFrame:
    """Строка на кандидата: тип, вид затравки, решение ревью (accept / reserve / reject), причина, вопрос."""
    return pd.DataFrame([{
        "cid": cid, "qtype": records[cid]["qtype"], "seed_kind": records[cid]["meta"].get("kind", ""),
        "decision": d["decision"], "reason": d["reason"], "question": records[cid]["question"],
    } for cid, d in sorted(decisions.items())])
