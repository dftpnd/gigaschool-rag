"""LLM-судья ответа RAG по трём независимым критериям, каждый — отдельным запросом со шкалой 1–5.

- **faithfulness** — всё ли в ответе подтверждается контекстом;
- **relevance** — отвечает ли ответ на заданный вопрос;
- **utilization** — использует ли ответ весь относящийся к вопросу контекст.

Судья видит вопрос, контекст и ответ, но не эталон gold: критерии — про связь ответа с контекстом и вопросом,
а правильность по эталону и полнота поиска меряются отдельно. Отказ («Не знаю…») оценивается по тому,
есть ли ответ в контексте: честный отказ при пустом контексте — не ошибка генерации.

Промпты прошли итерации (`PROMPTS`): версия задаёт тексты всех трёх критериев, разбор ответа общий.
"""

from collections.abc import Sequence
from dataclasses import dataclass

from gigarag.enrich import MODEL, SEED, TEMPERATURE
from gigarag.generation import context_block
from gigarag.llm import ChatRequest, chat_request, json_object_or_empty

CRITERIA = ("faithfulness", "relevance", "utilization")
SCORES = (1, 2, 3, 4, 5)
THINKING = False  # оценка — одно число и короткое обоснование

_ROLE = """Ты — строгий эксперт, проверяющий ответы RAG-системы о балансе Dota 2. Тебе дадут вопрос игрока,
пронумерованные фрагменты контекста, которые получила система, и её ответ. Оцени ответ по одному критерию."""

_FORMAT = """Сначала коротко обоснуй оценку (1–3 предложения), потом поставь балл.
Ответ — строго JSON: {"rationale": "...", "score": 1|2|3|4|5}"""

@dataclass(frozen=True)
class Scale:
    """Шкала критерия: определение, описание каждого балла и правило для отказа. Одна и та же — в промпте
    судьи любой версии и на странице ручной разметки: человек и судья оценивают по одной шкале."""
    title: str
    definition: str
    anchors: tuple[str, str, str, str, str]  # описания баллов 5, 4, 3, 2, 1
    abstention: str

    def text(self) -> str:
        levels = "\n".join(f"{score} — {anchor};" for score, anchor in zip((5, 4, 3, 2, 1), self.anchors, strict=True))
        return f"Критерий — {self.title}: {self.definition}\n{levels}\n{self.abstention}"


SCALES = {
    "faithfulness": Scale(
        "FAITHFULNESS", "содержит ли ответ только информацию из фрагментов, без выдумок.",
        ("каждое утверждение ответа подтверждается фрагментами",
         "всё подтверждается, кроме незначительной детали или неточной формулировки",
         "часть утверждений не подтверждается фрагментами",
         "большинство утверждений не подтверждается фрагментами",
         "ответ выдуман или противоречит фрагментам"),
        "Отказ («Не знаю…») ничего не утверждает — это 5."),
    "relevance": Scale(
        "ANSWER RELEVANCE", "отвечает ли ответ на поставленный вопрос.",
        ("полностью и прямо отвечает на вопрос",
         "отвечает, но с лишним или с небольшим пробелом",
         "отвечает на часть вопроса",
         "по теме, но не отвечает на вопрос",
         "не отвечает на вопрос"),
        "Отказ («Не знаю…»): если во фрагментах действительно нет ответа — 5; если ответ во фрагментах есть — 1."),
    "utilization": Scale(
        "CONTEXT UTILIZATION", "использует ли ответ всю относящуюся к вопросу информацию из фрагментов.",
        ("использованы все фрагменты и факты, относящиеся к вопросу",
         "упущена мелкая деталь",
         "использована примерно половина относящейся к вопросу информации",
         "использована малая часть",
         "относящаяся к вопросу информация не использована"),
        "Фрагменты не о вопросе не учитываются. Если относящейся к вопросу информации во фрагментах нет — 5."),
}

V1 = {criterion: f"{_ROLE}\n\n{scale.text()}\n{_FORMAT}" for criterion, scale in SCALES.items()}

PROMPTS = {1: V1}


@dataclass(frozen=True)
class Verdict:
    score: int | None  # None — ответ судьи не по формату
    rationale: str


def judge_user(question: str, passages: Sequence[str], answer: str) -> str:
    return f"Вопрос: {question}\n\nФрагменты:\n\n{context_block(passages)}\n\nОтвет системы:\n{answer}"


def judge_request(criterion: str, question: str, passages: Sequence[str], answer: str,
                  version: int) -> ChatRequest:
    return chat_request(MODEL, PROMPTS[version][criterion], judge_user(question, passages, answer), TEMPERATURE,
                        SEED, json_mode=True, thinking=THINKING)


def parse_verdict(content: str) -> Verdict:
    """Балл 1–5 и обоснование; нераспознанный балл — None (такой ответ не участвует в средних)."""
    verdict = json_object_or_empty(content)
    score = verdict.get("score")
    rationale = verdict.get("rationale")
    return Verdict(score if score in SCORES and not isinstance(score, bool) else None,
                   rationale if isinstance(rationale, str) else "")
