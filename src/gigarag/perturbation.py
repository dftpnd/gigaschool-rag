"""Намеренные искажения ответа для контрольной проверки судьи достоверности.

Человек поставил всем ответам выборки разметки максимальную достоверность: на такой выборке судья, всегда
возвращающий 5, неотличим от хорошего. Контрольная выборка проверяет, ловит ли судья выдумку: к подтверждённому
ответу применяется искажение, после которого в ответе гарантированно есть утверждение, которого нет во фрагментах.

- `swap_value` — одно числовое значение ответа заменено другим, которого нет ни в одном фрагменте
  (ответ противоречит контексту);
- `add_invented_claim` — к ответу добавлено правдоподобное утверждение со ссылкой [1] и числом, которого нет
  во фрагментах (ответ сообщает неподтверждённый факт).
"""

import random
from collections.abc import Sequence
from dataclasses import dataclass

from gigarag.patterns import value_spans
from gigarag.stats import weighted_kappa

# Шаблоны выдуманных утверждений: типичные правки баланса, ссылка на первый фрагмент — как в настоящих ответах
INVENTED_CLAIMS = (
    "Кроме того, в этом же патче перезарядка уменьшена на {n} секунд [1].",
    "Также базовая броня увеличена на {n} [1].",
    "Помимо этого, стоимость снижена на {n} золота [1].",
    "Ещё в этом обновлении радиус действия увеличен на {n} [1].",
)
CLAIM_VALUES = range(13, 98)  # двузначные значения: правдоподобны для правок и редко совпадают со значениями контекста
SWAP_FACTORS = range(2, 50)   # множители при поиске значения вместо исходного: хватает даже для плотного контекста
CAUGHT_MAX = 3                # «поймал» — балл не выше 3: по шкале это «часть утверждений не подтверждается»


def context_values(passages: Sequence[str]) -> set[str]:
    """Все числовые значения фрагментов как строки — чтобы подставляемое значение в них не встречалось."""
    return {text[a:b] for text in passages for a, b in value_spans(text)}


def _absent_value(value: str, taken: set[str]) -> str:
    """Значение того же вида (целое или с запятой), отличное от исходного и отсутствующее в контексте:
    перебираются value × 2, × 3, … — первое подходящее."""
    whole, _, fraction = value.partition(",")
    for factor in SWAP_FACTORS:
        candidate = str(int(whole) * factor + 1) + (f",{fraction}" if fraction else "")
        if candidate not in taken and candidate != value:
            return candidate
    raise ValueError(f"не нашлось значения вместо {value}")


def swap_value(answer: str, passages: Sequence[str], rng: random.Random) -> str | None:
    """Ответ с одним числовым значением, заменённым на отсутствующее во фрагментах; None, если чисел в ответе нет."""
    spans = value_spans(answer)
    if not spans:
        return None
    start, end = rng.choice(spans)
    return answer[:start] + _absent_value(answer[start:end], context_values(passages)) + answer[end:]


def add_invented_claim(answer: str, passages: Sequence[str], rng: random.Random) -> str:
    """Ответ с выдуманным утверждением в конце первого абзаца; число в нём отсутствует во фрагментах."""
    taken = context_values(passages)
    value = rng.choice([n for n in CLAIM_VALUES if str(n) not in taken])
    claim = rng.choice(INVENTED_CLAIMS).format(n=value)
    first, separator, rest = answer.partition("\n\n")
    return f"{first} {claim}{separator}{rest}"


@dataclass(frozen=True)
class Detection:
    rates: dict[str, float]  # вид → доля ответов с баллом ≤ CAUGHT_MAX (для исходных — ложная тревога)
    kappa: float             # Cohen's kappa между меткой по построению и «балл ≤ CAUGHT_MAX»


def detection(scored: Sequence[tuple[str, int | None]], original: str) -> Detection:
    """Насколько судья ловит искажения: пары (вид ответа, балл судьи). Метка по построению — искажён ли ответ
    (вид ≠ `original`); «поймал» — балл ≤ CAUGHT_MAX, ответ не по формату считается непойманным.
    Kappa на двух категориях с квадратичными весами совпадает с обычной Cohen's kappa."""
    caught = [score is not None and score <= CAUGHT_MAX for _, score in scored]
    kinds = list(dict.fromkeys(kind for kind, _ in scored))
    rates = {kind: sum(c for (k, _), c in zip(scored, caught, strict=True) if k == kind) /
             sum(k == kind for k, _ in scored) for kind in kinds}
    truth = [int(kind != original) for kind, _ in scored]
    return Detection(rates, weighted_kappa(truth, [int(c) for c in caught], (0, 1)))
