"""Замер ресурсов вычислений, которые живут в процессе Python (индекс BM25, массив векторов)."""

import tracemalloc
from collections.abc import Callable

MIB = 2 ** 20


def traced_peak_mb[T](compute: Callable[[], T]) -> tuple[T, float]:
    """Значение и пик памяти, выделенной Python во время расчёта, МБ (tracemalloc: только аллокации Python
    и numpy, без памяти библиотек на C, которые выделяют её в обход аллокатора Python)."""
    tracemalloc.start()
    try:
        value = compute()
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return value, peak / MIB
