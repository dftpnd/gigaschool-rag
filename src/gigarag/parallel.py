"""Параллельное выполнение независимых задач с сохранением порядка результатов."""

from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor


def parallel_map[T, R](fn: Callable[[T], R], items: Iterable[T], workers: int = 8) -> list[R]:
    """`[fn(x) for x in items]`, но в пуле потоков — для I/O-задач вроде запросов к API."""
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(fn, items))
