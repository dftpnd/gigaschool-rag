"""Воспроизводимая кластеризация эмбеддингов."""

import numpy as np
from sklearn.cluster import KMeans
from threadpoolctl import threadpool_limits


def deterministic_kmeans(vectors: np.ndarray, n_clusters: int, seed: int, n_init: int = 10) -> np.ndarray:
    """Метки KMeans, одинаковые от запуска к запуску.

    В многопоточном режиме sklearn KMeans суммирует частичные результаты в недетерминированном
    порядке, и пограничные точки переходят между кластерами даже при фиксированном seed;
    поэтому расчёт ограничен одним потоком.
    """
    with threadpool_limits(limits=1):
        return KMeans(n_clusters=n_clusters, n_init=n_init, random_state=seed).fit_predict(vectors)
