"""Подписи тематических кластеров через c-TF-IDF."""

import numpy as np
from scipy.sparse import csr_matrix, spmatrix


def cluster_term_counts(doc_term: spmatrix, labels: np.ndarray, n_clusters: int) -> np.ndarray:
    """Матрица «кластер × термин»: суммы частот термина по документам кластера."""
    membership = csr_matrix((np.ones(len(labels)), (labels, np.arange(len(labels)))), shape=(n_clusters, len(labels)))
    return (membership @ doc_term).toarray().astype(float)


def class_tfidf(term_counts: np.ndarray) -> np.ndarray:
    """c-TF-IDF (как в BERTopic): tf в кластере × log(1 + A / частота термина в корпусе).

    A — среднее число слов в кластере. Высокий вес — термин частый в кластере и редкий в остальных.
    """
    words_per_cluster = term_counts.sum(axis=1, keepdims=True)
    tf = term_counts / words_per_cluster
    idf = np.log1p(words_per_cluster.mean() / term_counts.sum(axis=0, keepdims=True))
    return tf * idf


def top_terms(weights: np.ndarray, vocabulary: np.ndarray, k: int = 10) -> list[list[str]]:
    """k терминов с наибольшим весом для каждой строки матрицы."""
    return [list(vocabulary[np.argsort(-row)[:k]]) for row in weights]
