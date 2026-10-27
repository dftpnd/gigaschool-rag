"""Векторная БД Qdrant в Docker: конфигурации индекса, индексирование, поиск и замер памяти.

Каждая конфигурация индексируется в свежем контейнере: пик памяти контейнера (`memory.peak` cgroup v2)
тогда относится к одному индексированию, а не к сумме всех предыдущих. Пик включает страничный кэш файлов,
которые Qdrant пишет на диск, и кучу, оставшуюся после разбора загрузки. Поэтому отдельно меряется
RAM рабочего индекса: контейнер перезапускается, Qdrant заново поднимает коллекцию с диска, и берётся
собственная память процесса плюс отображённые в память файлы (векторы и граф Qdrant держит через mmap).
"""

import socket
import subprocess
import time
import uuid
from collections.abc import Sequence
from dataclasses import dataclass

import httpx
import numpy as np

from gigarag.retrieval import Ranking

QDRANT_IMAGE = "qdrant/qdrant:v1.19.2"
QDRANT_PORT = 6333
READY_TIMEOUT_S = 60
UPSERT_BATCH = 1024
MIB = 2 ** 20


@dataclass(frozen=True)
class IndexConfig:
    """Конфигурация индекса. `hnsw_m = 0` — без графа HNSW: точный перебор всех векторов (flat)."""
    name: str
    hnsw_m: int = 16
    ef_construct: int = 100
    hnsw_ef: int = 64             # ширина поиска по графу: больше — точнее и медленнее
    int8: bool = False            # скалярная квантизация векторов в int8, поиск с пересчётом по исходным
    on_disk: bool = False         # исходные векторы на диске (mmap), в RAM — только граф и int8-копии

    @property
    def exact(self) -> bool:
        return self.hnsw_m == 0

    def params(self) -> dict:
        return {"hnsw_m": self.hnsw_m, "ef_construct": self.ef_construct, "hnsw_ef": self.hnsw_ef, "int8": self.int8,
                "on_disk": self.on_disk}


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class QdrantContainer:
    """Свежий контейнер Qdrant на время блока `with`; удаляется при выходе."""

    def __init__(self, image: str = QDRANT_IMAGE) -> None:
        self.image = image
        self.name = f"gigarag-qdrant-{uuid.uuid4().hex[:8]}"
        self.port = free_port()

    def __enter__(self) -> "QdrantContainer":
        subprocess.run(["docker", "run", "-d", "--rm", "--name", self.name,
                        "-p", f"127.0.0.1:{self.port}:{QDRANT_PORT}", self.image],
                       check=True, capture_output=True)
        self._wait_ready()
        return self

    def __exit__(self, *_) -> None:
        subprocess.run(["docker", "stop", self.name], check=False, capture_output=True)

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def _wait_ready(self) -> None:
        deadline = time.monotonic() + READY_TIMEOUT_S
        while time.monotonic() < deadline:
            try:
                if httpx.get(f"{self.url}/readyz", timeout=1).status_code == 200:
                    return
            except httpx.TransportError:
                pass
            time.sleep(0.2)
        raise TimeoutError(f"Qdrant {self.name} не поднялся за {READY_TIMEOUT_S} с")

    def _cgroup_file(self, name: str) -> str:
        return subprocess.run(["docker", "exec", self.name, "cat", f"/sys/fs/cgroup/{name}"],
                              check=True, capture_output=True, text=True).stdout

    def peak_memory_mb(self) -> float:
        """Пиковая память контейнера с момента запуска, МБ (cgroup v2 `memory.peak`, со страничным кэшем)."""
        return int(self._cgroup_file("memory.peak")) / MIB

    def resident_memory_mb(self) -> float:
        """Текущая RAM процессов контейнера, МБ: собственная память (`anon`) + отображённые файлы
        (`file_mapped`) из `memory.stat`; кэш файлов, которые никто не отобразил, не считается."""
        stats = dict(line.split() for line in self._cgroup_file("memory.stat").splitlines())
        return (int(stats["anon"]) + int(stats["file_mapped"])) / MIB

    def restart(self) -> None:
        """Перезапуск процесса с теми же данными: коллекции поднимаются с диска, куча загрузки уходит."""
        subprocess.run(["docker", "restart", self.name], check=True, capture_output=True)
        self._wait_ready()

    def client(self):
        from qdrant_client import QdrantClient
        return QdrantClient(url=self.url, timeout=600)


def build_collection(client, name: str, vectors: np.ndarray, config: IndexConfig) -> float:
    """Создаёт коллекцию, загружает векторы (id = индекс чанка) и ждёт окончания построения индекса.
    Возвращает время от создания до готовности индекса, с."""
    from qdrant_client import models
    start = time.perf_counter()
    client.create_collection(
        name,
        vectors_config=models.VectorParams(size=vectors.shape[1], distance=models.Distance.COSINE,
                                           on_disk=config.on_disk),
        hnsw_config=models.HnswConfigDiff(m=config.hnsw_m, ef_construct=config.ef_construct),
        # Порог 1 КБ: граф строится на любом объёме, иначе маленькие сегменты Qdrant оставляет без индекса.
        optimizers_config=models.OptimizersConfigDiff(indexing_threshold=1),
        quantization_config=models.ScalarQuantization(
            scalar=models.ScalarQuantizationConfig(type=models.ScalarType.INT8, always_ram=True))
        if config.int8 else None,
    )
    for offset in range(0, len(vectors), UPSERT_BATCH):
        batch = vectors[offset:offset + UPSERT_BATCH]
        client.upsert(name, points=models.Batch(ids=list(range(offset, offset + len(batch))),
                                                vectors=batch.tolist()), wait=True)
    _wait_indexed(client, name, len(vectors), config)
    return time.perf_counter() - start


def wait_loaded(client, name: str) -> None:
    """Ждёт, пока коллекция после перезапуска поднимется с диска и станет зелёной."""
    from qdrant_client import models
    while True:
        try:
            if client.get_collection(name).status == models.CollectionStatus.GREEN:
                return
        except Exception:  # пока коллекция загружается, API отвечает ошибкой
            pass
        time.sleep(0.2)


def _wait_indexed(client, name: str, n_vectors: int, config: IndexConfig) -> None:
    from qdrant_client import models
    while True:
        info = client.get_collection(name)
        indexed = config.exact or (info.indexed_vectors_count or 0) >= n_vectors
        if info.status == models.CollectionStatus.GREEN and indexed:
            return
        time.sleep(0.1)


def search(client, name: str, query_vectors: Sequence[np.ndarray], config: IndexConfig, limit: int
           ) -> tuple[list[Ranking], np.ndarray]:
    """Выдачи по каждому вектору запроса и время каждого запроса к БД, с (запросы по одному, как в проде)."""
    from qdrant_client import models
    params = models.SearchParams(
        hnsw_ef=config.hnsw_ef, exact=config.exact,
        quantization=models.QuantizationSearchParams(rescore=True) if config.int8 else None)
    rankings, latencies = [], []
    for vector in query_vectors:
        start = time.perf_counter()
        points = client.query_points(name, query=vector.tolist(), limit=limit, search_params=params).points
        latencies.append(time.perf_counter() - start)
        rankings.append([(int(p.id), float(p.score)) for p in points])
    return rankings, np.array(latencies)
