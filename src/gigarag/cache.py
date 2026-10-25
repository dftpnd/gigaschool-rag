"""Кэш дорогих вычислений, привязанный к отпечатку данных, параметрам и коду расчёта."""

import hashlib
import inspect
import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd

from gigarag.atomic import atomic_save


def cache_key(fingerprint: str, params: dict) -> str:
    """Короткий ключ от отпечатка данных и параметров: меняется что угодно — меняется ключ."""
    payload = json.dumps({"data": fingerprint, **params}, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def code_fingerprint(*modules: ModuleType) -> str:
    """Отпечаток исходного кода модулей: правка правил (регулярок, формул) меняет ключ кэша сама."""
    source = "".join(inspect.getsource(m) for m in modules)
    return hashlib.sha256(source.encode()).hexdigest()[:16]


@dataclass(frozen=True)
class CacheStore:
    """Каталог кэша для одного набора данных.

    Пересборка данных или смена модели/параметров/кода дают новый ключ — старый файл
    просто не находится, и результат пересчитывается.
    """
    root: Path
    fingerprint: str

    def path(self, name: str, ext: str, **params) -> Path:
        self.root.mkdir(parents=True, exist_ok=True)
        return self.root / f"{name}-{cache_key(self.fingerprint, params)}.{ext}"


def load_or_compute[T](path: Path, compute: Callable[[], T], *,
                    load: Callable[[Path], T], save: Callable[[T, Path], None]) -> T:
    """Значение из кэша, а если его нет — вычислить и сохранить атомарно
    (прерванная запись не оставит повреждённого файла, который потом загрузится как кэш)."""
    if path.exists():
        return load(path)
    value = compute()
    atomic_save(save, value, path)
    return value


# Пары load/save для load_or_compute.

def read_parquet(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)


def write_parquet(df: pd.DataFrame, path: Path) -> None:
    df.to_parquet(path)


def read_npy(path: Path) -> np.ndarray:
    return np.load(path)


def write_npy(array: np.ndarray, path: Path) -> None:
    np.save(path, array)


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(value, path: Path) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
