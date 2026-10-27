"""Атомарная запись файлов: прерванная запись не оставляет повреждённого файла на итоговом пути."""

import os
import uuid
from collections.abc import Callable
from pathlib import Path


def temp_path(path: Path) -> Path:
    """Временный файл рядом с итоговым, с тем же расширением (np.save сам дописывает .npy без него).
    Имя уникально: параллельные записи одного пути (два потока с одинаковым запросом к LLM)
    не делят временный файл, и последний `os.replace` просто побеждает."""
    return path.with_name(f".{path.stem}.{uuid.uuid4().hex}.tmp{path.suffix}")


def atomic_save[T](save: Callable[[T, Path], None], value: T, path: Path) -> None:
    """Сохраняет `value` функцией `save` во временный файл и переименовывает его в `path` только после успеха."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = temp_path(path)
    try:
        save(value, tmp)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def atomic_write_text(path: Path, text: str) -> None:
    atomic_save(lambda t, p: p.write_text(t, encoding="utf-8"), text, path)


def atomic_write_bytes(path: Path, payload: bytes) -> None:
    atomic_save(lambda b, p: p.write_bytes(b), payload, path)
