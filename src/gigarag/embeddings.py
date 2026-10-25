"""Эмбеддинги текстов моделями sentence-transformers с закреплённой ревизией."""

from collections.abc import Sequence

import numpy as np


def pick_device() -> str:
    """`cuda`, если доступен GPU, иначе `cpu` (результат тот же, только медленнее)."""
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


def encode_passages(texts: Sequence[str], model_name: str, revision: str, *, max_seq_length: int,
                    prefix: str = "", batch_size: int = 128, device: str | None = None) -> np.ndarray:
    """Нормированные эмбеддинги; `prefix` — служебный префикс модели (у e5 — `"passage: "`)."""
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name, revision=revision, device=device or pick_device())
    model.max_seq_length = max_seq_length
    return model.encode([prefix + t for t in texts], batch_size=batch_size,
                        normalize_embeddings=True, show_progress_bar=False)
