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


def load_sentence_model(model_name: str, revision: str, *, max_seq_length: int, half: bool = False,
                        device: str | None = None):
    """Модель sentence-transformers; `half` — fp16 на GPU (на CPU игнорируется: там fp16 медленнее)."""
    from sentence_transformers import SentenceTransformer
    device = device or pick_device()
    model = SentenceTransformer(model_name, revision=revision, device=device)
    model.max_seq_length = max_seq_length
    return model.half() if half and device == "cuda" else model


def encode(model, texts: Sequence[str], *, prefix: str = "", batch_size: int = 64) -> np.ndarray:
    """Нормированные эмбеддинги уже загруженной моделью (float32)."""
    return model.encode([prefix + t for t in texts], batch_size=batch_size, normalize_embeddings=True,
                        show_progress_bar=False, convert_to_numpy=True).astype(np.float32)


def memo_embeddings(model, *, prefix: str = "", batch_size: int = 64):
    """Эмбеддинги для чанкеров Chonkie с памятью «текст → вектор».

    Семантический чанкер на каждом пороге заново эмбеддит те же предложения и окна; память делает
    перебор порогов почти бесплатным, а время эмбеддинга границ считается один раз на модель.
    """
    from chonkie import SentenceTransformerEmbeddings

    class MemoEmbeddings(SentenceTransformerEmbeddings):
        def __init__(self) -> None:
            super().__init__(model)
            self.memo: dict[str, np.ndarray] = {}

        def embed(self, text: str) -> np.ndarray:
            return self.embed_batch([text])[0]

        def embed_batch(self, texts: list[str]) -> list[np.ndarray]:
            missing = list(dict.fromkeys(t for t in texts if t not in self.memo))
            if missing:
                self.memo |= zip(missing, encode(model, missing, prefix=prefix, batch_size=batch_size), strict=True)
            return [self.memo[t] for t in texts]

    return MemoEmbeddings()
