from __future__ import annotations

from functools import lru_cache

import numpy as np
from sentence_transformers import SentenceTransformer


@lru_cache(maxsize=4)
def _load_model(model_name: str) -> SentenceTransformer:
    return SentenceTransformer(model_name)


def embed(texts: list[str], model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> list[list[float]]:
    model = _load_model(model_name)
    vectors: np.ndarray = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return vectors.tolist()
