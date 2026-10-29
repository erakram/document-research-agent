"""Text embeddings: turn text into vectors where similar meanings are close together.

Uses model2vec (a small static embedding model that runs on CPU with only NumPy),
so it is free, fast, and fits on a free hosting plan. Anything with an
embed(texts) -> array method can be swapped in, e.g. a hosted embeddings API.
"""

import logging
import time
from typing import Protocol

import numpy as np

logger = logging.getLogger(__name__)


class Embedder(Protocol):
    def embed(self, texts: list[str]) -> np.ndarray:
        """Return one L2-normalised float32 vector per text."""
        ...


def normalize(vectors: np.ndarray) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.maximum(norms, 1e-12)


class Model2VecEmbedder:
    def __init__(self, model_name: str):
        from model2vec import StaticModel  # imported here so tests don't need the model

        start = time.perf_counter()
        self.model = StaticModel.from_pretrained(model_name)
        logger.info("embedding model=%s loaded in %.1fs", model_name, time.perf_counter() - start)

    def embed(self, texts: list[str]) -> np.ndarray:
        return normalize(self.model.encode(texts))
