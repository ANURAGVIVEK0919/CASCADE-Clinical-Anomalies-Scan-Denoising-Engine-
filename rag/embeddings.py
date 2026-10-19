"""
CASCADE Semantic Embedding Engine

Replaces legacy hash(token) projection with genuine semantic sentence embeddings:
- Default Model: sentence-transformers/all-MiniLM-L6-v2
- Embedding Dimension: 384
- Similarity Metric: Cosine (L2-normalized unit vectors)
- Namespace Strategy: Explicitly isolated (e.g. clinical-guidelines-v1)
"""

import os
import math
from typing import List, Union
import numpy as np

DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_VECTOR_DIM = 384
DEFAULT_NAMESPACE = "clinical-guidelines-v1"


class SemanticEmbeddingEngine:
    """
    Production-grade text embedding engine using genuine sentence-transformers.
    Enforces strict dimension and normalization contracts.
    """

    def __init__(self, model_name: str = DEFAULT_MODEL_NAME, vector_dim: int = DEFAULT_VECTOR_DIM):
        self.model_name = os.environ.get("CASCADE_EMBEDDING_MODEL", model_name)
        self.vector_dim = int(os.environ.get("CASCADE_EMBEDDING_DIM", vector_dim))
        self._model = None
        self._is_loaded = False

    def _load_model(self):
        if not self._is_loaded:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self.model_name)
                self._is_loaded = True
            except Exception as e:
                raise RuntimeError(
                    f"Failed to load embedding model '{self.model_name}': {e}. "
                    "Ensure sentence-transformers is installed in the active environment."
                )

    def embed_text(self, text: str) -> List[float]:
        """Generates an L2-normalized 384-dimensional embedding vector for a single text string."""
        if not text or not text.strip():
            return [0.0] * self.vector_dim

        self._load_model()
        vec = self._model.encode(text, normalize_embeddings=True)
        if len(vec) != self.vector_dim:
            raise ValueError(
                f"Vector dimension mismatch: expected {self.vector_dim}, got {len(vec)} from model {self.model_name}"
            )
        return [float(x) for x in vec]

    def embed_batch(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        """Batched L2-normalized embedding generation for efficient corpus ingestion."""
        if not texts:
            return []

        self._load_model()
        # Clean empty texts to prevent crashes
        cleaned = [t if (t and t.strip()) else "empty" for t in texts]
        vecs = self._model.encode(cleaned, batch_size=batch_size, normalize_embeddings=True)

        results = []
        for v in vecs:
            if len(v) != self.vector_dim:
                raise ValueError(f"Vector dimension mismatch: expected {self.vector_dim}, got {len(v)}")
            results.append([float(x) for x in v])
        return results

    @staticmethod
    def cosine_similarity(v1: List[float], v2: List[float]) -> float:
        """Computes true cosine similarity between two normalized vectors."""
        if not v1 or not v2 or len(v1) != len(v2):
            return 0.0
        # Dot product for L2-normalized vectors equals cosine similarity
        return float(np.dot(v1, v2))
