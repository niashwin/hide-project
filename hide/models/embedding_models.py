"""
Embedding model loading and encoding.

Supports: MiniLM-L6-v2 (384d), BGE-base (768d), BGE-large (1024d).
All models are open-weight with permissive licenses.
"""

import torch
import numpy as np
from typing import List, Optional


class EmbeddingManager:
    """Manages loading and encoding with sentence embedding models."""

    MODELS = {
        "minilm": ("sentence-transformers/all-MiniLM-L6-v2", 384),
        "bge-base": ("BAAI/bge-base-en-v1.5", 768),
        "bge-large": ("BAAI/bge-large-en-v1.5", 1024),
    }

    def __init__(self, model_name: str = "minilm", device: str = "cuda:1"):
        if model_name not in self.MODELS:
            raise ValueError(f"Unknown model: {model_name}. Choose from {list(self.MODELS.keys())}")
        self.model_id, self._dim = self.MODELS[model_name]
        self.device = device
        self.model = None
        self.tokenizer = None

    @property
    def dim(self) -> int:
        return self._dim

    def text_dim(self) -> int:
        return self._dim

    def load(self):
        """Load the model and tokenizer."""
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(self.model_id, device=self.device)

    def encode(self, texts: List[str], batch_size: int = 256,
               show_progress: bool = False) -> np.ndarray:
        """Encode texts to embeddings."""
        if self.model is None:
            self.load()
        embeddings = self.model.encode(
            texts, batch_size=batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=True,
        )
        return np.array(embeddings)
