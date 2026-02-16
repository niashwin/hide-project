"""
hide/core/hide_space.py — Core HIDE Space Implementation
=========================================================
The HIDE space is a numpy/torch array that stores contextual embeddings
with metadata. This is the memory substrate for the entire project.
"""

import numpy as np
import json
import time
from typing import List, Tuple, Dict, Optional, Callable


class HIDESpace:
    """
    High-Dimensional Embedding space as memory substrate.

    Stores embeddings with metadata, supports cosine retrieval,
    temporal decay, and capacity tracking.
    """

    def __init__(self, dim: int = 384, max_memories: int = 10000):
        self.dim = dim
        self.max_memories = max_memories
        self.embeddings = np.zeros((max_memories, dim), dtype=np.float32)
        self.metadata: List[Dict] = []
        self.timestamps: List[float] = []
        self.count = 0

    def store(self, embedding: np.ndarray, metadata: Dict) -> int:
        """Store embedding with metadata. Returns memory ID."""
        if self.count >= self.max_memories:
            # Expand if needed
            new_max = self.max_memories * 2
            new_embs = np.zeros((new_max, self.dim), dtype=np.float32)
            new_embs[: self.count] = self.embeddings[: self.count]
            self.embeddings = new_embs
            self.max_memories = new_max

        memory_id = self.count
        self.embeddings[memory_id] = embedding / (
            np.linalg.norm(embedding) + 1e-8
        )  # L2 normalize
        self.metadata.append(metadata)
        self.timestamps.append(time.time())
        self.count += 1
        return memory_id

    def retrieve(
        self,
        query: np.ndarray,
        k: int = 5,
        filter_fn: Optional[Callable] = None,
        decay_fn: Optional[Callable] = None,
        query_time: Optional[float] = None,
    ) -> List[Tuple[int, float, Dict]]:
        """
        Retrieve top-k memories by cosine similarity.
        Optional temporal decay and metadata filtering.
        Returns: List of (memory_id, score, metadata)
        """
        if self.count == 0:
            return []

        query_norm = query / (np.linalg.norm(query) + 1e-8)
        similarities = self.embeddings[: self.count] @ query_norm

        # Apply temporal decay if provided
        if decay_fn is not None:
            for i in range(self.count):
                # decay_fn can accept metadata dict or time_delta float
                try:
                    weight = decay_fn(self.metadata[i])
                except (TypeError, KeyError):
                    if query_time is not None:
                        time_delta = abs(query_time - self.timestamps[i])
                        weight = decay_fn(time_delta)
                    else:
                        weight = 1.0
                similarities[i] *= weight

        # Apply filter
        if filter_fn is not None:
            for i in range(self.count):
                if not filter_fn(self.metadata[i]):
                    similarities[i] = -float("inf")

        # Get top-k
        k = min(k, self.count)
        top_indices = np.argpartition(similarities, -k)[-k:]
        top_indices = top_indices[np.argsort(similarities[top_indices])[::-1]]

        return [
            (int(idx), float(similarities[idx]), self.metadata[idx])
            for idx in top_indices
            if similarities[idx] > -float("inf")
        ]

    def clear(self):
        """Reset memory."""
        self.embeddings[:] = 0
        self.metadata.clear()
        self.timestamps.clear()
        self.count = 0

    def get_all_embeddings(self) -> np.ndarray:
        """Return all stored embeddings."""
        return self.embeddings[: self.count].copy()

    def capacity(self) -> int:
        return self.count

    def replace(self, memory_id: int, new_embedding: np.ndarray):
        """Replace an embedding in-place (used by consolidation/replay)."""
        self.embeddings[memory_id] = new_embedding / (
            np.linalg.norm(new_embedding) + 1e-8
        )

    def remove_indices(self, indices: List[int]):
        """Remove memories by index. Compacts the array."""
        keep_mask = np.ones(self.count, dtype=bool)
        keep_mask[indices] = False
        kept_indices = np.where(keep_mask)[0]

        new_count = len(kept_indices)
        new_embeddings = self.embeddings[kept_indices].copy()
        new_metadata = [self.metadata[i] for i in kept_indices]
        new_timestamps = [self.timestamps[i] for i in kept_indices]

        self.embeddings[:new_count] = new_embeddings
        self.embeddings[new_count:] = 0
        self.metadata = new_metadata
        self.timestamps = new_timestamps
        self.count = new_count

    def size(self) -> int:
        return self.count

    def save(self, path: str):
        """Save HIDE space to disk."""
        np.save(f"{path}_embeddings.npy", self.embeddings[: self.count])
        with open(f"{path}_metadata.json", "w") as f:
            json.dump(
                {"metadata": self.metadata, "timestamps": self.timestamps},
                f,
            )

    def load(self, path: str):
        """Load HIDE space from disk."""
        embs = np.load(f"{path}_embeddings.npy")
        with open(f"{path}_metadata.json") as f:
            data = json.load(f)
        self.count = len(embs)
        self.embeddings[: self.count] = embs
        self.metadata = data["metadata"]
        self.timestamps = data["timestamps"]
