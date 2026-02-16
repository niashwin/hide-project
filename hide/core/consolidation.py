"""
Memory consolidation via geometric merging.

Implements HDBSCAN-based clustering and centroid merging for memory compression.
Note: The paper demonstrates this as an informative negative result — naive vector
averaging destroys discriminative structure needed for accurate retrieval.
"""

import numpy as np
from typing import List, Dict, Tuple


def consolidate_memories(embeddings: np.ndarray, metadata: List[Dict],
                         min_cluster_size: int = 10,
                         merge_threshold: float = 0.15) -> Tuple[np.ndarray, List[Dict]]:
    """Consolidate memories via HDBSCAN clustering and centroid merging.

    WARNING: This produces an informative negative result. Geometric merging
    achieves compression but increases backward interference. See paper Discussion.
    """
    try:
        import hdbscan
    except ImportError:
        raise ImportError("hdbscan required for consolidation: pip install hdbscan")

    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, metric='cosine')
    labels = clusterer.fit_predict(embeddings)

    new_embeddings = []
    new_metadata = []

    for label in set(labels):
        if label == -1:
            # Noise points: keep as-is
            mask = labels == label
            for i in np.where(mask)[0]:
                new_embeddings.append(embeddings[i])
                new_metadata.append(metadata[i])
        else:
            mask = labels == label
            cluster_embs = embeddings[mask]
            centroid = cluster_embs.mean(axis=0)
            centroid = centroid / np.linalg.norm(centroid)
            new_embeddings.append(centroid)
            new_metadata.append({"merged_from": int(mask.sum()), "cluster": int(label)})

    return np.stack(new_embeddings), new_metadata
