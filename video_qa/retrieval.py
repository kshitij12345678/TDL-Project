"""
retrieval.py
FAISS-based vector store for temporal clip retrieval.
Uses Inner Product (dot product) on L2-normalized vectors → cosine similarity.
"""

import logging
import os
import pickle

import faiss
import numpy as np

logger = logging.getLogger(__name__)


class VideoRetriever:
    """
    Manages a FAISS index over clip-level embeddings.

    Usage:
        retriever = VideoRetriever(dim=512)
        retriever.build_index(clip_embeddings, clip_metadata)
        results = retriever.search(query_embedding, k=5)
    """

    def __init__(self, dim: int = 512):
        self.dim = dim
        # Inner-product index — works as cosine similarity when vectors are L2-normalized
        self.index: faiss.IndexFlatIP = faiss.IndexFlatIP(dim)
        self.clip_metadata: list[dict] = []

    # ── Build ─────────────────────────────────────────────────────────────────

    def build_index(self, clip_embeddings: np.ndarray, clip_metadata: list[dict]):
        """
        Populate FAISS index.

        Args:
            clip_embeddings: (num_clips, dim) float32, L2-normalized.
            clip_metadata:   Parallel list of clip dicts (from temporal.py).
        """
        assert clip_embeddings.shape[0] == len(clip_metadata), (
            "Mismatch between embeddings and metadata"
        )
        self.index.reset()
        self.index.add(clip_embeddings)
        self.clip_metadata = clip_metadata
        logger.info(f"FAISS index built with {len(clip_metadata)} clips.")

    # ── Search ────────────────────────────────────────────────────────────────

    def search(self, query_embedding: np.ndarray, k: int = 5) -> list[dict]:
        """
        Retrieve top-k most similar clips.

        Args:
            query_embedding: (1, dim) float32, L2-normalized.
            k:               Number of results to return.

        Returns:
            List of clip dicts, each augmented with 'score' (cosine similarity, higher = better).
        """
        k = min(k, self.index.ntotal)
        if k == 0:
            return []

        distances, indices = self.index.search(query_embedding, k)

        results = []
        for score, idx in zip(distances[0], indices[0]):
            if idx < 0:
                continue
            clip = dict(self.clip_metadata[idx])
            clip["score"] = float(score)
            results.append(clip)

        # Sort descending by score (already ordered by FAISS, but be explicit)
        results.sort(key=lambda c: c["score"], reverse=True)
        return results

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, folder: str):
        """Save FAISS index + metadata to disk."""
        os.makedirs(folder, exist_ok=True)
        faiss.write_index(self.index, os.path.join(folder, "index.faiss"))
        with open(os.path.join(folder, "metadata.pkl"), "wb") as f:
            pickle.dump({
                "clip_metadata": self.clip_metadata,
                "dim": self.dim,
            }, f)
        logger.info(f"Retriever saved to '{folder}'")

    def load(self, folder: str):
        """Load FAISS index + metadata from disk."""
        index_path = os.path.join(folder, "index.faiss")
        meta_path = os.path.join(folder, "metadata.pkl")
        if not os.path.exists(index_path) or not os.path.exists(meta_path):
            raise FileNotFoundError(f"No saved index found at '{folder}'")
        self.index = faiss.read_index(index_path)
        with open(meta_path, "rb") as f:
            data = pickle.load(f)
        self.dim = data["dim"]
        self.clip_metadata = data["clip_metadata"]
        logger.info(f"Retriever loaded from '{folder}' ({len(self.clip_metadata)} clips)")
