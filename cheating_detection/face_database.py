"""
Persistent storage for face embeddings using simple mean encoding.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np

LOGGER = logging.getLogger(__name__)


class FaceDatabase:
    """
    Manage a dictionary mapping person labels to mean face embeddings.
    """

    def __init__(self, database_path: Path) -> None:
        self.path = Path(database_path)
        self._embeddings: Dict[str, np.ndarray] = {}
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            LOGGER.warning("Face database not found at %s. Starting empty.", self.path)
            self._embeddings = {}
            return
        with self.path.open("rb") as handle:
            raw = pickle.load(handle)
        # Ensure arrays are numpy arrays
        self._embeddings = {
            str(name): np.asarray(embedding, dtype=np.float32)
            for name, embedding in raw.items()
        }
        LOGGER.info("Loaded face database with %d identities", len(self._embeddings))

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("wb") as handle:
            pickle.dump(self._embeddings, handle)
        LOGGER.info("Persisted face database with %d identities", len(self._embeddings))

    @property
    def people(self) -> Tuple[str, ...]:
        return tuple(sorted(self._embeddings.keys()))

    def has_person(self, name: str) -> bool:
        return name in self._embeddings

    def add_person(self, name: str, embeddings: Iterable[np.ndarray]) -> int:
        """
        Add a new person using the mean embedding of the provided vectors.

        Returns:
            Number of embeddings that were consumed.
        """
        vectors = [
            np.asarray(embedding, dtype=np.float32)
            for embedding in embeddings
            if embedding is not None
        ]
        if not vectors:
            raise ValueError("No embeddings supplied for new identity")
        mean_embedding = np.mean(vectors, axis=0)
        self._embeddings[name] = mean_embedding
        self.save()
        return len(vectors)

    def identify(self, embedding: np.ndarray) -> Tuple[str, float]:
        """
        Identify the closest person based on cosine similarity.

        Returns:
            A tuple of (name, score). Score is in [-1, 1]; higher is better.
            If the database is empty, returns ("Unknown", 0.0).
        """
        if not self._embeddings:
            return "Unknown", 0.0
        query = _normalize(embedding)
        best_score = -1.0
        best_name = "Unknown"
        for name, stored in self._embeddings.items():
            score = float(np.dot(query, _normalize(stored)))
            if score > best_score:
                best_score = score
                best_name = name
        return best_name, best_score


def _normalize(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector
    return vector / norm
