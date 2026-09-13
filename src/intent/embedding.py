"""Embedding-based intent classification.

Provides:

- `Embedder` protocol plus two implementations:
    * `SentenceTransformerEmbedder` — the intended production embedder,
      wrapping a small sentence-transformer model (default
      `sentence-transformers/all-MiniLM-L6-v2`). Requires the
      `sentence-transformers` package and, on first use, network access to
      download model weights. **Not used anywhere in this repo's test
      suite** — tests must run offline and without any model download.
    * `HashingEmbedder` — a deterministic, dependency-light fallback with no
      model download, used for tests and any environment without
      network/model access. It captures crude lexical overlap only, not
      semantics, and must never be silently substituted for the real model
      when reporting production-quality results — anything produced with it
      should say so.

- `EmbeddingCache` — persists embeddings to disk keyed by
  (embedder name, sha256(text)), so a repeated run over the same corpus
  never re-embeds a message it has already embedded.

- `EmbeddingNearestCentroidClassifier` — computes one centroid per training
  intent (the mean embedding of that intent's training examples) and
  classifies new messages by cosine similarity to the nearest centroid.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Optional, Protocol, Sequence, runtime_checkable

import numpy as np

from src.intent.classifier import BaseIntentClassifier, IntentPrediction, IntentTaxonomy

DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


@runtime_checkable
class Embedder(Protocol):
    name: str
    dimension: int

    def embed(self, texts: Sequence[str]) -> np.ndarray: ...


class EmbeddingUnavailableError(RuntimeError):
    """Raised when a real embedding backend can't be constructed (package
    not installed, or the model can't be fetched)."""


class SentenceTransformerEmbedder:
    """Wraps a small sentence-transformer model (default
    sentence-transformers/all-MiniLM-L6-v2). This is the intended
    production embedder for `EmbeddingNearestCentroidClassifier` — it is
    intentionally never imported by the test suite, since it needs the
    `sentence-transformers` package installed and network access to
    download model weights the first time it runs.
    """

    def __init__(self, model_name: str = DEFAULT_MODEL_NAME):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise EmbeddingUnavailableError(
                "sentence-transformers is not installed. Run "
                "`pip install sentence-transformers` and ensure "
                f"'{model_name}' can be downloaded, or use HashingEmbedder "
                "for offline development and tests."
            ) from exc
        try:
            self._model = SentenceTransformer(model_name)
        except Exception as exc:  # network failure, bad model name, etc.
            raise EmbeddingUnavailableError(
                f"Could not load sentence-transformer model '{model_name}': {exc}"
            ) from exc
        self.name = model_name
        self.dimension = self._model.get_sentence_embedding_dimension()

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        return np.asarray(
            self._model.encode(list(texts), show_progress_bar=False, normalize_embeddings=True),
            dtype=np.float32,
        )


class HashingEmbedder:
    """Deterministic signed feature-hashing 'embedding'. No model download
    and no heavyweight dependency — used for tests and any offline
    development. This captures crude lexical overlap only; it is not a
    substitute for real semantic embeddings and results produced with it
    should always be labeled as such.
    """

    def __init__(self, dimension: int = 256, name: str = "hashing-fallback-v1"):
        self.dimension = dimension
        self.name = name
        self._token_re = re.compile(r"[a-z0-9']+")

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        vectors = np.zeros((len(texts), self.dimension), dtype=np.float32)
        for i, text in enumerate(texts):
            tokens = self._token_re.findall(text.lower())
            for token in tokens:
                digest = hashlib.sha256(token.encode("utf-8")).hexdigest()
                h = int(digest, 16)
                idx = h % self.dimension
                sign = 1.0 if (h // self.dimension) % 2 == 0 else -1.0
                vectors[i, idx] += sign
            norm = np.linalg.norm(vectors[i])
            if norm > 0:
                vectors[i] /= norm
        return vectors


def _safe_filename(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", name)


class EmbeddingCache:
    """Disk-backed embedding cache, one `.npz` file per embedder name, keyed
    by `sha256(text)` so an identical message is only ever embedded once
    across runs.
    """

    def __init__(self, cache_dir: str, embedder: Embedder):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.embedder = embedder
        self._path = self.cache_dir / f"{_safe_filename(embedder.name)}.npz"
        self._store: dict[str, np.ndarray] = {}
        self._loaded = False

    @staticmethod
    def _key(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def _load(self) -> None:
        if self._loaded:
            return
        if self._path.exists():
            data = np.load(self._path, allow_pickle=False)
            keys = data["keys"]
            vectors = data["vectors"]
            self._store = {str(k): vectors[i] for i, k in enumerate(keys)}
        self._loaded = True

    def _save(self) -> None:
        keys = list(self._store.keys())
        if keys:
            vectors = np.vstack([self._store[k] for k in keys])
        else:
            vectors = np.zeros((0, self.embedder.dimension), dtype=np.float32)
        np.savez_compressed(self._path, keys=np.array(keys), vectors=vectors)

    def embed_many(self, texts: Sequence[str]) -> np.ndarray:
        self._load()
        texts = list(texts)
        if not texts:
            return np.zeros((0, self.embedder.dimension), dtype=np.float32)

        keys = [self._key(t) for t in texts]
        missing_positions = [i for i, k in enumerate(keys) if k not in self._store]
        if missing_positions:
            missing_texts = [texts[i] for i in missing_positions]
            new_vectors = self.embedder.embed(missing_texts)
            for pos, vec in zip(missing_positions, new_vectors):
                self._store[keys[pos]] = np.asarray(vec, dtype=np.float32)
            self._save()
        return np.vstack([self._store[k] for k in keys])

    @property
    def cache_size(self) -> int:
        self._load()
        return len(self._store)


def _cosine_similarity_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a_norm = a / np.clip(np.linalg.norm(a, axis=1, keepdims=True), 1e-12, None)
    b_norm = b / np.clip(np.linalg.norm(b, axis=1, keepdims=True), 1e-12, None)
    return a_norm @ b_norm.T


class EmbeddingNearestCentroidClassifier(BaseIntentClassifier):
    """Embedding classifier: one centroid per training intent (the mean
    embedding of that intent's training examples), classify by cosine
    similarity to the nearest centroid.

    `confidence` is a softmax over similarities to all centroids — a
    relative measure of how much more this message resembles the winning
    intent than the alternatives, not a calibrated probability.
    """

    def __init__(
        self,
        taxonomy: IntentTaxonomy,
        embedder: Optional[Embedder] = None,
        cache_dir: Optional[str] = None,
    ):
        super().__init__(taxonomy)
        self.embedder = embedder or HashingEmbedder()
        self.cache = EmbeddingCache(cache_dir, self.embedder) if cache_dir else None
        self._centroids: dict[str, np.ndarray] = {}
        self._centroid_labels: list[str] = []
        self._centroid_matrix: Optional[np.ndarray] = None

    def _embed(self, texts: Sequence[str]) -> np.ndarray:
        if self.cache is not None:
            return self.cache.embed_many(texts)
        return self.embedder.embed(list(texts))

    def fit(self, texts: Sequence[str], labels: Sequence[str]) -> "EmbeddingNearestCentroidClassifier":
        texts = list(texts)
        labels = list(labels)
        if not texts:
            raise ValueError("Cannot fit EmbeddingNearestCentroidClassifier on zero training examples.")
        if len(texts) != len(labels):
            raise ValueError("texts and labels must be the same length.")
        self.taxonomy.validate_labels(labels)

        vectors = self._embed(texts)
        by_label: dict[str, list[np.ndarray]] = {}
        for vec, label in zip(vectors, labels):
            by_label.setdefault(label, []).append(vec)

        self._centroids = {label: np.mean(np.vstack(vecs), axis=0) for label, vecs in by_label.items()}
        self._centroid_labels = list(self._centroids.keys())
        self._centroid_matrix = np.vstack([self._centroids[l] for l in self._centroid_labels])
        self._fitted = True
        return self

    def predict(self, texts: Sequence[str]) -> list[IntentPrediction]:
        self._require_fitted()
        texts = list(texts)
        vectors = self._embed(texts)
        sims = _cosine_similarity_matrix(vectors, self._centroid_matrix)

        predictions = []
        for row in sims:
            exp = np.exp(row - row.max())
            probs = exp / exp.sum()
            idx = int(probs.argmax())
            label = self._centroid_labels[idx]
            confidence = float(probs[idx])

            sorted_idx = np.argsort(-row)
            top_label = self._centroid_labels[sorted_idx[0]]
            reason = (
                f"Embedding nearest-centroid ({self.embedder.name}): closest to the "
                f"'{top_label}' training centroid (cosine similarity {row[idx]:.3f})"
            )
            if len(sorted_idx) > 1:
                runner_up = self._centroid_labels[sorted_idx[1]]
                reason += f", ahead of runner-up '{runner_up}'."
            else:
                reason += "."

            predictions.append(IntentPrediction(intent=label, confidence=confidence, reason=reason))
        return predictions
