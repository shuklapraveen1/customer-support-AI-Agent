"""Search over the historical resolution corpus.

Two retrieval implementations, sharing the same `retrieve(query, intent=None,
top_k=5)` contract and the same `RetrievalResult` shape:

- `ResolutionIndex` — semantic search: embeds the corpus (via any
  `Embedder`, cached through `EmbeddingCache`), indexes it with FAISS when
  available (`faiss.IndexFlatIP` over L2-normalized vectors, i.e. exact
  cosine similarity) and falls back automatically to
  `sklearn.neighbors.NearestNeighbors` (cosine metric) if FAISS can't be
  imported — matching "FAISS is preferred if practical; if it causes
  portability problems, use sklearn." Which backend actually ran is
  recorded on `ResolutionIndex.backend_name` rather than hidden.

- `TfidfRetrievalBaseline` — a simple, non-semantic baseline: TF-IDF +
  cosine similarity over the same historical customer messages. This is the
  retrieval half of "two baselines" for the eventual end-to-end system.

Both support optional intent filtering, and both apply the same ranking
rule: primary key is similarity score (descending); ties are broken by
recency (most recent first) — see `_rank_and_slice`. The similarity score
returned is always the true similarity for that (query, candidate) pair,
never adjusted by the recency tiebreak, and the timestamp is always
included in the result, whatever it is (including missing).

`evaluate_retrieval` implements a *proxy* Recall@k / MRR evaluation — see
its docstring for exactly what it does and does not measure.
"""

from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, Protocol, Sequence

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.intent.embedding import Embedder, EmbeddingCache
from src.retrieval.index import ResolutionRecord


@dataclass(frozen=True)
class RetrievalResult:
    resolution_id: str
    customer_message: str
    brand_reply: str
    intent: str
    similarity_score: float
    conversation_id: str
    timestamp: Optional[datetime]

    def as_dict(self) -> dict:
        return {
            "resolution_id": self.resolution_id,
            "customer_message": self.customer_message,
            "brand_reply": self.brand_reply,
            "intent": self.intent,
            "similarity_score": self.similarity_score,
            "conversation_id": self.conversation_id,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
        }


def _normalize_rows(vectors: np.ndarray) -> np.ndarray:
    norms = np.clip(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12, None)
    return vectors / norms


def _rank_and_slice(
    candidate_positions: Sequence[int],
    scores: Sequence[float],
    records: Sequence[ResolutionRecord],
    top_k: int,
) -> list[RetrievalResult]:
    """Sort candidates by (similarity desc, recency desc) and take the top_k.

    Recency only ever breaks ties in similarity — it never lets an
    older-but-more-similar result lose to a newer-but-less-similar one. A
    missing timestamp sorts as "oldest" for ranking purposes but is still
    reported as None in the result, never hidden or substituted.
    """
    scored = list(zip(candidate_positions, scores))

    def _sort_key(pos_score: tuple[int, float]) -> tuple[float, float]:
        pos, score = pos_score
        ts = records[pos].timestamp
        recency_key = -ts.timestamp() if ts is not None else float("inf")
        return (-round(float(score), 6), recency_key)

    scored.sort(key=_sort_key)
    results = []
    for pos, score in scored[:top_k]:
        r = records[pos]
        results.append(
            RetrievalResult(
                resolution_id=r.resolution_id,
                customer_message=r.customer_message,
                brand_reply=r.brand_reply,
                intent=r.intent,
                similarity_score=float(score),
                conversation_id=r.conversation_id,
                timestamp=r.timestamp,
            )
        )
    return results


class _IndexBackend(Protocol):
    def search(self, query_vector: np.ndarray, top_k: int) -> tuple[list[int], list[float]]: ...


class _FaissBackend:
    def __init__(self, vectors: np.ndarray):
        import faiss  # local import: only required when this backend is actually used

        self._faiss = faiss
        normalized = _normalize_rows(vectors).astype("float32")
        self.index = faiss.IndexFlatIP(normalized.shape[1])
        self.index.add(normalized)
        self._n = normalized.shape[0]

    def search(self, query_vector: np.ndarray, top_k: int) -> tuple[list[int], list[float]]:
        q = _normalize_rows(query_vector.reshape(1, -1)).astype("float32")
        k = min(top_k, self._n)
        scores, indices = self.index.search(q, k)
        valid = [(int(i), float(s)) for i, s in zip(indices[0], scores[0]) if i >= 0]
        return [i for i, _ in valid], [s for _, s in valid]


class _SklearnBackend:
    def __init__(self, vectors: np.ndarray):
        from sklearn.neighbors import NearestNeighbors

        self._normalized = _normalize_rows(vectors)
        self._n = self._normalized.shape[0]
        self.model = NearestNeighbors(metric="cosine", n_neighbors=self._n)
        self.model.fit(self._normalized)

    def search(self, query_vector: np.ndarray, top_k: int) -> tuple[list[int], list[float]]:
        q = _normalize_rows(query_vector.reshape(1, -1))
        k = min(top_k, self._n)
        distances, indices = self.model.kneighbors(q, n_neighbors=k)
        similarities = [1.0 - d for d in distances[0]]
        return list(indices[0]), similarities


def _build_backend(vectors: np.ndarray, prefer_faiss: bool = True) -> tuple[_IndexBackend, str]:
    if prefer_faiss:
        try:
            return _FaissBackend(vectors), "faiss"
        except ImportError:
            pass
    return _SklearnBackend(vectors), "sklearn"


class ResolutionIndex:
    """Semantic search index over a `ResolutionRecord` corpus."""

    def __init__(
        self,
        records: list[ResolutionRecord],
        vectors: np.ndarray,
        embedder: Embedder,
        backend: _IndexBackend,
        backend_name: str,
    ):
        self.records = records
        self.vectors = vectors
        self.embedder = embedder
        self._backend = backend
        self.backend_name = backend_name

        self._intent_positions: dict[str, list[int]] = {}
        for i, r in enumerate(records):
            self._intent_positions.setdefault(r.intent, []).append(i)

    @classmethod
    def build(
        cls,
        records: list[ResolutionRecord],
        embedder: Embedder,
        cache_dir: Optional[str] = None,
        prefer_faiss: bool = True,
    ) -> "ResolutionIndex":
        if not records:
            raise ValueError("Cannot build a ResolutionIndex over zero records.")
        texts = [r.customer_message for r in records]
        if cache_dir:
            vectors = EmbeddingCache(cache_dir, embedder).embed_many(texts)
        else:
            vectors = embedder.embed(texts)
        backend, backend_name = _build_backend(np.asarray(vectors), prefer_faiss=prefer_faiss)
        return cls(records, np.asarray(vectors), embedder, backend, backend_name)

    def retrieve(self, query: str, intent: Optional[str] = None, top_k: int = 5) -> list[RetrievalResult]:
        if top_k <= 0:
            raise ValueError("top_k must be a positive integer.")
        query_vector = np.asarray(self.embedder.embed([query])[0])

        if intent is not None:
            candidate_positions = self._intent_positions.get(intent, [])
            if not candidate_positions:
                return []
            candidate_vectors = self.vectors[candidate_positions]
            sims = cosine_similarity(query_vector.reshape(1, -1), candidate_vectors)[0]
            return _rank_and_slice(candidate_positions, sims, self.records, top_k)

        positions, scores = self._backend.search(query_vector, top_k)
        return _rank_and_slice(positions, scores, self.records, top_k)

    def save(self, index_dir: str) -> None:
        """Persist embeddings + metadata to `index_dir`.

        Deliberately does NOT serialize the FAISS/sklearn backend object
        itself — only the raw vectors and a small metadata file. On load,
        the backend is rebuilt from the vectors (cheap for corpus sizes
        this project deals with), which sidesteps FAISS's own binary index
        format entirely and keeps `artifacts/index/` portable across
        machines and across whichever backend happened to be available
        when it was built vs. loaded.
        """
        path = Path(index_dir)
        path.mkdir(parents=True, exist_ok=True)
        np.save(path / "vectors.npy", self.vectors)
        meta = {
            "embedder_name": self.embedder.name,
            "backend_name_at_build_time": self.backend_name,
            "n_records": len(self.records),
            "vector_dimension": int(self.vectors.shape[1]),
        }
        with open(path / "meta.json", "w", encoding="utf-8") as fh:
            json.dump(meta, fh, indent=2)

    @classmethod
    def load(
        cls,
        index_dir: str,
        records: list[ResolutionRecord],
        embedder: Embedder,
        prefer_faiss: bool = True,
    ) -> "ResolutionIndex":
        """Load vectors saved by `save()` and rebuild a search backend over
        them. `records` must be the exact same corpus (same order) the
        index was built from — pass what
        `src.retrieval.index.load_corpus_parquet` returns for the matching
        `artifacts/resolutions.parquet`.
        """
        path = Path(index_dir)
        vectors = np.load(path / "vectors.npy")
        if len(records) != vectors.shape[0]:
            raise ValueError(
                f"records/vectors length mismatch ({len(records)} vs {vectors.shape[0]}) — "
                "was this index built from a different corpus?"
            )
        backend, backend_name = _build_backend(vectors, prefer_faiss=prefer_faiss)
        return cls(records, vectors, embedder, backend, backend_name)


def retrieve(index: ResolutionIndex, query: str, intent: Optional[str] = None, top_k: int = 5) -> list[RetrievalResult]:
    """Module-level convenience wrapper matching the assignment's literal
    `retrieve(query, intent=None, top_k=5)` signature."""
    return index.retrieve(query, intent=intent, top_k=top_k)


class TfidfRetrievalBaseline:
    """Baseline retrieval: TF-IDF + cosine similarity over historical
    customer messages. No semantic embedding at all — this is what the
    embedding-based `ResolutionIndex` has to beat, and it's also the
    retrieval component the eventual simple end-to-end baseline uses.
    """

    def __init__(self, records: list[ResolutionRecord]):
        if not records:
            raise ValueError("Cannot build TfidfRetrievalBaseline over zero records.")
        self.records = records
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
        self._matrix = self.vectorizer.fit_transform([r.customer_message for r in records])

        self._intent_positions: dict[str, list[int]] = {}
        for i, r in enumerate(records):
            self._intent_positions.setdefault(r.intent, []).append(i)

    def retrieve(self, query: str, intent: Optional[str] = None, top_k: int = 5) -> list[RetrievalResult]:
        if top_k <= 0:
            raise ValueError("top_k must be a positive integer.")
        query_vec = self.vectorizer.transform([query])

        if intent is not None:
            candidate_positions = self._intent_positions.get(intent, [])
            if not candidate_positions:
                return []
            sims = cosine_similarity(query_vec, self._matrix[candidate_positions])[0]
            return _rank_and_slice(candidate_positions, sims, self.records, top_k)

        sims = cosine_similarity(query_vec, self._matrix)[0]
        all_positions = list(range(len(self.records)))
        return _rank_and_slice(all_positions, sims, self.records, top_k)


# --- Retrieval evaluation ----------------------------------------------------


@dataclass
class RetrievalEvalReport:
    recall_at_1: float
    recall_at_3: float
    recall_at_5: float
    mrr: float
    n_queries: int
    method: str

    def as_dict(self) -> dict:
        return {
            "recall_at_1": self.recall_at_1,
            "recall_at_3": self.recall_at_3,
            "recall_at_5": self.recall_at_5,
            "mrr": self.mrr,
            "n_queries": self.n_queries,
            "method": self.method,
        }


_WORD_RE = re.compile(r"\S+")

PROXY_METHOD_DESCRIPTION = (
    "self-retrieval with synthetic word-drop query perturbation (PROXY METRIC, "
    "not human-judged relevance — see build_self_retrieval_eval_queries docstring)"
)


def _perturb_message(text: str, rng: random.Random, drop_word_prob: float) -> str:
    """Light, deterministic (given `rng`) paraphrase-ish corruption: drop
    some words and lowercase everything, but keep enough content that the
    corrupted query is still recognizably about the same issue."""
    words = _WORD_RE.findall(text)
    kept = [w for w in words if rng.random() > drop_word_prob]
    if not kept:
        kept = words[:1]
    return " ".join(kept).lower()


def build_self_retrieval_eval_queries(
    records: list[ResolutionRecord],
    seed: int = 42,
    sample_size: Optional[int] = None,
    drop_word_prob: float = 0.3,
) -> list[tuple[str, str]]:
    """Build a *proxy* retrieval evaluation set: `(perturbed_query,
    gold_resolution_id)` pairs, where `gold_resolution_id` is the id of the
    exact record the perturbed query was derived from.

    LIMITATION — read before trusting these numbers: this dataset has no
    human relevance judgments linking a genuinely new customer message to
    "the single correct historical precedent" for it, and building one is
    exactly the kind of hand-labeling that belongs in the golden-eval-set
    phase, not here. In the absence of that, this proxy asks a narrower,
    answerable question instead: *given a corrupted, paraphrase-like
    version of a message already in the corpus, can retrieval still find
    that exact original message again?* A high score here means the index
    and embedder can survive superficial rewording; it does NOT mean the
    system finds the best historical precedent for a truly novel complaint
    the corpus has never seen anything like. Report and read these numbers
    with that distinction explicit — do not present them as IR relevance
    metrics.
    """
    rng = random.Random(seed)
    pool = list(records)
    rng.shuffle(pool)
    if sample_size is not None:
        pool = pool[:sample_size]

    queries = []
    for r in pool:
        perturbed = _perturb_message(r.customer_message, rng, drop_word_prob)
        if perturbed.strip():
            queries.append((perturbed, r.resolution_id))
    return queries


def evaluate_retrieval(
    searcher,
    eval_queries: list[tuple[str, str]],
    top_k: int = 5,
) -> RetrievalEvalReport:
    """Compute Recall@1/3/5 and MRR of `searcher` (a `ResolutionIndex` or
    `TfidfRetrievalBaseline`, or anything with a compatible `.retrieve()`)
    against `eval_queries` — real calls to `searcher.retrieve(...)` for
    every query, never estimated or hardcoded.
    """
    if not eval_queries:
        raise ValueError("eval_queries must be non-empty.")
    if top_k < 5:
        raise ValueError("top_k must be at least 5 to compute Recall@5.")

    hits_at = {1: 0, 3: 0, 5: 0}
    reciprocal_ranks = []

    for query_text, gold_id in eval_queries:
        results = searcher.retrieve(query_text, top_k=top_k)
        found_rank = None
        for rank, result in enumerate(results, start=1):
            if result.resolution_id == gold_id:
                found_rank = rank
                break
        if found_rank is not None:
            reciprocal_ranks.append(1.0 / found_rank)
            for k in (1, 3, 5):
                if found_rank <= k:
                    hits_at[k] += 1
        else:
            reciprocal_ranks.append(0.0)

    n = len(eval_queries)
    return RetrievalEvalReport(
        recall_at_1=hits_at[1] / n,
        recall_at_3=hits_at[3] / n,
        recall_at_5=hits_at[5] / n,
        mrr=sum(reciprocal_ranks) / n,
        n_queries=n,
        method=PROXY_METHOD_DESCRIPTION,
    )
