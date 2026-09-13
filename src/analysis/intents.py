"""Intent discovery workflow: turn a brand's own customer messages into
candidate intent clusters for a human (optionally LLM-assisted) to curate
into `configs/intents.yaml`.

This module produces *candidates*, not a taxonomy — nothing here writes to
`configs/intents.yaml`. Finalizing the taxonomy from these candidates is a
human step (see `scripts/create_intent_candidates.py` and the README).

Discovery deliberately only looks at customer messages from the **train**
split by default. The taxonomy is itself a modeling decision, made before
any classifier is trained on top of it — defining intents by looking at
dev/test messages would leak eval-set information into the label space
before evaluation even starts, the same way it would if a feature were
derived from the test set.

Methodology note on Banking77: Banking77 (a public dataset of banking-
customer-support utterances with a 77-intent taxonomy) is sometimes used as
a methodological reference for what a well-scoped intent taxonomy looks
like — flat, mutually-distinguishable, grounded in actual user utterances.
It is not used here as the final taxonomy: this project's data is general
Twitter customer support for a brand chosen from the dataset, not banking,
so Banking77's specific 77 labels (card issues, "activate_my_card",
foreign-currency transfers, etc.) do not describe this brand's actual
support issues at all. Any taxonomy actually used by this project's
classifier must come out of this module's clustering over that brand's own
customer messages, not off-the-shelf labels from an unrelated domain.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score

from src.config import Config, load_config
from src.data.conversations import build_conversations, extract_resolution_pairs
from src.data.normalize import load_raw_csv, normalize_dataframe
from src.data.split import split_conversations
from src.intent.embedding import Embedder, HashingEmbedder


@dataclass
class ClusterCandidate:
    cluster_id: int
    size: int
    keywords: list[str]
    representative_examples: list[str]
    suggested_name: str

    def as_dict(self) -> dict:
        return asdict(self)


def load_brand_customer_messages(config: Config, brand: str, split: str = "train") -> list[str]:
    """Customer messages from resolution pairs for `brand`, restricted to
    conversations assigned to `split` under the configured seed — so
    discovery never sees dev/test customer messages.
    """
    if split not in ("train", "dev", "test"):
        raise ValueError(f"split must be one of 'train'/'dev'/'test', got {split!r}")

    df = load_raw_csv(config.paths.raw_csv)
    tweets, _ = normalize_dataframe(df)
    conversations = build_conversations(tweets)
    pairs = extract_resolution_pairs(conversations)

    conversation_ids = [c.conversation_id for c in conversations]
    assignment = split_conversations(
        conversation_ids,
        train=config.split.train,
        dev=config.split.dev,
        test=config.split.test,
        seed=config.split.seed,
    )
    allowed_ids = set(getattr(assignment, split))

    return [p.customer_message for p in pairs if p.brand == brand and p.conversation_id in allowed_ids]


def _dedupe_preserve_order(messages: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for m in messages:
        key = m.strip().lower()
        if key and key not in seen:
            seen.add(key)
            out.append(m)
    return out


def _choose_k(n_samples: int, vectors: np.ndarray, min_k: int, max_k: int, seed: int) -> int:
    """Pick the number of clusters via silhouette score over [min_k, max_k],
    bounded by the number of available samples. This is how "the data
    determines the number" in practice: min_k/max_k set the assignment's
    8-15 guideline as a search range, not a fixed count, and silhouette
    score picks the best-separated value inside it. Falls back to a small
    fixed value if too few samples exist to search at all (e.g. in tests).
    """
    max_k = min(max_k, n_samples - 1)
    if max_k < min_k:
        return max(1, min(min_k, n_samples))

    best_k, best_score = min_k, -1.0
    for k in range(min_k, max_k + 1):
        try:
            km = KMeans(n_clusters=k, random_state=seed, n_init=10)
            labels = km.fit_predict(vectors)
            if len(set(labels)) < 2:
                continue
            score = silhouette_score(vectors, labels)
        except Exception:
            continue
        if score > best_score:
            best_k, best_score = k, score
    return best_k


def discover_intent_candidates(
    messages: Sequence[str],
    embedder: Optional[Embedder] = None,
    min_k: int = 8,
    max_k: int = 15,
    seed: int = 42,
    top_keywords: int = 8,
    representatives_per_cluster: int = 5,
) -> list[ClusterCandidate]:
    """Cluster `messages` and return one `ClusterCandidate` per cluster.

    - Clustering representation: `embedder` (defaults to a deterministic,
      offline `HashingEmbedder`; pass a `SentenceTransformerEmbedder` for
      production-quality semantic clustering).
    - Cluster count: chosen by silhouette score within [min_k, max_k] — see
      `_choose_k`.
    - Keywords: top TF-IDF terms for each cluster, from a TF-IDF fit over
      the deduplicated message set — a representation independent of
      whatever the clustering embedder used, so keywords stay interpretable
      even when clustering used opaque embeddings.
    - Representative examples: the actual messages closest to each
      cluster's centroid in embedding space.
    - `suggested_name`: a cheap heuristic (top keywords joined) — see
      `name_cluster_with_llm` for the optional LLM-assisted alternative.
    """
    deduped = _dedupe_preserve_order(messages)
    if len(deduped) < 2:
        raise ValueError(
            f"Need at least 2 distinct customer messages to cluster, got {len(deduped)}."
        )

    embedder = embedder or HashingEmbedder()
    vectors = embedder.embed(deduped)

    k = _choose_k(len(deduped), vectors, min_k=min_k, max_k=max_k, seed=seed)
    km = KMeans(n_clusters=k, random_state=seed, n_init=10)
    labels = km.fit_predict(vectors)

    tfidf = TfidfVectorizer(max_features=2000, stop_words="english", ngram_range=(1, 2))
    tfidf_matrix = tfidf.fit_transform(deduped)
    vocab = np.array(tfidf.get_feature_names_out())

    candidates = []
    for cluster_id in range(k):
        idx = np.where(labels == cluster_id)[0]
        if len(idx) == 0:
            continue

        cluster_tfidf_mean = np.asarray(tfidf_matrix[idx].mean(axis=0)).ravel()
        top_idx = cluster_tfidf_mean.argsort()[::-1][:top_keywords]
        keywords = [vocab[i] for i in top_idx if cluster_tfidf_mean[i] > 0]

        centroid = km.cluster_centers_[cluster_id]
        member_vectors = vectors[idx]
        dists = np.linalg.norm(member_vectors - centroid, axis=1)
        closest_order = idx[np.argsort(dists)]
        representatives = [deduped[i] for i in closest_order[:representatives_per_cluster]]

        suggested_name = " / ".join(keywords[:3]) if keywords else f"cluster_{cluster_id}"

        candidates.append(
            ClusterCandidate(
                cluster_id=int(cluster_id),
                size=int(len(idx)),
                keywords=list(keywords),
                representative_examples=representatives,
                suggested_name=suggested_name,
            )
        )

    candidates.sort(key=lambda c: c.size, reverse=True)
    return candidates


def name_cluster_with_llm(candidate: ClusterCandidate) -> Optional[str]:
    """Optional hook for LLM-assisted naming of a candidate cluster.

    Deliberately unimplemented in Phase 2 — no LLM calls are made anywhere
    in this module or its callers, matching "do not build reply generation
    yet." A later phase can implement this (e.g. send the cluster's
    keywords + representative examples to the Anthropic API and ask for a
    short label + description) without changing anything else in the
    discovery workflow. Returning `None` means "use the heuristic
    `suggested_name` instead."
    """
    return None


def print_candidates(candidates: list[ClusterCandidate]) -> None:
    print(f"=== {len(candidates)} candidate intent clusters ===\n")
    for c in candidates:
        print(f"[cluster {c.cluster_id}] size={c.size} suggested_name='{c.suggested_name}'")
        print(f"  keywords: {c.keywords}")
        print("  representative examples:")
        for ex in c.representative_examples:
            print(f"    - {ex}")
        print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Discover candidate intent clusters for a brand's train-split customer messages."
    )
    parser.add_argument("--brand", required=True, help="Brand author_id, e.g. AmazonHelp")
    parser.add_argument("--config", default=None)
    parser.add_argument("--min-k", type=int, default=8)
    parser.add_argument("--max-k", type=int, default=15)
    parser.add_argument("--split", default="train", choices=["train", "dev", "test"])
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args()

    config = load_config(args.config).resolve_paths()
    messages = load_brand_customer_messages(config, args.brand, split=args.split)
    print(f"Loaded {len(messages)} customer messages for brand='{args.brand}' (split={args.split})")

    candidates = discover_intent_candidates(messages, min_k=args.min_k, max_k=args.max_k, seed=config.split.seed)
    print_candidates(candidates)

    out_path = (
        Path(args.json_out)
        if args.json_out
        else Path(config.paths.processed_dir) / f"intent_candidates_{args.brand}.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump([c.as_dict() for c in candidates], fh, indent=2)
    print(f"\nCandidates written to {out_path}")
    print("Review these and hand-curate configs/intents.yaml — this script does not write it for you.")


if __name__ == "__main__":
    main()
