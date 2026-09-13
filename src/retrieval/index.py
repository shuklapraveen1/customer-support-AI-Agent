"""Build the historical resolution corpus retrieval search over.

The corpus is built from customer -> brand resolution pairs
(`src.data.conversations.extract_resolution_pairs`) for a single brand,
restricted to conversations assigned to the TRAIN split. This restriction
is unconditional: `build_resolution_corpus` does not take a `split`
argument at all, specifically so there is no way to accidentally point it
at dev/test conversations — the retrieval corpus is the thing a later
drafting phase treats as "things we're allowed to reference," and if it
contained dev/test resolutions, evaluating on those splits would leak the
answer key into the retrieval evidence.

Each record's `intent` field is a *prediction*, not a gold label — there is
no hand-labeled intent for historical Twitter resolutions. Predictions come
from an `EmbeddingNearestCentroidClassifier` bootstrapped on nothing but
the worked examples already sitting in `configs/intents.yaml` (each
intent's `examples` list acts as a tiny set of seed exemplars). This is a
defensible way to get *some* intent tag onto historical evidence for
filtering purposes, but it is a heuristic bootstrap on top of a placeholder
taxonomy (see configs/intents.yaml's own header) and must not be reported
as verified/gold intent labels.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

from src.config import Config
from src.data.conversations import build_conversations, extract_resolution_pairs
from src.data.normalize import load_raw_csv, normalize_dataframe
from src.data.split import split_conversations
from src.intent.classifier import IntentTaxonomy
from src.intent.embedding import Embedder, EmbeddingNearestCentroidClassifier


@dataclass(frozen=True)
class ResolutionRecord:
    resolution_id: str
    customer_message: str
    brand_reply: str
    intent: str
    conversation_id: str
    timestamp: Optional[datetime]
    brand: str

    def as_dict(self) -> dict:
        d = asdict(self)
        d["timestamp"] = self.timestamp.isoformat() if self.timestamp else None
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "ResolutionRecord":
        ts = d.get("timestamp")
        return cls(
            resolution_id=d["resolution_id"],
            customer_message=d["customer_message"],
            brand_reply=d["brand_reply"],
            intent=d["intent"],
            conversation_id=d["conversation_id"],
            timestamp=datetime.fromisoformat(ts) if ts else None,
            brand=d["brand"],
        )


def _resolution_id(conversation_id: str, customer_tweet_id: str, brand_tweet_id: str) -> str:
    """Stable, content-derived id — deterministic across rebuilds so a
    cached embedding or a saved index still lines up with the same record
    after re-running the pipeline on unchanged data."""
    raw = f"{conversation_id}:{customer_tweet_id}:{brand_tweet_id}"
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
    return f"res_{digest}"


def _bootstrap_intent_classifier(taxonomy: IntentTaxonomy, embedder: Embedder) -> EmbeddingNearestCentroidClassifier:
    """Fit an embedding nearest-centroid classifier using nothing but the
    worked examples in the taxonomy itself as seed training data. This is a
    bootstrap, not a trained-on-real-labels classifier — see this module's
    docstring."""
    texts: list[str] = []
    labels: list[str] = []
    for intent_id in taxonomy.labels:
        definition = taxonomy.get(intent_id)
        for example in definition.examples:
            texts.append(example)
            labels.append(intent_id)
    clf = EmbeddingNearestCentroidClassifier(taxonomy, embedder=embedder)
    clf.fit(texts, labels)
    return clf


def build_resolution_corpus(
    config: Config,
    brand: str,
    taxonomy: Optional[IntentTaxonomy] = None,
    intent_embedder: Optional[Embedder] = None,
) -> list[ResolutionRecord]:
    """Build the historical resolution corpus for `brand` from TRAIN-split
    conversations only.

    If `taxonomy` is provided, each record's `intent` is predicted by a
    taxonomy-example-bootstrapped classifier (see
    `_bootstrap_intent_classifier`); otherwise every record's intent is
    `"unknown"`.
    """
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
    train_ids = set(assignment.train)  # the ONLY split this function ever reads from

    brand_train_pairs = [p for p in pairs if p.brand == brand and p.conversation_id in train_ids]

    intent_by_message: dict[int, str] = {}
    if taxonomy is not None and brand_train_pairs:
        from src.intent.embedding import HashingEmbedder

        clf = _bootstrap_intent_classifier(taxonomy, intent_embedder or HashingEmbedder())
        predictions = clf.predict([p.customer_message for p in brand_train_pairs])
        intent_by_message = {i: pred.intent for i, pred in enumerate(predictions)}

    records = []
    for i, p in enumerate(brand_train_pairs):
        intent = intent_by_message.get(i, "unknown")
        records.append(
            ResolutionRecord(
                resolution_id=_resolution_id(p.conversation_id, p.customer_tweet_id, p.brand_tweet_id),
                customer_message=p.customer_message,
                brand_reply=p.brand_reply,
                intent=intent,
                conversation_id=p.conversation_id,
                timestamp=p.brand_timestamp or p.customer_timestamp,
                brand=p.brand,
            )
        )
    return records


def save_corpus_parquet(records: list[ResolutionRecord], path: str) -> None:
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([r.as_dict() for r in records])
    df.to_parquet(path_obj, index=False)


def load_corpus_parquet(path: str) -> list[ResolutionRecord]:
    df = pd.read_parquet(path)
    return [ResolutionRecord.from_dict(row) for row in df.to_dict(orient="records")]
