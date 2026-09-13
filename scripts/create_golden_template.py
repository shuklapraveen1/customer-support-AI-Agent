#!/usr/bin/env python
"""Sample a stratified golden evaluation set template.

    python scripts/create_golden_template.py --brand <brand> --target-size 200

Draws candidate customer messages ONLY from DEV+TEST split conversations —
never TRAIN, since the golden set must never overlap what the retrieval
corpus or any classifier's training data was built from (that overlap is
exactly what `src.evaluation.run_eval`'s leakage checks look for).

Stratifies the sample across common/rare predicted intent, ambiguous
intent confidence, short/long/noisy messages, apparent missing context,
and the CURRENT system's own predicted auto_handle-vs-escalate decision —
using the system's predictions purely as SAMPLING STRATA to make sure the
golden set covers a range of situations, never as gold labels themselves.
A human's actual judgment of intent/escalation/reply-quality is expected to
sometimes disagree with these strata — that disagreement is exactly what
evaluation is for.

Writes `data/golden/golden_set.csv` with every `gold_*` column literally
set to the string `"UNLABELLED"`. This script does NOT use an LLM to guess
what a human would label, and does not invent labels under any
circumstances — see docs/GOLDEN_SET.md before filling it in by hand.
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.agent import run_agent
from src.agent.escalation import load_escalation_config
from src.agent.llm import MockLLMProvider
from src.config import load_config
from src.data.conversations import build_conversations
from src.data.normalize import is_url_only, load_raw_csv, normalize_dataframe
from src.data.split import split_conversations
from src.intent.baseline_tfidf import TfidfLogisticClassifier
from src.intent.classifier import IntentTaxonomy
from src.intent.embedding import HashingEmbedder
from src.retrieval.index import build_resolution_corpus
from src.retrieval.search import ResolutionIndex

GOLD_FIELDS = [
    "example_id",
    "conversation_id",
    "customer_message",
    "gold_intent",
    "gold_should_escalate",
    "gold_reason",
    "gold_reply_quality",
    "notes",
]
UNLABELLED = "UNLABELLED"


class _EmptySearcher:
    def retrieve(self, query, intent=None, top_k=5):
        return []


def _word_count(text: str) -> int:
    return len(text.split())


def _is_noisy(text: str) -> bool:
    if is_url_only(text):
        return True
    letters = sum(1 for c in text if c.isalpha())
    non_letters = len(text) - letters
    if len(text) > 0 and non_letters / len(text) > 0.5:
        return True
    if text.count("!") + text.count("?") >= 3:
        return True
    if letters > 3 and sum(1 for c in text if c.isupper()) / letters > 0.7:
        return True
    return False


def build_candidate_pool(config, brand):
    """Every customer turn from DEV+TEST conversations — never TRAIN."""
    df = load_raw_csv(config.paths.raw_csv)
    tweets, _ = normalize_dataframe(df)
    conversations = build_conversations(tweets)
    conversation_ids = [c.conversation_id for c in conversations]
    assignment = split_conversations(
        conversation_ids, train=config.split.train, dev=config.split.dev, test=config.split.test, seed=config.split.seed
    )
    eligible_ids = set(assignment.dev) | set(assignment.test)

    candidates = []
    seen = set()
    for conv in conversations:
        if conv.conversation_id not in eligible_ids:
            continue
        if brand and conv.brand != brand:
            continue
        for turn in conv.turns:
            if turn.author_type != "customer" or not turn.text.strip():
                continue
            key = (conv.conversation_id, turn.tweet_id)
            if key in seen:
                continue
            seen.add(key)
            candidates.append({"conversation_id": conv.conversation_id, "customer_message": turn.text})
    return candidates


def annotate_strata(candidates, classifier, searcher, escalation_config):
    for cand in candidates:
        prediction = classifier.predict([cand["customer_message"]])[0]
        cand["_predicted_intent"] = prediction.intent
        cand["_predicted_confidence"] = prediction.confidence
        cand["_word_count"] = _word_count(cand["customer_message"])
        cand["_noisy"] = _is_noisy(cand["customer_message"])
        try:
            result = run_agent(
                cand["customer_message"],
                intent_classifier=classifier,
                searcher=searcher,
                llm_provider=MockLLMProvider(),
                escalation_config=escalation_config,
            )
            cand["_predicted_decision"] = result.decision
        except Exception:
            cand["_predicted_decision"] = "unknown"


def assign_buckets(candidates, short_max_words=4, long_min_words=25, ambiguous_conf=0.4):
    intent_counts = Counter(c["_predicted_intent"] for c in candidates)
    common_intents = {i for i, _ in intent_counts.most_common(3)}
    rare_intents = {i for i, _ in intent_counts.most_common()[-3:]} if intent_counts else set()

    for c in candidates:
        buckets = set()
        if c["_predicted_intent"] in common_intents:
            buckets.add("common_intent")
        if c["_predicted_intent"] in rare_intents:
            buckets.add("rare_intent")
        if c["_predicted_confidence"] < ambiguous_conf:
            buckets.add("ambiguous")
        if c["_word_count"] <= short_max_words:
            buckets.add("short_message")
            buckets.add("missing_context")
        if c["_word_count"] >= long_min_words:
            buckets.add("long_message")
        if c["_noisy"]:
            buckets.add("noisy_message")
        if c["_predicted_decision"] == "escalate":
            buckets.add("predicted_escalate")
        elif c["_predicted_decision"] == "auto_handle":
            buckets.add("predicted_auto_handle")
        c["_buckets"] = buckets


def stratified_sample(candidates, target_n, seed):
    rng = random.Random(seed)
    pool = list(candidates)
    rng.shuffle(pool)

    all_buckets = sorted({b for c in pool for b in c["_buckets"]}) or ["uncategorized"]
    min_per_bucket = max(1, target_n // max(1, len(all_buckets)) // 2)

    def _key(c):
        return (c["conversation_id"], c["customer_message"])

    selected = []
    selected_keys = set()

    for bucket in all_buckets:
        bucket_pool = [c for c in pool if bucket in c["_buckets"] and _key(c) not in selected_keys]
        for c in bucket_pool[:min_per_bucket]:
            selected.append(c)
            selected_keys.add(_key(c))
        if len(selected) >= target_n:
            break

    if len(selected) < target_n:
        for c in pool:
            if len(selected) >= target_n:
                break
            if _key(c) not in selected_keys:
                selected.append(c)
                selected_keys.add(_key(c))

    rng.shuffle(selected)
    return selected[:target_n]


def write_golden_csv(selected, path):
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=GOLD_FIELDS)
        writer.writeheader()
        for i, c in enumerate(selected, start=1):
            buckets_str = ",".join(sorted(c["_buckets"])) or "uncategorized"
            note = (
                f"stratification_buckets={buckets_str}; "
                f"system_predicted_intent={c['_predicted_intent']} (confidence={c['_predicted_confidence']:.2f}); "
                f"system_predicted_decision={c['_predicted_decision']}. "
                "These are SYSTEM SUGGESTIONS for the annotator's convenience only — they are "
                "NOT labels, may well be wrong, and must be judged independently."
            )
            writer.writerow(
                {
                    "example_id": f"golden_{i:04d}",
                    "conversation_id": c["conversation_id"],
                    "customer_message": c["customer_message"],
                    "gold_intent": UNLABELLED,
                    "gold_should_escalate": UNLABELLED,
                    "gold_reason": UNLABELLED,
                    "gold_reply_quality": UNLABELLED,
                    "notes": note,
                }
            )


def main():
    parser = argparse.ArgumentParser(description="Sample a stratified golden evaluation set template.")
    parser.add_argument("--brand", default=None, help="Restrict to one brand; omit to sample across all brands")
    parser.add_argument("--target-size", type=int, default=200)
    parser.add_argument("--config", default=None)
    parser.add_argument("--intents-config", default=None)
    parser.add_argument("--escalation-config", default=None)
    parser.add_argument("--out", default=str(Path("data") / "golden" / "golden_set.csv"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    config = load_config(args.config).resolve_paths()
    taxonomy = IntentTaxonomy.from_yaml(args.intents_config)
    escalation_config = load_escalation_config(args.escalation_config)

    print("Building candidate pool from DEV+TEST split conversations only (never TRAIN) ...")
    candidates = build_candidate_pool(config, args.brand)
    print(f"  {len(candidates)} candidate customer message(s) available")
    if not candidates:
        print("No candidates found — check --brand and that data/raw/twcs.csv is populated.")
        sys.exit(1)

    print(
        "Training a taxonomy-bootstrapped classifier and building a retrieval index (from "
        "TRAIN-split resolution pairs) purely to compute SAMPLING STRATA — not gold labels ..."
    )
    texts, labels = [], []
    for intent_id in taxonomy.labels:
        definition = taxonomy.get(intent_id)
        for example in definition.examples:
            texts.append(example)
            labels.append(intent_id)
    classifier = TfidfLogisticClassifier(taxonomy).fit(texts, labels)

    records = build_resolution_corpus(config, args.brand, taxonomy=taxonomy) if args.brand else []
    if records:
        searcher = ResolutionIndex.build(records, embedder=HashingEmbedder(dimension=128))
    else:
        print(
            "  no --brand given, or no TRAIN-split resolution pairs found for it — using an "
            "always-empty searcher; predicted_escalate/auto_handle strata will just reflect "
            "the resulting 'insufficient evidence' outcome, not a real evidence-backed decision."
        )
        searcher = _EmptySearcher()

    print("Annotating candidates with system-predicted strata (intent, confidence, decision) ...")
    annotate_strata(candidates, classifier, searcher, escalation_config)
    assign_buckets(candidates)

    target_n = args.target_size
    if len(candidates) < target_n:
        print(
            f"\nWARNING: only {len(candidates)} candidate messages are available, below the "
            f"target of {target_n}. This sandbox's dataset is a tiny synthetic fixture (the "
            "real Kaggle 'Customer Support on Twitter' dataset could not be downloaded here — "
            "see README), so the golden set produced now is necessarily demo-scale, NOT the "
            "real 150-250 example set the assignment calls for. Re-run this script against the "
            "real dataset once it's available to produce the full-size set.\n"
        )
        target_n = len(candidates)

    selected = stratified_sample(candidates, target_n, seed=args.seed)
    write_golden_csv(selected, args.out)

    print(f"Wrote {len(selected)} UNLABELLED examples to {args.out}")
    print("Every gold_* column is literally the string 'UNLABELLED' — fill them in by hand.")
    print("Read docs/GOLDEN_SET.md before you start labeling.")


if __name__ == "__main__":
    main()
