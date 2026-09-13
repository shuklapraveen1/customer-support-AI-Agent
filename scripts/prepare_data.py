#!/usr/bin/env python
"""End-to-end Phase 1 data preparation.

    python scripts/prepare_data.py [--config configs/config.yaml]

Loads the raw CSV, normalizes it, reconstructs conversations, extracts
customer -> brand resolution pairs, computes a conversation-level
train/dev/test split, and writes everything under the configured
processed/splits directories.

This script does not classify intents, retrieve anything, or call an LLM —
that is a later phase.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

# Allow running as `python scripts/prepare_data.py` from the repo root
# without requiring an editable install.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config
from src.data.conversations import build_conversations, extract_resolution_pairs
from src.data.normalize import load_raw_csv, normalize_dataframe
from src.data.split import split_conversations


def _conv_to_dict(conv) -> dict:
    d = asdict(conv)
    for t in d["turns"]:
        t["timestamp"] = t["timestamp"].isoformat() if t["timestamp"] else None
    return d


def _pair_to_dict(pair) -> dict:
    d = asdict(pair)
    d["customer_timestamp"] = d["customer_timestamp"].isoformat() if d["customer_timestamp"] else None
    d["brand_timestamp"] = d["brand_timestamp"].isoformat() if d["brand_timestamp"] else None
    return d


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare the Hiver support-agent data foundation.")
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    args = parser.parse_args()

    config = load_config(args.config).resolve_paths()

    processed_dir = Path(config.paths.processed_dir)
    splits_dir = Path(config.paths.splits_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)
    splits_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading raw CSV from {config.paths.raw_csv} ...")
    df = load_raw_csv(config.paths.raw_csv)

    print("Normalizing ...")
    tweets, norm_report = normalize_dataframe(df)
    print(f"  {norm_report.as_dict()}")

    print("Building conversations ...")
    conversations = build_conversations(tweets)
    print(f"  {len(conversations)} conversations")

    print("Extracting resolution pairs ...")
    pairs = extract_resolution_pairs(conversations)
    print(f"  {len(pairs)} resolution pairs")

    print(
        "Splitting conversation ids "
        f"(train={config.split.train}, dev={config.split.dev}, "
        f"test={config.split.test}, seed={config.split.seed}) ..."
    )
    conversation_ids = [c.conversation_id for c in conversations]
    assignment = split_conversations(
        conversation_ids,
        train=config.split.train,
        dev=config.split.dev,
        test=config.split.test,
        seed=config.split.seed,
    )
    print(f"  train={len(assignment.train)} dev={len(assignment.dev)} test={len(assignment.test)}")

    conversations_path = processed_dir / "conversations.jsonl"
    with open(conversations_path, "w", encoding="utf-8") as fh:
        for conv in conversations:
            fh.write(json.dumps(_conv_to_dict(conv)) + "\n")

    pairs_path = processed_dir / "resolution_pairs.jsonl"
    with open(pairs_path, "w", encoding="utf-8") as fh:
        for pair in pairs:
            fh.write(json.dumps(_pair_to_dict(pair)) + "\n")

    norm_report_path = processed_dir / "normalization_report.json"
    with open(norm_report_path, "w", encoding="utf-8") as fh:
        json.dump(norm_report.as_dict(), fh, indent=2)

    for name, ids in assignment.as_dict().items():
        with open(splits_dir / f"{name}_conversation_ids.json", "w", encoding="utf-8") as fh:
            json.dump(ids, fh, indent=2)

    print("\nWrote:")
    print(f"  {conversations_path}")
    print(f"  {pairs_path}")
    print(f"  {norm_report_path}")
    for name in ("train", "dev", "test"):
        print(f"  {splits_dir / (name + '_conversation_ids.json')}")


if __name__ == "__main__":
    main()
