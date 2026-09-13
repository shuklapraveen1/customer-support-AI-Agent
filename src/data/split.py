"""Conversation-level train/dev/test split.

Splitting happens on `conversation_id`, never on individual tweets, so every
turn of a given conversation lands in exactly one split — this is what
guarantees no test leakage between a conversation's own turns.

The seeded shuffle is deterministic given the same seed and the same *set*
of conversation ids: inputs are de-duplicated and sorted before shuffling,
specifically so the split does not depend on whatever order conversations
happened to be discovered or passed in.
"""

from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass(frozen=True)
class SplitAssignment:
    train: list[str]
    dev: list[str]
    test: list[str]

    def as_dict(self) -> dict[str, list[str]]:
        return {"train": self.train, "dev": self.dev, "test": self.test}


def split_conversations(
    conversation_ids: list[str],
    train: float = 0.7,
    dev: float = 0.15,
    test: float = 0.15,
    seed: int = 42,
) -> SplitAssignment:
    """Deterministically assign each conversation id to train/dev/test.

    Every id is assigned exactly once; rounding remainders always fall to
    `test` last, so conversations never silently disappear from the split.
    """
    total = train + dev + test
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"train+dev+test must sum to 1.0, got {total}")

    unique_ids = sorted(set(conversation_ids))
    rng = random.Random(seed)
    rng.shuffle(unique_ids)

    n = len(unique_ids)
    n_train = min(round(n * train), n)
    n_dev = min(round(n * dev), n - n_train)

    train_ids = unique_ids[:n_train]
    dev_ids = unique_ids[n_train : n_train + n_dev]
    test_ids = unique_ids[n_train + n_dev :]

    return SplitAssignment(train=train_ids, dev=dev_ids, test=test_ids)


def assignment_lookup(assignment: SplitAssignment) -> dict[str, str]:
    """conversation_id -> "train" | "dev" | "test", for quick joins against
    conversations or resolution pairs."""
    lookup: dict[str, str] = {}
    for cid in assignment.train:
        lookup[cid] = "train"
    for cid in assignment.dev:
        lookup[cid] = "dev"
    for cid in assignment.test:
        lookup[cid] = "test"
    return lookup
