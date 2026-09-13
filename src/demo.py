#!/usr/bin/env python
"""Interactive / one-shot demo of the full agent pipeline.

    python -m src.demo
    python -m src.demo --message "My refund still hasn't arrived"

Builds a real (if bootstrapped/small, per this sandbox's data
constraints — see README) intent classifier and retrieval index, then
runs `run_agent` on either a single `--message` or messages typed
interactively, printing the full pipeline's output in the exact shape
requested: Customer / Intent / Confidence / Historical evidence / Draft
reply / Decision / Reason.

Uses the mock LLM provider by default (no API key needed) — pass
`--llm-provider openai` (with `OPENAI_API_KEY` set) for a real completion.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.agent import run_agent
from src.agent.escalation import load_escalation_config
from src.agent.llm import get_llm_provider
from src.intent.classifier import IntentTaxonomy
from src.intent.embedding import EmbeddingNearestCentroidClassifier, EmbeddingUnavailableError, HashingEmbedder, SentenceTransformerEmbedder
from src.retrieval.index import load_corpus_parquet
from src.retrieval.search import ResolutionIndex

DEFAULT_CORPUS_PATH = str(Path("artifacts") / "resolutions.parquet")


class _EmptySearcher:
    def retrieve(self, query, intent=None, top_k=5):
        return []


def _build_embedder(name: str):
    if name == "sentence-transformer":
        try:
            return SentenceTransformerEmbedder()
        except EmbeddingUnavailableError as exc:
            print(f"Could not load the production embedder ({exc}); falling back to HashingEmbedder.")
    return HashingEmbedder(dimension=256)


def build_pipeline(args):
    taxonomy = IntentTaxonomy.from_yaml(args.intents_config)

    texts, labels = [], []
    for intent_id in taxonomy.labels:
        definition = taxonomy.get(intent_id)
        for example in definition.examples:
            texts.append(example)
            labels.append(intent_id)

    embedder = _build_embedder(args.embedder)
    classifier = EmbeddingNearestCentroidClassifier(taxonomy, embedder=embedder).fit(texts, labels)

    corpus_path = Path(args.corpus)
    searcher = None
    if corpus_path.exists():
        records = load_corpus_parquet(str(corpus_path))
        if records:
            searcher = ResolutionIndex.build(records, embedder=embedder)
    if searcher is None:
        print(
            f"No retrieval corpus found at {corpus_path} (or it's empty) — run "
            "`python scripts/build_index.py --brand <brand>` first for real historical evidence. "
            "Continuing with an empty searcher; every message will look out-of-distribution.\n"
        )
        searcher = _EmptySearcher()

    escalation_config = load_escalation_config(args.escalation_config)
    llm_provider = get_llm_provider(args.llm_provider)
    return classifier, searcher, llm_provider, escalation_config


def print_result(result) -> None:
    print("Customer:")
    print(f"{result.customer_message}\n")

    print("Intent:")
    print(f"{result.intent}\n")

    print("Confidence:")
    print(f"{result.intent_confidence:.2f}\n")

    print("Historical evidence:")
    if result.retrieved_evidence:
        for e in result.retrieved_evidence:
            timestamp = e.timestamp.isoformat() if e.timestamp else "unknown date"
            print(f"  - [similarity {e.similarity_score:.2f}, intent={e.intent}, {timestamp}]")
            print(f"    customer: {e.customer_message}")
            print(f"    brand reply: {e.brand_reply}")
    else:
        print("  (none — no historical evidence retrieved)")
    print()

    print("Draft reply:")
    print(f"{result.reply or '(no reply drafted — the agent escalated instead)'}\n")

    print("Decision:")
    print(f"{'AUTO_HANDLE' if result.decision == 'auto_handle' else 'ESCALATE'}\n")

    print("Reason:")
    print(f"{result.reason}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Interactive demo of the Hiver support agent.")
    parser.add_argument("--message", default=None, help="Run once on a single message instead of interactive mode")
    parser.add_argument("--config", default=None)
    parser.add_argument("--intents-config", default=None)
    parser.add_argument("--escalation-config", default=None)
    parser.add_argument("--corpus", default=DEFAULT_CORPUS_PATH)
    parser.add_argument("--embedder", choices=["hashing", "sentence-transformer"], default="hashing")
    parser.add_argument("--llm-provider", default=None)
    args = parser.parse_args()

    classifier, searcher, llm_provider, escalation_config = build_pipeline(args)
    print(f"(LLM provider: {llm_provider.name} [{llm_provider.model}])\n")

    def _run(message: str) -> None:
        result = run_agent(
            message,
            intent_classifier=classifier,
            searcher=searcher,
            llm_provider=llm_provider,
            escalation_config=escalation_config,
        )
        print_result(result)

    if args.message:
        _run(args.message)
        return

    print("Interactive demo — type a customer message and press Enter.")
    print("Type 'quit', 'exit', or press Ctrl+D to stop.\n")
    while True:
        try:
            message = input("> ").strip()
        except EOFError:
            print()
            break
        if not message:
            continue
        if message.lower() in ("quit", "exit"):
            break
        print()
        _run(message)


if __name__ == "__main__":
    main()
