#!/usr/bin/env python
"""Human-readable case inspection over `artifacts/evaluation/predictions.jsonl`.

    python -m src.evaluation.inspect_cases --n 20
    python -m src.evaluation.inspect_cases --n 10 --filter-decision escalate
    python -m src.evaluation.inspect_cases --system tfidf_baseline

Prints each case as CUSTOMER / INTENT / CONFIDENCE / EVIDENCE / REPLY /
DECISION / ESCALATION REASON — the same information
`artifacts/evaluation/predictions.jsonl` carries per example, just
formatted for a human to skim quickly instead of parsing JSON lines by
hand. Reads only what `run_eval.py` already wrote; makes no predictions
of its own.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

DEFAULT_PREDICTIONS_PATH = "artifacts/evaluation/predictions.jsonl"


def load_predictions(path: str, system: Optional[str]) -> list[dict]:
    predictions = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if system is None or row.get("system") == system:
                predictions.append(row)
    return predictions


def format_case(pred: dict) -> str:
    lines = [f"=== {pred.get('example_id')} (system={pred.get('system')}) ==="]

    lines.append("CUSTOMER:")
    lines.append(f"  {pred.get('customer_message')}")

    lines.append("INTENT:")
    gold_intent = pred.get("gold_intent")
    intent_line = f"  {pred.get('predicted_intent')}"
    if gold_intent:
        intent_line += f"  (gold: {gold_intent})"
    lines.append(intent_line)

    lines.append("CONFIDENCE:")
    confidence = pred.get("confidence")
    lines.append(f"  {confidence:.2f}" if isinstance(confidence, (int, float)) else "  N/A")

    lines.append("EVIDENCE:")
    evidence = pred.get("retrieved_evidence") or []
    if evidence:
        for ev in evidence:
            score = ev.get("similarity_score")
            score_str = f"{score:.2f}" if isinstance(score, (int, float)) else "N/A"
            lines.append(f"  - [{score_str}] ({ev.get('intent')}) {ev.get('customer_message')!r} -> {ev.get('brand_reply')!r}")
    else:
        lines.append("  (none)")

    lines.append("REPLY:")
    lines.append(f"  {pred.get('reply') or '(no reply drafted — agent escalated)'}")

    lines.append("DECISION:")
    decision = pred.get("decision")
    lines.append(f"  {'AUTO_HANDLE' if decision == 'auto_handle' else 'ESCALATE'}")

    lines.append("ESCALATION REASON:")
    lines.append(f"  {pred.get('reason')}")

    judge_score = pred.get("judge_score")
    if judge_score:
        lines.append("JUDGE SCORES:")
        lines.append(
            "  " + ", ".join(f"{k}={v}" for k, v in judge_score.items() if k not in ("rationale", "parse_error"))
        )

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect individual evaluation cases.")
    parser.add_argument("--n", type=int, default=20, help="Number of cases to print")
    parser.add_argument("--predictions", default=DEFAULT_PREDICTIONS_PATH)
    parser.add_argument("--system", default="final_system", help="Which system's predictions to show ('all' for every system)")
    parser.add_argument("--filter-decision", default=None, choices=["auto_handle", "escalate"])
    args = parser.parse_args()

    path = Path(args.predictions)
    if not path.exists():
        print(f"No predictions found at {path}. Run `python -m src.evaluation.run_eval` first.")
        sys.exit(1)

    system_filter = None if args.system == "all" else args.system
    predictions = load_predictions(str(path), system_filter)
    if args.filter_decision:
        predictions = [p for p in predictions if p.get("decision") == args.filter_decision]

    if not predictions:
        print("No matching cases found.")
        return

    for pred in predictions[: args.n]:
        print(format_case(pred))
        print()

    print(f"Shown {min(args.n, len(predictions))} of {len(predictions)} matching case(s).")


if __name__ == "__main__":
    main()
