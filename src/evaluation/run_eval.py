#!/usr/bin/env python
"""Run the full Phase 5 evaluation.

    python -m src.evaluation.run_eval

Runs three systems — a majority-class baseline, a TF-IDF baseline, and the
"final" (embedding classifier + semantic retrieval) system — through the
same `run_agent` pipeline over `data/golden/golden_set.csv`, scores every
final-system reply with the LLM judge, and writes every required artifact
under `artifacts/evaluation/`.

Before anything else, this runs automated leakage checks and refuses to
proceed if a golden conversation overlaps the TRAIN split or the retrieval
corpus. See `run_leakage_checks` below.

Every metric that needs a gold label degrades to an explicit "pending"
status rather than a fabricated number when that label is still the
literal placeholder "UNLABELLED" — which is the current, honest state of
`data/golden/golden_set.csv` in this sandbox (see docs/GOLDEN_SET.md for
why: no real dataset, so no real hand-labeling was possible here).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.agent import run_agent
from src.agent.escalation import EscalationConfig, load_escalation_config
from src.agent.llm import LLMProvider, get_llm_provider
from src.config import Config, load_config
from src.data.conversations import build_conversations, extract_resolution_pairs
from src.data.normalize import load_raw_csv, normalize_dataframe
from src.data.split import split_conversations
from src.evaluation.failure_analysis import analyze_failures, save_failure_analysis
from src.evaluation.judge import JudgeScore, judge_reply
from src.evaluation.judge_agreement import compute_agreement, create_judge_validation_template, load_judge_validation_rows
from src.evaluation.metrics import (
    RatioMetric,
    SafeAutomationRow,
    automation_coverage,
    escalation_precision,
    escalation_recall,
    false_auto_handle_rate,
    intent_classification_report,
    is_labeled,
    parse_bool_label,
    parse_reply_quality_label,
    safe_automation_coverage,
    safe_automation_coverage_over_labeled_subset,
)
from src.evaluation.retrieval_metrics import MIN_RECORDS_FOR_MEANINGFUL_EVAL, evaluate_retrieval_for_report
from src.intent.baseline_majority import MajorityClassifier
from src.intent.baseline_tfidf import TfidfLogisticClassifier
from src.intent.classifier import IntentTaxonomy
from src.intent.embedding import EmbeddingNearestCentroidClassifier, HashingEmbedder
from src.retrieval.index import ResolutionRecord, build_resolution_corpus, load_corpus_parquet, save_corpus_parquet
from src.retrieval.search import ResolutionIndex, TfidfRetrievalBaseline

GOLDEN_SET_PATH = "data/golden/golden_set.csv"
DEFAULT_OUTPUT_DIR = "artifacts/evaluation"


# --- Leakage checks -----------------------------------------------------


@dataclass
class LeakageCheckResult:
    name: str
    passed: Optional[bool]  # None = not machine-checkable (process attestation only)
    detail: str

    def as_dict(self) -> dict:
        return {"name": self.name, "passed": self.passed, "detail": self.detail}


def run_leakage_checks(
    config: Config,
    golden_rows: list[dict],
    corpus_records: list[ResolutionRecord],
) -> list[LeakageCheckResult]:
    """The five checks the assignment calls out, in order. The first two
    are hard, automated, abort-on-failure checks. The third
    (`no_evidence_is_its_own_gold_example`) is checked per-example during
    the evaluation loop itself (see `main`) and appended afterward. The
    last two are fundamentally process guarantees — see each result's
    `detail` for exactly what is and isn't machine-verifiable here.
    """
    df = load_raw_csv(config.paths.raw_csv)
    tweets, _ = normalize_dataframe(df)
    conversations = build_conversations(tweets)
    conversation_ids = [c.conversation_id for c in conversations]
    assignment = split_conversations(
        conversation_ids, train=config.split.train, dev=config.split.dev, test=config.split.test, seed=config.split.seed
    )
    train_ids = set(assignment.train)

    golden_conv_ids = {r["conversation_id"] for r in golden_rows}
    corpus_conv_ids = {r.conversation_id for r in corpus_records}

    overlap_train = golden_conv_ids & train_ids
    overlap_corpus = golden_conv_ids & corpus_conv_ids

    return [
        LeakageCheckResult(
            "golden_conversations_not_in_train_split",
            passed=(len(overlap_train) == 0),
            detail=(
                f"{len(overlap_train)} golden conversation(s) overlap the TRAIN split: {sorted(overlap_train)[:10]}"
                if overlap_train
                else "No golden conversation ids appear in the TRAIN split."
            ),
        ),
        LeakageCheckResult(
            "golden_conversations_not_in_retrieval_corpus",
            passed=(len(overlap_corpus) == 0),
            detail=(
                f"{len(overlap_corpus)} golden conversation(s) appear in the retrieval corpus: {sorted(overlap_corpus)[:10]}"
                if overlap_corpus
                else "No golden conversation ids appear in the retrieval corpus (which is itself TRAIN-only by construction)."
            ),
        ),
        LeakageCheckResult(
            "thresholds_not_tuned_on_golden_set",
            passed=None,
            detail=(
                "PROCESS CHECK (not machine-verifiable): confirm configs/escalation.yaml's "
                "thresholds were chosen using dev-split analysis only, never by looking at "
                "golden-set outcomes. See docs/EVALUATION.md."
            ),
        ),
        LeakageCheckResult(
            "golden_labels_not_used_in_fitting",
            passed=True,
            detail=(
                "All three intent classifiers (majority/tfidf/final) are fit only on "
                "configs/intents.yaml's own worked examples, never on golden_set.csv content — "
                "true by construction; see build_systems() in this file."
            ),
        ),
    ]


def _check_no_self_leakage_in_evidence(predictions: list[dict]) -> LeakageCheckResult:
    """Per-example defensive check: no retrieved evidence item's
    conversation_id equals the query example's own conversation_id (i.e.
    an example's own historical resolution was never handed back to it as
    'evidence'). This should be structurally impossible given the other
    two checks, but is verified directly rather than assumed."""
    violations = []
    for pred in predictions:
        for ev in pred.get("retrieved_evidence") or []:
            if ev.get("conversation_id") == pred.get("conversation_id"):
                violations.append(pred.get("example_id"))
    return LeakageCheckResult(
        "gold_example_never_returned_as_its_own_evidence",
        passed=(len(violations) == 0),
        detail=(
            f"{len(violations)} example(s) had their own conversation_id show up in their "
            f"retrieved evidence: {violations[:10]}"
            if violations
            else "No golden example ever retrieved its own conversation as evidence."
        ),
    )


# --- Golden set / helpers ------------------------------------------------


def load_golden_set(path: str) -> list[dict]:
    p = Path(path)
    if not p.exists():
        print(f"No golden set found at {path}.")
        print("Run `python scripts/create_golden_template.py` first, then label it — see docs/GOLDEN_SET.md.")
        sys.exit(1)
    with open(p, "r", newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        print(f"{path} exists but has zero rows — nothing to evaluate.")
        sys.exit(1)
    return rows


def bootstrap_training_data(taxonomy: IntentTaxonomy) -> tuple[list[str], list[str]]:
    """Training data for all three intent classifiers: the taxonomy's own
    worked examples, and NOTHING from the golden set — this is what makes
    'evaluation labels are not used during model fitting' true by
    construction rather than by discipline alone."""
    texts, labels = [], []
    for intent_id in taxonomy.labels:
        definition = taxonomy.get(intent_id)
        for example in definition.examples:
            texts.append(example)
            labels.append(intent_id)
    return texts, labels


def build_reference_lookup(config: Config) -> dict[str, str]:
    """conversation_id -> historical brand_reply, built from the FULL
    dataset (all splits) — used ONLY to give the LLM judge a gold/reference
    resolution for judging purposes when one exists for a golden example's
    conversation. This is deliberately separate from the retrieval corpus
    (train-only) and is never used as retrieval evidence or for training
    anything; it only ever appears inside a judge prompt, at evaluation
    time, for a single example being judged.
    """
    df = load_raw_csv(config.paths.raw_csv)
    tweets, _ = normalize_dataframe(df)
    conversations = build_conversations(tweets)
    pairs = extract_resolution_pairs(conversations)
    lookup: dict[str, str] = {}
    for p in pairs:
        lookup.setdefault(p.conversation_id, p.brand_reply)
    return lookup


def build_systems(
    taxonomy: IntentTaxonomy,
    corpus_records: list[ResolutionRecord],
    embedder,
    cache_dir: str,
) -> dict[str, dict]:
    texts, labels = bootstrap_training_data(taxonomy)

    majority = MajorityClassifier(taxonomy).fit(texts, labels)
    tfidf = TfidfLogisticClassifier(taxonomy).fit(texts, labels)
    final_classifier = EmbeddingNearestCentroidClassifier(taxonomy, embedder=embedder).fit(texts, labels)

    tfidf_searcher = TfidfRetrievalBaseline(corpus_records) if corpus_records else None
    semantic_index = (
        ResolutionIndex.build(corpus_records, embedder=embedder, cache_dir=cache_dir) if corpus_records else None
    )

    return {
        "majority_baseline": {"classifier": majority, "searcher": tfidf_searcher, "run_judge": False},
        "tfidf_baseline": {"classifier": tfidf, "searcher": tfidf_searcher, "run_judge": False},
        "final_system": {"classifier": final_classifier, "searcher": semantic_index, "run_judge": True},
    }


class _EmptySearcher:
    def retrieve(self, query, intent=None, top_k=5):
        return []


def _prediction_dict(system_name: str, row: dict, result, judge_score: Optional[JudgeScore]) -> dict:
    gold_intent = row.get("gold_intent")
    return {
        "system": system_name,
        "example_id": row.get("example_id"),
        "conversation_id": row.get("conversation_id"),
        "customer_message": result.customer_message,
        "gold_intent": gold_intent if is_labeled(gold_intent) else None,
        "gold_should_escalate": row.get("gold_should_escalate") if is_labeled(row.get("gold_should_escalate")) else None,
        "gold_reply_quality": row.get("gold_reply_quality") if is_labeled(row.get("gold_reply_quality")) else None,
        "predicted_intent": result.intent,
        "confidence": result.intent_confidence,
        "retrieved_evidence": [e.as_dict() for e in result.retrieved_evidence],
        "reply": result.reply,
        "grounding_score": result.signals.get("grounding_score"),
        "decision": result.decision,
        "reason": result.reason,
        "signals": result.signals,
        "judge_score": judge_score.as_dict() if judge_score else None,
    }


# --- Reporting helpers -----------------------------------------------------


def _intent_report_or_pending(golden_rows: list[dict], predictions_by_id: dict, labels: list[str]) -> dict:
    labeled = [(r["example_id"], r["gold_intent"]) for r in golden_rows if is_labeled(r.get("gold_intent"))]
    if not labeled:
        return {"status": "pending_human_labels", "message": "No gold_intent labels filled in yet.", "n_labeled": 0}
    y_true = [gold for _, gold in labeled]
    y_pred = [predictions_by_id[eid]["predicted_intent"] for eid, _ in labeled]
    report = intent_classification_report(y_true, y_pred, labels=labels)
    return {"status": "computed", "n_labeled": len(labeled), **report.as_dict()}


def _retrieval_report_or_pending(searcher, records: list[ResolutionRecord], seed: int) -> dict:
    if searcher is None or len(records) < MIN_RECORDS_FOR_MEANINGFUL_EVAL:
        return {
            "status": "insufficient_data",
            "message": (
                f"Only {len(records)} record(s) in the retrieval corpus — need at least "
                f"{MIN_RECORDS_FOR_MEANINGFUL_EVAL} for a meaningful proxy Recall@5/MRR. "
                "Expected against the tiny synthetic fixture; re-run against the real dataset."
            ),
        }
    report = evaluate_retrieval_for_report(searcher, records, seed=seed)
    return {"status": "computed", **report.as_dict()}


def _escalation_metrics(golden_rows: list[dict], predictions_by_id: dict) -> dict:
    gold_escalate = [parse_bool_label(r.get("gold_should_escalate")) for r in golden_rows]
    predicted_escalate = [predictions_by_id[r["example_id"]]["decision"] == "escalate" for r in golden_rows]

    safe_rows = []
    for r in golden_rows:
        pred = predictions_by_id[r["example_id"]]
        _, acceptable = parse_reply_quality_label(r.get("gold_reply_quality"))
        safe_rows.append(
            SafeAutomationRow(
                example_id=r["example_id"],
                predicted_decision=pred["decision"],
                predicted_intent=pred["predicted_intent"],
                gold_intent=r.get("gold_intent") if is_labeled(r.get("gold_intent")) else None,
                gold_should_escalate=parse_bool_label(r.get("gold_should_escalate")),
                reply_quality_acceptable=acceptable,
            )
        )

    return {
        "escalation_precision": escalation_precision(gold_escalate, predicted_escalate).as_dict(),
        "escalation_recall": escalation_recall(gold_escalate, predicted_escalate).as_dict(),
        "false_auto_handle_rate": false_auto_handle_rate(gold_escalate, predicted_escalate).as_dict(),
        "automation_coverage": automation_coverage(predicted_escalate).as_dict(),
        "safe_automation_coverage_full_denominator": safe_automation_coverage(
            safe_rows, total_golden_examples=len(golden_rows)
        ).as_dict(),
        "safe_automation_coverage_labeled_subset": safe_automation_coverage_over_labeled_subset(safe_rows).as_dict(),
    }


def _judge_aggregate(predictions: list[dict]) -> dict:
    scored = [p["judge_score"] for p in predictions if p.get("judge_score") and not p["judge_score"].get("parse_error")]
    if not scored:
        return {"status": "no_scored_replies", "n_scored": 0}
    axes = ["correctness", "groundedness", "relevance", "helpfulness", "tone", "hallucination_safety", "escalation_appropriateness", "overall"]
    means = {axis: round(sum(s[axis] for s in scored) / len(scored), 3) for axis in axes}
    return {"status": "computed", "n_scored": len(scored), "n_total": len(predictions), "mean_scores": means}


def save_confusion_matrix_png(intent_report: dict, path: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    if intent_report.get("status") != "computed":
        ax.text(
            0.5, 0.5,
            "PENDING\n\nNo gold_intent labels have been filled in yet in\n"
            "data/golden/golden_set.csv, so there is nothing to plot.\n"
            "See docs/GOLDEN_SET.md.",
            ha="center", va="center", wrap=True, fontsize=12,
        )
        ax.set_axis_off()
    else:
        labels = intent_report["labels"]
        cm = np.array(intent_report["confusion_matrix"])
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels)
        ax.set_xlabel("Predicted intent")
        ax.set_ylabel("Gold intent")
        ax.set_title(f"Final system — intent confusion matrix (n={intent_report['n_labeled']})")
        for i in range(len(labels)):
            for j in range(len(labels)):
                ax.text(j, i, str(cm[i][j]), ha="center", va="center", color="black", fontsize=8)
        fig.colorbar(im, ax=ax)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def write_results_csv(all_results: dict, path: str) -> None:
    rows = []
    for system_name, result in all_results.items():
        for section in ("escalation",):
            for metric_name, metric in result[section].items():
                rows.append(
                    {
                        "system": system_name,
                        "metric": metric_name,
                        "numerator": metric.get("numerator"),
                        "denominator": metric.get("denominator"),
                        "value": metric.get("value"),
                        "formatted": metric.get("formatted"),
                    }
                )
        intent = result["intent"]
        if intent.get("status") == "computed":
            rows.append(
                {
                    "system": system_name, "metric": "intent_accuracy", "numerator": None,
                    "denominator": intent["n_labeled"], "value": intent["accuracy"],
                    "formatted": f"{intent['accuracy']*100:.0f}% intent accuracy ({intent['n_labeled']} labeled)",
                }
            )
            rows.append(
                {
                    "system": system_name, "metric": "intent_macro_f1", "numerator": None,
                    "denominator": intent["n_labeled"], "value": intent["macro_f1"], "formatted": None,
                }
            )
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["system", "metric", "numerator", "denominator", "value", "formatted"])
        writer.writeheader()
        writer.writerows(rows)


def build_headline_report(all_results: dict, leakage_results: list[LeakageCheckResult], agreement, failure_categories: dict) -> dict:
    final = all_results["final_system"]
    labeled_cov = final["escalation"]["safe_automation_coverage_labeled_subset"]
    full_cov = final["escalation"]["safe_automation_coverage_full_denominator"]

    headline = labeled_cov if labeled_cov["denominator"] > 0 else full_cov

    return {
        "headline_metric": "safe automation coverage",
        "headline_value": headline,
        "headline_formatted": headline["formatted"],
        "what_could_be_misleading_about_this_number": [
            "The denominator above is the number of LABELED golden examples, not the full "
            "golden set size, while human labeling is incomplete — see 'full_denominator_value' "
            "for the (currently much smaller-looking) version divided by the entire golden set.",
            "In this sandbox, data/golden/golden_set.csv ships fully UNLABELLED (no real "
            "hand-labeling was possible — see docs/GOLDEN_SET.md), so n_labeled is 0 and this "
            "number cannot be computed honestly at all yet; it will read as 'N/A (0/0)' below.",
            "Intent classifiers here are bootstrapped on configs/intents.yaml's illustrative "
            "worked examples, not a real labeled training set — intent predictions (and "
            "therefore safe automation coverage) inherit that weakness.",
            "The LLM judge and the reply-drafting LLM currently use the SAME provider "
            "(mock, or the same OpenAI configuration) — this is a self-grading setup that can "
            "inflate agreement between 'the model likes its own output' and 'the output is "
            "actually good'. See docs/EVALUATION.md.",
            "Retrieval Recall@k/MRR is a self-retrieval proxy (paraphrase-of-a-known-message "
            "recall), not human relevance judgment — see docs/EVALUATION.md and "
            "src/retrieval/search.py.",
        ],
        "full_denominator_value": full_cov,
        "leakage_checks": [r.as_dict() for r in leakage_results],
        "human_judge_agreement": agreement.as_dict(),
        "failure_analysis": {
            system: [c.as_dict(denominator) for c in cats]
            for system, (cats, denominator) in failure_categories.items()
        },
    }


# --- Main ------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the full Phase 5 evaluation.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--intents-config", default=None)
    parser.add_argument("--escalation-config", default=None)
    parser.add_argument("--golden-set", default=GOLDEN_SET_PATH)
    parser.add_argument("--brand", default=None)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--llm-provider", default=None, help="Overrides LLM_PROVIDER for both reply drafting and judging")
    parser.add_argument("--judge-validation-size", type=int, default=30)
    args = parser.parse_args()

    config = load_config(args.config).resolve_paths()
    taxonomy = IntentTaxonomy.from_yaml(args.intents_config)
    escalation_config = load_escalation_config(args.escalation_config)
    llm_provider = get_llm_provider(args.llm_provider)
    print(f"LLM provider: {llm_provider.name} ({llm_provider.model})")
    if llm_provider.name == "mock":
        print("NOTE: reply drafting AND judging both use the mock provider here — see "
              "docs/EVALUATION.md's self-grading-bias caveat.")

    golden_rows = load_golden_set(args.golden_set)
    print(f"Loaded {len(golden_rows)} golden examples from {args.golden_set}")
    n_labeled_intent = sum(1 for r in golden_rows if is_labeled(r.get("gold_intent")))
    n_labeled_escalate = sum(1 for r in golden_rows if is_labeled(r.get("gold_should_escalate")))
    n_labeled_quality = sum(1 for r in golden_rows if is_labeled(r.get("gold_reply_quality")))
    print(f"  labeled: gold_intent={n_labeled_intent} gold_should_escalate={n_labeled_escalate} gold_reply_quality={n_labeled_quality}")

    print("Building/loading the TRAIN-split-only retrieval corpus ...")
    corpus_path = Path("artifacts") / "resolutions.parquet"
    if corpus_path.exists():
        corpus_records = load_corpus_parquet(str(corpus_path))
    elif args.brand:
        corpus_records = build_resolution_corpus(config, args.brand, taxonomy=taxonomy)
        save_corpus_parquet(corpus_records, str(corpus_path))
    else:
        corpus_records = []
    print(f"  {len(corpus_records)} historical resolution record(s)")

    print("Running automated leakage checks ...")
    leakage_results = run_leakage_checks(config, golden_rows, corpus_records)
    for check in leakage_results:
        status = "PASS" if check.passed else ("SKIP (process check)" if check.passed is None else "FAIL")
        print(f"  [{status}] {check.name}: {check.detail}")
    hard_failures = [c for c in leakage_results if c.passed is False]
    if hard_failures:
        print("\nABORTING: one or more automated leakage checks failed. Fix the data split before evaluating.")
        sys.exit(1)

    embedder = HashingEmbedder(dimension=256)
    output_dir = Path(args.output_dir)
    systems = build_systems(taxonomy, corpus_records, embedder, cache_dir=str(output_dir / "embedding_cache"))
    reference_lookup = build_reference_lookup(config) if corpus_records else {}

    all_predictions: list[dict] = []
    all_results: dict[str, dict] = {}

    for system_name, system in systems.items():
        print(f"\nRunning system: {system_name} ...")
        searcher = system["searcher"] or _EmptySearcher()
        predictions_by_id: dict[str, dict] = {}

        for row in golden_rows:
            result = run_agent(
                row["customer_message"],
                intent_classifier=system["classifier"],
                searcher=searcher,
                llm_provider=llm_provider,
                escalation_config=escalation_config,
            )

            judge_score = None
            if system["run_judge"]:
                gold_reference = reference_lookup.get(row["conversation_id"])
                judge_score = judge_reply(
                    customer_message=result.customer_message,
                    agent_reply=result.reply,
                    evidence=[e.as_dict() for e in result.retrieved_evidence],
                    intent=result.intent,
                    decision=result.decision,
                    reason=result.reason,
                    llm_provider=llm_provider,
                    gold_reference=gold_reference,
                    grounding_score=result.signals.get("grounding_score"),
                    confidence=result.intent_confidence,
                )

            pred = _prediction_dict(system_name, row, result, judge_score)
            predictions_by_id[row["example_id"]] = pred
            all_predictions.append(pred)

        intent_report = _intent_report_or_pending(golden_rows, predictions_by_id, taxonomy.labels)
        retrieval_report = _retrieval_report_or_pending(system["searcher"], corpus_records, seed=config.split.seed)
        escalation_metrics = _escalation_metrics(golden_rows, predictions_by_id)
        judge_summary = _judge_aggregate(list(predictions_by_id.values())) if system["run_judge"] else {"status": "not_run_for_this_system"}

        all_results[system_name] = {
            "intent": intent_report,
            "retrieval": retrieval_report,
            "escalation": escalation_metrics,
            "judge": judge_summary,
        }
        print(f"  intent: {intent_report.get('status')}  retrieval: {retrieval_report.get('status')}  "
              f"safe_automation_coverage(labeled): {escalation_metrics['safe_automation_coverage_labeled_subset']['formatted']}")

    self_leakage_check = _check_no_self_leakage_in_evidence(all_predictions)
    leakage_results.append(self_leakage_check)
    print(f"  [{'PASS' if self_leakage_check.passed else 'FAIL'}] {self_leakage_check.name}: {self_leakage_check.detail}")

    failure_categories = {
        system_name: (
            analyze_failures([p for p in all_predictions if p["system"] == system_name]),
            len([p for p in all_predictions if p["system"] == system_name]),
        )
        for system_name in systems
    }

    final_predictions_for_failure_analysis = [p for p in all_predictions if p["system"] == "final_system"]
    failure_analysis_report = save_failure_analysis(
        final_predictions_for_failure_analysis, path=str(output_dir / "failure_analysis.json")
    )
    print(f"\nTop failure modes (final_system): {[m['name'] for m in failure_analysis_report['top_failure_modes']]}")
    print(f"Wrote {output_dir}/failure_analysis.json")

    validation_rows = load_judge_validation_rows()
    if not validation_rows:
        final_predictions = [p for p in all_predictions if p["system"] == "final_system"]
        n_written = create_judge_validation_template(
            final_predictions, n=args.judge_validation_size, seed=config.split.seed
        )
        print(f"\nWrote {n_written} rows to data/golden/judge_validation.csv for human labeling (target: {args.judge_validation_size}).")
        validation_rows = load_judge_validation_rows()
    agreement = compute_agreement(validation_rows)
    print(f"Human/judge agreement: {agreement.status} ({agreement.message or 'computed'})")

    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "predictions.jsonl", "w", encoding="utf-8") as fh:
        for pred in all_predictions:
            fh.write(json.dumps(pred, default=str) + "\n")

    with open(output_dir / "results.json", "w", encoding="utf-8") as fh:
        json.dump(all_results, fh, indent=2, default=str)

    write_results_csv(all_results, str(output_dir / "results.csv"))

    save_confusion_matrix_png(all_results["final_system"]["intent"], str(output_dir / "confusion_matrix.png"))

    report = build_headline_report(all_results, leakage_results, agreement, failure_categories)
    report["golden_set_meta"] = {
        "n_total": len(golden_rows),
        "n_labeled_gold_intent": n_labeled_intent,
        "n_labeled_gold_should_escalate": n_labeled_escalate,
        "n_labeled_gold_reply_quality": n_labeled_quality,
    }
    with open(output_dir / "report.json", "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, default=str)

    print(f"\n{report['headline_formatted']}")
    print(f"\nWrote outputs to {output_dir}/: predictions.jsonl, results.json, results.csv, confusion_matrix.png, report.json")


if __name__ == "__main__":
    main()
