"""Human vs. LLM-judge agreement analysis over
`data/golden/judge_validation.csv`.

If `human_score` is missing/blank/`"UNLABELLED"` for every row — the
expected state until someone actually does the labeling — this module
reports status `"pending_human_labels"` with the literal message
`"HUMAN VALIDATION PENDING"` rather than computing agreement from partial
or absent data. Agreement numbers are only ever computed from rows where a
human has actually filled in a score.
"""

from __future__ import annotations

import csv
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from src.evaluation.metrics import is_labeled

DEFAULT_JUDGE_VALIDATION_PATH = "data/golden/judge_validation.csv"
JUDGE_VALIDATION_FIELDS = [
    "example_id",
    "customer_message",
    "agent_reply",
    "evidence",
    "human_score",
    "human_notes",
    "judge_score",
]
UNLABELLED = "UNLABELLED"
HUMAN_VALIDATION_PENDING = "HUMAN VALIDATION PENDING"


@dataclass
class AgreementReport:
    status: str  # "computed" | "pending_human_labels"
    n_total_rows: int
    n_labeled: int
    exact_agreement: Optional[float] = None
    mean_absolute_difference: Optional[float] = None
    pearson_correlation: Optional[float] = None
    weighted_kappa: Optional[float] = None
    message: Optional[str] = None

    def as_dict(self) -> dict:
        return {
            "status": self.status,
            "n_total_rows": self.n_total_rows,
            "n_labeled": self.n_labeled,
            "exact_agreement": self.exact_agreement,
            "mean_absolute_difference": self.mean_absolute_difference,
            "pearson_correlation": self.pearson_correlation,
            "weighted_kappa": self.weighted_kappa,
            "message": self.message,
        }


def load_judge_validation_rows(path: str = DEFAULT_JUDGE_VALIDATION_PATH) -> list[dict]:
    p = Path(path)
    if not p.exists():
        return []
    with open(p, "r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def create_judge_validation_template(
    predictions: list[dict],
    path: str = DEFAULT_JUDGE_VALIDATION_PATH,
    n: int = 30,
    seed: int = 42,
) -> int:
    """Sample `n` predictions (each expected to have `example_id`,
    `customer_message`, `reply`, `retrieved_evidence`, and a nested
    `judge_score.overall`) into a human-labeling template.
    `human_score`/`human_notes` are written as the literal placeholder
    `"UNLABELLED"` — never guessed, never pre-filled from the judge's own
    score. Returns the number of rows written.
    """
    eligible = [p for p in predictions if p.get("reply") is not None and p.get("judge_score") is not None]
    rng = random.Random(seed)
    pool = list(eligible)
    rng.shuffle(pool)
    sample = pool[: min(n, len(pool))]

    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=JUDGE_VALIDATION_FIELDS)
        writer.writeheader()
        for p in sample:
            evidence_summary = "; ".join(
                f"{e.get('resolution_id')}@{e.get('similarity_score'):.2f}" for e in (p.get("retrieved_evidence") or [])
            )
            writer.writerow(
                {
                    "example_id": p.get("example_id"),
                    "customer_message": p.get("customer_message"),
                    "agent_reply": p.get("reply") or "",
                    "evidence": evidence_summary,
                    "human_score": UNLABELLED,
                    "human_notes": UNLABELLED,
                    "judge_score": p["judge_score"].get("overall"),
                }
            )
    return len(sample)


def compute_agreement(rows: list[dict]) -> AgreementReport:
    if not rows:
        return AgreementReport(
            status="pending_human_labels",
            n_total_rows=0,
            n_labeled=0,
            message=f"{HUMAN_VALIDATION_PENDING} — data/golden/judge_validation.csv does not exist yet.",
        )

    labeled = [r for r in rows if is_labeled(r.get("human_score"))]
    if not labeled:
        return AgreementReport(
            status="pending_human_labels",
            n_total_rows=len(rows),
            n_labeled=0,
            message=f"{HUMAN_VALIDATION_PENDING} — no human_score values have been filled in yet.",
        )

    human_scores: list[float] = []
    judge_scores: list[float] = []
    for r in labeled:
        try:
            human_scores.append(float(r["human_score"]))
            judge_scores.append(float(r["judge_score"]))
        except (TypeError, ValueError, KeyError):
            continue

    if not human_scores:
        return AgreementReport(
            status="pending_human_labels",
            n_total_rows=len(rows),
            n_labeled=0,
            message=f"{HUMAN_VALIDATION_PENDING} — human_score values present but none were parseable numbers.",
        )

    n = len(human_scores)
    exact = sum(1 for h, j in zip(human_scores, judge_scores) if round(h) == round(j)) / n
    mad = sum(abs(h - j) for h, j in zip(human_scores, judge_scores)) / n

    pearson = None
    if len(set(human_scores)) > 1 and len(set(judge_scores)) > 1:
        pearson = float(np.corrcoef(human_scores, judge_scores)[0, 1])

    kappa = None
    try:
        from sklearn.metrics import cohen_kappa_score

        human_int = [round(h) for h in human_scores]
        judge_int = [round(j) for j in judge_scores]
        if len(set(human_int)) > 1 or len(set(judge_int)) > 1:
            kappa = float(cohen_kappa_score(human_int, judge_int, weights="quadratic"))
    except Exception:
        kappa = None

    return AgreementReport(
        status="computed",
        n_total_rows=len(rows),
        n_labeled=n,
        exact_agreement=exact,
        mean_absolute_difference=mad,
        pearson_correlation=pearson,
        weighted_kappa=kappa,
        message=(
            None
            if n >= 30
            else f"Only {n} human-labeled examples so far (target: 30) — treat this as preliminary."
        ),
    )
