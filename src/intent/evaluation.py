"""Classification evaluation: accuracy, macro/weighted F1, per-intent
precision/recall/F1, and a confusion matrix.

Not part of the file list given for Phase 2, but added because requirement
9 ("Implement evaluation for classification") needs somewhere to live —
this is the natural place given it evaluates the classifiers in this same
package. Every number here comes from `sklearn.metrics` run against actual
`y_true`/`y_pred` arrays passed in by the caller; nothing is estimated,
hardcoded, or fabricated.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)


@dataclass
class PerIntentMetrics:
    intent: str
    precision: float
    recall: float
    f1: float
    support: int

    def as_dict(self) -> dict:
        return {
            "intent": self.intent,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "support": self.support,
        }


@dataclass
class ClassificationReport:
    accuracy: float
    macro_f1: float
    weighted_f1: float
    per_intent: list[PerIntentMetrics]
    confusion_matrix: list[list[int]]
    labels: list[str]
    n_examples: int

    def as_dict(self) -> dict:
        return {
            "accuracy": self.accuracy,
            "macro_f1": self.macro_f1,
            "weighted_f1": self.weighted_f1,
            "per_intent": [p.as_dict() for p in self.per_intent],
            "confusion_matrix": self.confusion_matrix,
            "labels": self.labels,
            "n_examples": self.n_examples,
        }


def evaluate_predictions(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    labels: Optional[Sequence[str]] = None,
) -> ClassificationReport:
    """Compute the full classification report from actual predictions.

    `labels` fixes the row/column order of the confusion matrix and the set
    of intents scored per-intent; if omitted, it defaults to the sorted
    union of labels actually seen in `y_true`/`y_pred` (so an intent with
    zero support in both still gets a defined precision/recall of 0 when
    you pass the full taxonomy explicitly, rather than silently vanishing).
    """
    y_true = list(y_true)
    y_pred = list(y_pred)
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must be the same length.")
    if len(y_true) == 0:
        raise ValueError("Cannot evaluate zero examples.")

    all_labels = list(labels) if labels is not None else sorted(set(y_true) | set(y_pred))

    accuracy = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, labels=all_labels, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, labels=all_labels, average="weighted", zero_division=0))

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=all_labels, zero_division=0
    )
    per_intent = [
        PerIntentMetrics(intent=label, precision=float(p), recall=float(r), f1=float(f), support=int(s))
        for label, p, r, f, s in zip(all_labels, precision, recall, f1, support)
    ]

    cm = confusion_matrix(y_true, y_pred, labels=all_labels).tolist()

    return ClassificationReport(
        accuracy=accuracy,
        macro_f1=macro_f1,
        weighted_f1=weighted_f1,
        per_intent=per_intent,
        confusion_matrix=cm,
        labels=all_labels,
        n_examples=len(y_true),
    )
