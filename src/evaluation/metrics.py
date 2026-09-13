"""Non-retrieval evaluation metrics: intent classification (a thin wrapper
around `src.intent.evaluation`, reused rather than reimplemented), and
escalation policy metrics — including the headline "safe automation
coverage" metric.

Every ratio-shaped metric here is a `RatioMetric`: it always carries its
numerator and denominator, and formats itself as e.g. "62% safe automation
coverage (124/200)" — never a bare percentage on its own. Every metric that
depends on a gold label degrades gracefully to `denominator == 0` /
`value is None` when that label is still the literal placeholder
`"UNLABELLED"`, rather than silently treating a missing label as a wrong
answer (which would make the metric look worse than reality) or as a right
one (which would fabricate a result).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

from src.intent.evaluation import ClassificationReport, evaluate_predictions

UNLABELLED = "UNLABELLED"


def is_labeled(value: object) -> bool:
    """A field counts as labeled iff it's a non-empty string other than the
    literal placeholder used throughout the golden-set template."""
    if value is None:
        return False
    text = str(value).strip()
    return bool(text) and text.upper() != UNLABELLED


def parse_bool_label(value: object) -> Optional[bool]:
    """Parse a gold_should_escalate-style cell into True/False/None (None
    meaning "not labeled yet", never a guessed default)."""
    if not is_labeled(value):
        return None
    text = str(value).strip().lower()
    if text in ("true", "yes", "1", "escalate", "should_escalate"):
        return True
    if text in ("false", "no", "0", "auto_handle", "auto-handle"):
        return False
    return None


def parse_reply_quality_label(value: object, acceptable_threshold: float = 4.0) -> tuple[Optional[float], Optional[bool]]:
    """Parse a gold_reply_quality cell (expected to be a 1-5 score) into
    `(score, is_acceptable)`, both `None` if unlabeled or unparseable."""
    if not is_labeled(value):
        return None, None
    try:
        score = float(value)
    except (TypeError, ValueError):
        return None, None
    return score, score >= acceptable_threshold


@dataclass
class RatioMetric:
    """A ratio that always carries its numerator/denominator, so it can
    never be reported, printed, or serialized as a bare, context-free
    percentage."""

    name: str
    numerator: int
    denominator: int
    note: Optional[str] = None

    @property
    def value(self) -> Optional[float]:
        return (self.numerator / self.denominator) if self.denominator > 0 else None

    def formatted(self) -> str:
        if self.denominator == 0:
            return f"{self.name}: N/A (0/0{' — ' + self.note if self.note else ''})"
        pct = self.value * 100
        return f"{pct:.0f}% {self.name} ({self.numerator}/{self.denominator})"

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "numerator": self.numerator,
            "denominator": self.denominator,
            "value": self.value,
            "formatted": self.formatted(),
            "note": self.note,
        }


def intent_classification_report(
    y_true: Sequence[str], y_pred: Sequence[str], labels: Optional[Sequence[str]] = None
) -> ClassificationReport:
    """Reuses `src.intent.evaluation.evaluate_predictions` directly —
    Phase 5 composes Phase 2's intent metrics rather than reimplementing
    them."""
    return evaluate_predictions(y_true, y_pred, labels=labels)


# --- Escalation metrics ------------------------------------------------------


def _labeled_pairs(gold: Sequence[Optional[bool]], predicted: Sequence[bool]) -> list[tuple[bool, bool]]:
    return [(g, p) for g, p in zip(gold, predicted) if g is not None]


def escalation_precision(gold_should_escalate: Sequence[Optional[bool]], predicted_escalate: Sequence[bool]) -> RatioMetric:
    """Of everything the system escalated, how much SHOULD have been
    escalated per gold. Only defined over examples with a labeled gold
    decision."""
    pairs = _labeled_pairs(gold_should_escalate, predicted_escalate)
    predicted_pos = [(g, p) for g, p in pairs if p]
    tp = sum(1 for g, _ in predicted_pos if g)
    return RatioMetric(
        "escalation precision", tp, len(predicted_pos),
        note="of labeled examples the system escalated, how many gold says should have escalated",
    )


def escalation_recall(gold_should_escalate: Sequence[Optional[bool]], predicted_escalate: Sequence[bool]) -> RatioMetric:
    """Of everything gold says SHOULD have escalated, how much the system
    actually escalated."""
    pairs = _labeled_pairs(gold_should_escalate, predicted_escalate)
    gold_pos = [(g, p) for g, p in pairs if g]
    tp = sum(1 for _, p in gold_pos if p)
    return RatioMetric(
        "escalation recall", tp, len(gold_pos),
        note="of labeled examples gold says should have escalated, how many the system actually escalated",
    )


def false_auto_handle_rate(gold_should_escalate: Sequence[Optional[bool]], predicted_escalate: Sequence[bool]) -> RatioMetric:
    """Of everything the system auto-handled, how much gold says SHOULD
    have been escalated instead — the single most safety-relevant
    escalation metric, since this is the failure mode that actually reaches
    a customer unreviewed."""
    pairs = _labeled_pairs(gold_should_escalate, predicted_escalate)
    auto_handled = [(g, p) for g, p in pairs if not p]
    bad = sum(1 for g, _ in auto_handled if g)
    return RatioMetric(
        "false auto-handle rate", bad, len(auto_handled),
        note="of labeled examples the system auto-handled, how many gold says should have escalated instead",
    )


def automation_coverage(predicted_escalate: Sequence[bool]) -> RatioMetric:
    """Fraction of ALL golden examples the system chose to auto-handle.
    Doesn't require any gold label — this is purely about system behavior,
    not correctness."""
    total = len(predicted_escalate)
    auto = sum(1 for p in predicted_escalate if not p)
    return RatioMetric(
        "automation coverage", auto, total,
        note="fraction of ALL golden examples the system auto-handled (no gold label required)",
    )


@dataclass
class SafeAutomationRow:
    example_id: str
    predicted_decision: str  # "auto_handle" | "escalate"
    predicted_intent: str
    gold_intent: Optional[str]
    gold_should_escalate: Optional[bool]
    reply_quality_acceptable: Optional[bool]


def _satisfies_safe_automation(row: SafeAutomationRow) -> Optional[bool]:
    """True/False if we have enough gold labels to judge this row at all,
    None if we don't (and therefore can't count it toward the numerator
    OR conclude it's a miss — it's simply unknown)."""
    if row.gold_intent is None or row.gold_should_escalate is None or row.reply_quality_acceptable is None:
        return None
    if row.predicted_decision != "auto_handle":
        return False
    if row.predicted_intent != row.gold_intent:
        return False
    if not row.reply_quality_acceptable:
        return False
    if row.gold_should_escalate:
        return False
    return True


def safe_automation_coverage(rows: list[SafeAutomationRow], total_golden_examples: Optional[int] = None) -> RatioMetric:
    """THE HEADLINE METRIC: of all golden examples, how many were
    auto-handled AND had the correct predicted intent AND an acceptable
    reply AND didn't contradict gold's escalation call.

    `total_golden_examples` lets the denominator reflect the true golden
    set size (e.g. 200) even while only a subset is labeled — but see
    `safe_automation_coverage_over_labeled_subset` for the narrower,
    label-count-denominated version that `run_eval` prefers while labeling
    is incomplete (reporting the full-200 denominator against a near-zero
    numerator during early labeling would itself be a misleading headline
    number — see docs/EVALUATION.md).
    """
    denom = total_golden_examples if total_golden_examples is not None else len(rows)
    numerator = sum(1 for r in rows if _satisfies_safe_automation(r) is True)
    return RatioMetric(
        "safe automation coverage",
        numerator,
        denom,
        note=(
            "requires auto-handled + correct intent + acceptable reply + gold agrees "
            "auto-handling was appropriate; examples missing any gold label can never "
            "count toward the numerator"
        ),
    )


def safe_automation_coverage_over_labeled_subset(rows: list[SafeAutomationRow]) -> RatioMetric:
    """The narrower, defensible fallback named in the assignment: same
    criteria, but the denominator is the number of rows that actually have
    every required gold field filled in, so missing labels don't silently
    deflate the ratio by inflating an unrelated denominator. Prefer this
    while labeling is incomplete."""
    labeled_rows = [r for r in rows if _satisfies_safe_automation(r) is not None]
    return safe_automation_coverage(labeled_rows, total_golden_examples=len(labeled_rows))
