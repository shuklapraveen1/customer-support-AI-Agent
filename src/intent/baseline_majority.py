"""Baseline #1: majority-class classifier.

Always predicts whichever intent was most frequent in the training data,
completely ignoring the input text. Any classifier worth using on this task
has to beat this — it is the floor a real system must clear, not a serious
competitor.
"""

from __future__ import annotations

from collections import Counter
from typing import Optional, Sequence

from src.intent.classifier import BaseIntentClassifier, IntentPrediction, IntentTaxonomy


class MajorityClassifier(BaseIntentClassifier):
    def __init__(self, taxonomy: IntentTaxonomy):
        super().__init__(taxonomy)
        self.majority_label: Optional[str] = None
        self.majority_share: float = 0.0

    def fit(self, texts: Sequence[str], labels: Sequence[str]) -> "MajorityClassifier":
        if not labels:
            raise ValueError("Cannot fit MajorityClassifier on zero training examples.")
        self.taxonomy.validate_labels(labels)

        counts = Counter(labels)
        label, count = counts.most_common(1)[0]
        self.majority_label = label
        self.majority_share = count / len(labels)
        self._fitted = True
        return self

    def predict(self, texts: Sequence[str]) -> list[IntentPrediction]:
        self._require_fitted()
        reason = (
            f"Majority-class baseline: '{self.majority_label}' was the most frequent "
            f"training intent ({self.majority_share:.1%} of training examples); "
            "the input text is not used."
        )
        return [
            IntentPrediction(intent=self.majority_label, confidence=self.majority_share, reason=reason)
            for _ in texts
        ]
