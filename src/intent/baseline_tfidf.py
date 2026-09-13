"""Baseline #2: TF-IDF + Logistic Regression.

A standard, cheap, text-aware baseline: word/bigram TF-IDF features feeding
a multinomial logistic regression. This is the bar the embedding classifier
(`src.intent.embedding.EmbeddingNearestCentroidClassifier`) has to clear to
justify its extra complexity — if it doesn't beat this, that's a real
finding to report, not something to paper over.
"""

from __future__ import annotations

from typing import Sequence

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from src.intent.classifier import BaseIntentClassifier, IntentPrediction, IntentTaxonomy


class TfidfLogisticClassifier(BaseIntentClassifier):
    def __init__(
        self,
        taxonomy: IntentTaxonomy,
        max_features: int = 5000,
        C: float = 1.0,
        random_state: int = 42,
    ):
        super().__init__(taxonomy)
        self.vectorizer = TfidfVectorizer(
            max_features=max_features, ngram_range=(1, 2), min_df=1, sublinear_tf=True
        )
        self.model = LogisticRegression(max_iter=2000, C=C, random_state=random_state)
        self._classes: list[str] = []

    def fit(self, texts: Sequence[str], labels: Sequence[str]) -> "TfidfLogisticClassifier":
        texts = list(texts)
        labels = list(labels)
        if not texts:
            raise ValueError("Cannot fit TfidfLogisticClassifier on zero training examples.")
        if len(texts) != len(labels):
            raise ValueError("texts and labels must be the same length.")
        self.taxonomy.validate_labels(labels)

        X = self.vectorizer.fit_transform(texts)
        self.model.fit(X, labels)
        self._classes = list(self.model.classes_)
        self._fitted = True
        return self

    def predict(self, texts: Sequence[str]) -> list[IntentPrediction]:
        self._require_fitted()
        X = self.vectorizer.transform(list(texts))
        probs = self.model.predict_proba(X)

        predictions = []
        for row in probs:
            idx = int(row.argmax())
            label = self._classes[idx]
            confidence = float(row[idx])
            predictions.append(
                IntentPrediction(
                    intent=label,
                    confidence=confidence,
                    reason=(
                        "TF-IDF + Logistic Regression: highest predicted probability "
                        f"among trained intents was '{label}' ({confidence:.2f})."
                    ),
                )
            )
        return predictions
