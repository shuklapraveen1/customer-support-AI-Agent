"""Shared intent classification interface.

Every classifier in this package — the majority baseline
(`baseline_majority.py`), the TF-IDF + Logistic Regression baseline
(`baseline_tfidf.py`), and the embedding nearest-centroid classifier
(`embedding.py`) — implements the same `fit` / `predict` contract defined
here and returns the same `IntentPrediction` shape: `intent`, `confidence`,
`reason`.

All of them are built against a single `IntentTaxonomy` loaded from
`configs/intents.yaml`. `IntentTaxonomy.validate_label` /
`validate_labels` is the one place an out-of-taxonomy label is rejected —
every classifier routes both training labels and predicted labels through
it, so nothing here can silently invent a new intent that isn't in the
config file.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import yaml

DEFAULT_INTENTS_PATH = "configs/intents.yaml"


class IntentConfigError(ValueError):
    """Raised when configs/intents.yaml is missing, malformed, or a required
    field on an intent entry is absent."""


class InvalidIntentLabelError(ValueError):
    """Raised when a classifier is asked to train on or predicts a label
    that is not part of the loaded IntentTaxonomy."""


@dataclass(frozen=True)
class IntentDefinition:
    id: str
    name: str
    description: str
    examples: tuple[str, ...]


@dataclass(frozen=True)
class IntentPrediction:
    intent: str
    confidence: float
    reason: str


class IntentTaxonomy:
    """The single source of truth for which intent labels are legal.

    Loaded from a YAML file shaped like:

        unknown_label: other
        intents:
          - id: delivery_delay
            name: Delivery Delay
            description: "..."
            examples: ["...", "..."]
    """

    REQUIRED_FIELDS = ("id", "name", "description", "examples")

    def __init__(self, intents: list[IntentDefinition], unknown_label: str = "other"):
        if not intents:
            raise IntentConfigError("Intent taxonomy must contain at least one intent.")
        ids = [i.id for i in intents]
        if len(ids) != len(set(ids)):
            dupes = sorted({i for i in ids if ids.count(i) > 1})
            raise IntentConfigError(f"Duplicate intent id(s) in taxonomy: {dupes}")
        self._intents = {i.id: i for i in intents}
        self._order = ids
        self.unknown_label = unknown_label

    @classmethod
    def from_yaml(cls, path: Optional[str] = None) -> "IntentTaxonomy":
        config_path = Path(path or os.environ.get("INTENTS_CONFIG_PATH", DEFAULT_INTENTS_PATH))
        if not config_path.exists():
            raise IntentConfigError(f"Intent taxonomy file not found: {config_path}")
        with open(config_path, "r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, raw: dict) -> "IntentTaxonomy":
        unknown_label = raw.get("unknown_label", "other")
        raw_intents = raw.get("intents")
        if not raw_intents:
            raise IntentConfigError("Taxonomy config has no 'intents' entries.")

        intents = []
        for entry in raw_intents:
            missing = [f for f in cls.REQUIRED_FIELDS if f not in entry]
            if missing:
                raise IntentConfigError(
                    f"Intent entry {entry.get('id', '<no id>')} is missing required "
                    f"field(s): {missing}"
                )
            examples = entry["examples"]
            if not isinstance(examples, list) or len(examples) == 0:
                raise IntentConfigError(
                    f"Intent '{entry['id']}' must have a non-empty 'examples' list."
                )
            intents.append(
                IntentDefinition(
                    id=str(entry["id"]),
                    name=str(entry["name"]),
                    description=str(entry["description"]),
                    examples=tuple(str(e) for e in examples),
                )
            )
        return cls(intents, unknown_label=unknown_label)

    @property
    def labels(self) -> list[str]:
        """Ordered list of legal intent ids (excludes the unknown/fallback
        label, which is not a discoverable intent)."""
        return list(self._order)

    @property
    def allowed_labels(self) -> set[str]:
        """Legal intent ids plus the unknown/fallback label."""
        return set(self._order) | {self.unknown_label}

    def __contains__(self, label: str) -> bool:
        return label in self.allowed_labels

    def __len__(self) -> int:
        return len(self._order)

    def get(self, intent_id: str) -> IntentDefinition:
        if intent_id not in self._intents:
            raise KeyError(intent_id)
        return self._intents[intent_id]

    def validate_label(self, label: str) -> str:
        if label not in self.allowed_labels:
            raise InvalidIntentLabelError(
                f"'{label}' is not a valid intent. Allowed labels: {sorted(self.allowed_labels)}"
            )
        return label

    def validate_labels(self, labels: Sequence[str]) -> None:
        for label in labels:
            self.validate_label(label)


class BaseIntentClassifier:
    """Shared fit/predict contract.

    Subclasses must route every label they are about to train on or return
    through `self.taxonomy.validate_label(...)` (or `self._validate`) —
    that call is the only thing standing between a classifier and silently
    inventing an intent outside the configured taxonomy.
    """

    def __init__(self, taxonomy: IntentTaxonomy):
        self.taxonomy = taxonomy
        self._fitted = False

    def _validate(self, label: str) -> str:
        return self.taxonomy.validate_label(label)

    def fit(self, texts: Sequence[str], labels: Sequence[str]) -> "BaseIntentClassifier":
        raise NotImplementedError

    def predict(self, texts: Sequence[str]) -> list[IntentPrediction]:
        raise NotImplementedError

    def _require_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError(f"{type(self).__name__} must be fit() before predict().")
