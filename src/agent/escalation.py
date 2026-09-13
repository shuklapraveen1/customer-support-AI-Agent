"""Multi-signal auto-handle vs. escalate policy.

Deliberately NOT a single `confidence < threshold` check: `decide()` walks
an ordered list of named rules, each backed by its own signal (safety
keywords, out-of-distribution retrieval, configured high-risk
intents/keywords, evidence volume, message length as a crude "missing
context" proxy, intent confidence, an "ambiguous — only just above
threshold" band distinct from plain low-confidence, retrieved-evidence
agreement with the predicted intent, and reply grounding). The first rule
that fires wins and supplies the stated reason; a message is only
auto-handled once every one of them has cleared its bar. This mirrors how
a real escalation policy should read: a named, auditable trail of which
specific condition sent something to a human, not an opaque score.

Threshold provenance: the values in `configs/escalation.yaml` are
illustrative engineering defaults, NOT tuned against dev-set human
judgments — this project has no real, human-labeled dev data yet (see
README; the real Kaggle dataset couldn't be downloaded in this sandbox).
Recalibrating them once that data exists must use DEV-split performance
only, never the test split — mirroring the same discipline as everything
else in this project.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel, Field

from src.intent.classifier import IntentPrediction
from src.retrieval.search import RetrievalResult

DEFAULT_ESCALATION_CONFIG_PATH = "configs/escalation.yaml"

_DEFAULT_HIGH_RISK_KEYWORDS = [
    "lawsuit", "attorney", "lawyer", "legal action", "sue you", "gdpr",
    "data breach", "hacked", "security breach", "fraud", "chargeback",
    "delete my account", "close my account", "cancel my account",
]

_DEFAULT_SAFETY_KEYWORDS = [
    "kill myself", "suicide", "self harm", "self-harm", "hurt myself", "end my life",
]


class EscalationConfig(BaseModel):
    min_intent_confidence: float = 0.55
    min_retrieval_similarity: float = 0.45
    min_grounding_score: float = 0.5
    min_evidence_count: int = 1
    min_evidence_agreement: float = 0.5
    ood_similarity_threshold: float = 0.35
    ambiguous_margin: float = 0.1
    min_message_words: int = 3
    high_risk_intents: list[str] = Field(default_factory=list)
    high_risk_keywords: list[str] = Field(default_factory=lambda: list(_DEFAULT_HIGH_RISK_KEYWORDS))
    safety_keywords: list[str] = Field(default_factory=lambda: list(_DEFAULT_SAFETY_KEYWORDS))


def load_escalation_config(path: Optional[str] = None) -> EscalationConfig:
    """Load `configs/escalation.yaml` (or `path`), falling back to the
    built-in defaults above if the file doesn't exist — so tests never
    require one on disk, same pattern as `src.config.load_config`."""
    config_path = Path(path or os.environ.get("ESCALATION_CONFIG_PATH", DEFAULT_ESCALATION_CONFIG_PATH))
    raw: dict = {}
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}
    return EscalationConfig(**raw)


@dataclass
class EscalationSignals:
    intent_confidence: float
    retrieval_score: float
    evidence_count: int
    risk_flag: bool
    ood_flag: bool
    grounding_score: float
    reply_confidence: float
    evidence_agreement: float
    ambiguous_intent: bool
    missing_context: bool
    safety_flag: bool
    llm_parse_error: bool

    def as_dict(self) -> dict:
        return {
            "intent_confidence": self.intent_confidence,
            "retrieval_score": self.retrieval_score,
            "evidence_count": self.evidence_count,
            "risk_flag": self.risk_flag,
            "ood_flag": self.ood_flag,
            "grounding_score": self.grounding_score,
            "reply_confidence": self.reply_confidence,
            "evidence_agreement": self.evidence_agreement,
            "ambiguous_intent": self.ambiguous_intent,
            "missing_context": self.missing_context,
            "safety_flag": self.safety_flag,
            "llm_parse_error": self.llm_parse_error,
        }


@dataclass
class EscalationDecision:
    decision: str  # "auto_handle" | "escalate"
    reason: str
    signals: EscalationSignals

    def as_dict(self) -> dict:
        return {"decision": self.decision, "reason": self.reason, "signals": self.signals.as_dict()}


def _contains_any(text: str, phrases: list[str]) -> bool:
    lowered = text.lower()
    return any(phrase.lower() in lowered for phrase in phrases)


def compute_signals(
    customer_message: str,
    intent_prediction: IntentPrediction,
    evidence: list[RetrievalResult],
    grounding_score: float,
    reply_confidence: float,
    config: EscalationConfig,
    llm_parse_error: bool = False,
) -> EscalationSignals:
    """Compute every signal `decide()` can act on. Pure function of its
    inputs — nothing here calls out to a model or reuses a cached/stale
    value, so signals always reflect the actual message/evidence/reply
    passed in for this call."""
    retrieval_score = evidence[0].similarity_score if evidence else 0.0
    evidence_count = len(evidence)
    ood_flag = retrieval_score < config.ood_similarity_threshold

    matching_intent_evidence = [e for e in evidence if e.intent == intent_prediction.intent]
    evidence_agreement = (len(matching_intent_evidence) / evidence_count) if evidence_count else 0.0

    risk_flag = (
        intent_prediction.intent in config.high_risk_intents
        or _contains_any(customer_message, config.high_risk_keywords)
    )
    safety_flag = _contains_any(customer_message, config.safety_keywords)

    ambiguous_intent = (
        config.min_intent_confidence
        <= intent_prediction.confidence
        < config.min_intent_confidence + config.ambiguous_margin
    )
    missing_context = len(customer_message.split()) < config.min_message_words

    return EscalationSignals(
        intent_confidence=intent_prediction.confidence,
        retrieval_score=retrieval_score,
        evidence_count=evidence_count,
        risk_flag=risk_flag,
        ood_flag=ood_flag,
        grounding_score=grounding_score,
        reply_confidence=reply_confidence,
        evidence_agreement=evidence_agreement,
        ambiguous_intent=ambiguous_intent,
        missing_context=missing_context,
        safety_flag=safety_flag,
        llm_parse_error=llm_parse_error,
    )


def decide(
    signals: EscalationSignals,
    intent_prediction: IntentPrediction,
    config: EscalationConfig,
) -> EscalationDecision:
    """Ordered rule evaluation — first match wins, and the reason names
    exactly which signal fired. `auto_handle` is only reached at the end,
    once every rule above it has been checked and passed.
    """
    if signals.safety_flag:
        return EscalationDecision(
            decision="escalate",
            reason=(
                "Message contains language suggesting a safety/self-harm concern; "
                "always routed to a human regardless of every other signal."
            ),
            signals=signals,
        )

    if signals.llm_parse_error:
        return EscalationDecision(
            decision="escalate",
            reason=(
                "Reply generation did not produce valid structured output; escalating "
                "rather than risking a malformed or ungrounded reply reaching the customer."
            ),
            signals=signals,
        )

    if signals.ood_flag:
        return EscalationDecision(
            decision="escalate",
            reason=(
                f"Historical evidence is insufficient: top retrieval similarity "
                f"{signals.retrieval_score:.2f} is below the out-of-distribution threshold "
                f"{config.ood_similarity_threshold:.2f} — this message doesn't closely "
                "resemble anything in the historical corpus."
            ),
            signals=signals,
        )

    if signals.risk_flag:
        return EscalationDecision(
            decision="escalate",
            reason=(
                f"Intent '{intent_prediction.intent}' or the message content matched a "
                "configured high-risk category; high-risk requests always require human review."
            ),
            signals=signals,
        )

    if signals.evidence_count < config.min_evidence_count:
        return EscalationDecision(
            decision="escalate",
            reason=(
                f"Only {signals.evidence_count} historical precedent(s) found; below the "
                f"minimum of {config.min_evidence_count} required to auto-handle."
            ),
            signals=signals,
        )

    if signals.missing_context:
        return EscalationDecision(
            decision="escalate",
            reason="The message is too short to reliably act on automatically; needs a human to ask a follow-up.",
            signals=signals,
        )

    if signals.intent_confidence < config.min_intent_confidence:
        return EscalationDecision(
            decision="escalate",
            reason=(
                f"Intent confidence {signals.intent_confidence:.2f} is below the minimum "
                f"{config.min_intent_confidence:.2f} required to auto-handle."
            ),
            signals=signals,
        )

    if signals.ambiguous_intent:
        return EscalationDecision(
            decision="escalate",
            reason=(
                f"Intent confidence {signals.intent_confidence:.2f} only marginally clears "
                f"the minimum threshold ({config.min_intent_confidence:.2f}); treated as "
                "ambiguous and routed to a human rather than trusted outright."
            ),
            signals=signals,
        )

    if signals.retrieval_score < config.min_retrieval_similarity:
        return EscalationDecision(
            decision="escalate",
            reason=(
                f"Top retrieval similarity {signals.retrieval_score:.2f} is below the "
                f"minimum {config.min_retrieval_similarity:.2f} required to auto-handle."
            ),
            signals=signals,
        )

    if signals.evidence_agreement < config.min_evidence_agreement:
        return EscalationDecision(
            decision="escalate",
            reason=(
                f"Retrieved historical evidence disagrees with the predicted intent "
                f"(agreement {signals.evidence_agreement:.2f} < {config.min_evidence_agreement:.2f}); "
                "needs human judgment rather than an automated reply."
            ),
            signals=signals,
        )

    if signals.grounding_score < config.min_grounding_score:
        return EscalationDecision(
            decision="escalate",
            reason=(
                f"Reply grounding score {signals.grounding_score:.2f} is below the minimum "
                f"{config.min_grounding_score:.2f}; the drafted reply may not be well-supported "
                "by historical evidence."
            ),
            signals=signals,
        )

    return EscalationDecision(
        decision="auto_handle",
        reason=(
            f"All signals within configured thresholds: intent confidence "
            f"{signals.intent_confidence:.2f}, retrieval similarity {signals.retrieval_score:.2f}, "
            f"evidence agreement {signals.evidence_agreement:.2f}, grounding "
            f"{signals.grounding_score:.2f}, no risk/safety flags."
        ),
        signals=signals,
    )
