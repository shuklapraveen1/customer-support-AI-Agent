"""The Hiver support agent: classify intent -> retrieve historical evidence
-> draft a grounded reply -> decide auto-handle vs. escalate.

`run_agent(...)` is the single entrypoint tying together `src.intent`
(classification), `src.retrieval` (historical evidence), `src.agent.reply`
(grounded drafting via an LLM provider), and `src.agent.escalation` (the
multi-signal auto-handle/escalate policy) into one `AgentResult`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from src.agent.escalation import (
    EscalationConfig,
    compute_signals,
    decide,
    load_escalation_config,
)
from src.agent.llm import LLMProvider, get_llm_provider
from src.agent.reply import generate_reply
from src.intent.classifier import BaseIntentClassifier
from src.retrieval.search import RetrievalResult


@dataclass
class AgentResult:
    customer_message: str
    intent: str
    intent_confidence: float
    intent_reason: str
    retrieved_evidence: list[RetrievalResult]
    reply: Optional[str]
    decision: str
    reason: str
    signals: dict

    def as_dict(self) -> dict:
        """The top-level result shape the assignment specifies. Extra
        context (`intent_reason`, the `RetrievalResult` objects themselves)
        stays available as attributes for anything that wants it, but is
        kept out of this dict to match the spec exactly."""
        return {
            "customer_message": self.customer_message,
            "intent": self.intent,
            "intent_confidence": self.intent_confidence,
            "retrieved_evidence": [e.as_dict() for e in self.retrieved_evidence],
            "reply": self.reply,
            "decision": self.decision,
            "reason": self.reason,
            "signals": self.signals,
        }


def run_agent(
    customer_message: str,
    intent_classifier: BaseIntentClassifier,
    searcher,
    llm_provider: Optional[LLMProvider] = None,
    escalation_config: Optional[EscalationConfig] = None,
    top_k: int = 5,
    filter_evidence_by_intent: bool = True,
) -> AgentResult:
    """Run the full pipeline for one customer message.

    `searcher` is anything with a `.retrieve(query, intent=None, top_k=5)`
    method returning `list[RetrievalResult]` — a `ResolutionIndex` or a
    `TfidfRetrievalBaseline`. `llm_provider`/`escalation_config` default to
    `get_llm_provider()` / `load_escalation_config()` if not given, which is
    what makes this runnable with zero configuration (mock LLM, default
    thresholds).
    """
    llm_provider = llm_provider or get_llm_provider()
    escalation_config = escalation_config or load_escalation_config()

    intent_prediction = intent_classifier.predict([customer_message])[0]

    evidence = searcher.retrieve(
        customer_message,
        intent=intent_prediction.intent if filter_evidence_by_intent else None,
        top_k=top_k,
    )
    # An intent-filtered search coming back empty doesn't necessarily mean
    # "nothing like this exists" — it can mean the predicted intent has no
    # corpus coverage. Fall back to an unfiltered search so a rare/
    # misclassified intent doesn't manufacture a false "no evidence" signal.
    # Each returned result still carries its own (possibly different)
    # intent — nothing here relabels evidence to match the prediction.
    if filter_evidence_by_intent and not evidence:
        evidence = searcher.retrieve(customer_message, intent=None, top_k=top_k)

    reply_result = generate_reply(customer_message, intent_prediction, evidence, llm_provider)

    signals = compute_signals(
        customer_message=customer_message,
        intent_prediction=intent_prediction,
        evidence=evidence,
        grounding_score=reply_result.grounding_score,
        reply_confidence=reply_result.confidence,
        config=escalation_config,
        llm_parse_error=reply_result.parse_error,
    )
    decision = decide(signals, intent_prediction, escalation_config)

    return AgentResult(
        customer_message=customer_message,
        intent=intent_prediction.intent,
        intent_confidence=intent_prediction.confidence,
        intent_reason=intent_prediction.reason,
        retrieved_evidence=evidence,
        reply=reply_result.reply,
        decision=decision.decision,
        reason=decision.reason,
        signals=decision.signals.as_dict(),
    )
