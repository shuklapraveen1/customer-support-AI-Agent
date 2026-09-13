from datetime import datetime
from pathlib import Path

import pytest

from src.agent.escalation import (
    EscalationConfig,
    compute_signals,
    decide,
    load_escalation_config,
)
from src.intent.classifier import IntentPrediction
from src.retrieval.search import RetrievalResult

REAL_ESCALATION_YAML = Path(__file__).parent.parent / "configs" / "escalation.yaml"


def _evidence(
    resolution_id: str = "res_1",
    similarity_score: float = 0.9,
    intent: str = "delivery_delay",
) -> RetrievalResult:
    return RetrievalResult(
        resolution_id=resolution_id,
        customer_message="my order never arrived",
        brand_reply="So sorry, we'll look into it.",
        intent=intent,
        similarity_score=similarity_score,
        conversation_id="c1",
        timestamp=datetime(2020, 1, 1),
    )


def _prediction(intent: str = "delivery_delay", confidence: float = 0.9) -> IntentPrediction:
    return IntentPrediction(intent=intent, confidence=confidence, reason="test reason")


def _config(**overrides) -> EscalationConfig:
    base = dict(
        min_intent_confidence=0.55,
        min_retrieval_similarity=0.45,
        min_grounding_score=0.5,
        min_evidence_count=1,
        min_evidence_agreement=0.5,
        ood_similarity_threshold=0.35,
        ambiguous_margin=0.1,
        min_message_words=3,
        high_risk_intents=["billing_dispute"],
    )
    base.update(overrides)
    return EscalationConfig(**base)


# --- Config loading -----------------------------------------------------


def test_escalation_config_has_sane_defaults_with_no_file():
    config = load_escalation_config("/nonexistent/path/escalation.yaml")
    assert 0.0 < config.min_intent_confidence < 1.0
    assert 0.0 < config.min_retrieval_similarity < 1.0
    assert config.ood_similarity_threshold < config.min_retrieval_similarity
    assert isinstance(config.high_risk_keywords, list) and config.high_risk_keywords
    assert isinstance(config.safety_keywords, list) and config.safety_keywords


def test_real_escalation_yaml_loads_and_is_internally_consistent():
    config = load_escalation_config(str(REAL_ESCALATION_YAML))
    assert config.ood_similarity_threshold < config.min_retrieval_similarity
    assert 0.0 <= config.min_intent_confidence <= 1.0
    assert 0.0 <= config.min_grounding_score <= 1.0
    assert config.min_evidence_count >= 1
    assert "billing_dispute" in config.high_risk_intents


# --- compute_signals -----------------------------------------------------


def test_compute_signals_basic_happy_path():
    config = _config()
    prediction = _prediction(confidence=0.9)
    evidence = [_evidence(similarity_score=0.9), _evidence("res_2", 0.85)]

    signals = compute_signals(
        customer_message="my order never arrived and its been a week",
        intent_prediction=prediction,
        evidence=evidence,
        grounding_score=0.8,
        reply_confidence=0.8,
        config=config,
    )
    assert signals.intent_confidence == 0.9
    assert signals.retrieval_score == 0.9
    assert signals.evidence_count == 2
    assert signals.risk_flag is False
    assert signals.ood_flag is False
    assert signals.evidence_agreement == 1.0
    assert signals.ambiguous_intent is False
    assert signals.missing_context is False
    assert signals.safety_flag is False


def test_compute_signals_no_evidence():
    config = _config()
    signals = compute_signals(
        customer_message="my order never arrived and its been a week",
        intent_prediction=_prediction(),
        evidence=[],
        grounding_score=0.0,
        reply_confidence=0.0,
        config=config,
    )
    assert signals.evidence_count == 0
    assert signals.retrieval_score == 0.0
    assert signals.ood_flag is True
    assert signals.evidence_agreement == 0.0


def test_compute_signals_risk_flag_from_intent():
    config = _config(high_risk_intents=["billing_dispute"])
    signals = compute_signals(
        customer_message="i was charged twice please help",
        intent_prediction=_prediction(intent="billing_dispute", confidence=0.9),
        evidence=[_evidence(intent="billing_dispute")],
        grounding_score=0.9,
        reply_confidence=0.9,
        config=config,
    )
    assert signals.risk_flag is True


def test_compute_signals_risk_flag_from_keyword_regardless_of_intent():
    config = _config(high_risk_keywords=["lawsuit"])
    signals = compute_signals(
        customer_message="I'm going to file a lawsuit against you",
        intent_prediction=_prediction(intent="delivery_delay", confidence=0.9),
        evidence=[_evidence()],
        grounding_score=0.9,
        reply_confidence=0.9,
        config=config,
    )
    assert signals.risk_flag is True


def test_compute_signals_safety_flag():
    config = _config()
    signals = compute_signals(
        customer_message="I feel like I want to kill myself over this",
        intent_prediction=_prediction(confidence=0.9),
        evidence=[_evidence()],
        grounding_score=0.9,
        reply_confidence=0.9,
        config=config,
    )
    assert signals.safety_flag is True


def test_compute_signals_missing_context_for_short_message():
    config = _config(min_message_words=3)
    signals = compute_signals(
        customer_message="help please",
        intent_prediction=_prediction(),
        evidence=[_evidence()],
        grounding_score=0.9,
        reply_confidence=0.9,
        config=config,
    )
    assert signals.missing_context is True


def test_compute_signals_ambiguous_band():
    config = _config(min_intent_confidence=0.5, ambiguous_margin=0.1)
    signals = compute_signals(
        customer_message="my order never arrived and its been a week",
        intent_prediction=_prediction(confidence=0.55),
        evidence=[_evidence()],
        grounding_score=0.9,
        reply_confidence=0.9,
        config=config,
    )
    assert signals.ambiguous_intent is True

    signals_confident = compute_signals(
        customer_message="my order never arrived and its been a week",
        intent_prediction=_prediction(confidence=0.9),
        evidence=[_evidence()],
        grounding_score=0.9,
        reply_confidence=0.9,
        config=config,
    )
    assert signals_confident.ambiguous_intent is False


def test_compute_signals_evidence_agreement_with_mixed_intents():
    config = _config()
    evidence = [
        _evidence("res_1", 0.9, intent="delivery_delay"),
        _evidence("res_2", 0.8, intent="billing_dispute"),
    ]
    signals = compute_signals(
        customer_message="my order never arrived and its been a week",
        intent_prediction=_prediction(intent="delivery_delay", confidence=0.9),
        evidence=evidence,
        grounding_score=0.9,
        reply_confidence=0.9,
        config=config,
    )
    assert signals.evidence_agreement == 0.5


# --- decide() ---------------------------------------------------------------


def test_decide_auto_handles_when_all_signals_pass():
    config = _config()
    prediction = _prediction(confidence=0.9)
    evidence = [_evidence(similarity_score=0.9), _evidence("res_2", 0.85)]
    signals = compute_signals(
        "my order never arrived and its been a week", prediction, evidence, 0.9, 0.9, config
    )
    decision = decide(signals, prediction, config)
    assert decision.decision == "auto_handle"
    assert "confidence" in decision.reason.lower()


def test_decide_escalates_on_safety_flag_even_with_strong_evidence():
    config = _config()
    prediction = _prediction(confidence=0.99)
    evidence = [_evidence(similarity_score=0.99), _evidence("res_2", 0.98)]
    signals = compute_signals(
        "I want to kill myself because of this order issue", prediction, evidence, 0.99, 0.99, config
    )
    decision = decide(signals, prediction, config)
    assert decision.decision == "escalate"
    assert "safety" in decision.reason.lower()


def test_decide_escalates_on_low_similarity_ood():
    config = _config()
    prediction = _prediction(confidence=0.9)
    evidence = [_evidence(similarity_score=0.1)]
    signals = compute_signals(
        "some totally unrelated novel message", prediction, evidence, 0.9, 0.9, config
    )
    decision = decide(signals, prediction, config)
    assert decision.decision == "escalate"
    assert "insufficient" in decision.reason.lower() or "out-of-distribution" in decision.reason.lower()


def test_decide_escalates_on_high_risk_intent_even_with_high_confidence():
    config = _config(high_risk_intents=["billing_dispute"])
    prediction = _prediction(intent="billing_dispute", confidence=0.95)
    evidence = [_evidence(intent="billing_dispute", similarity_score=0.95)]
    signals = compute_signals(
        "i was charged twice for the same order", prediction, evidence, 0.95, 0.95, config
    )
    decision = decide(signals, prediction, config)
    assert decision.decision == "escalate"
    assert "high-risk" in decision.reason.lower()


def test_decide_auto_handles_high_confidence_and_strong_evidence():
    config = _config()
    prediction = _prediction(intent="delivery_delay", confidence=0.95)
    evidence = [_evidence(similarity_score=0.92), _evidence("res_2", 0.9)]
    signals = compute_signals(
        "my order never arrived and its been over a week now", prediction, evidence, 0.9, 0.9, config
    )
    decision = decide(signals, prediction, config)
    assert decision.decision == "auto_handle"


def test_decide_escalates_on_ambiguous_intent():
    config = _config(min_intent_confidence=0.5, ambiguous_margin=0.15)
    prediction = _prediction(confidence=0.55)
    evidence = [_evidence(similarity_score=0.9)]
    signals = compute_signals(
        "my order never arrived and its been a week", prediction, evidence, 0.9, 0.9, config
    )
    decision = decide(signals, prediction, config)
    assert decision.decision == "escalate"
    assert "ambiguous" in decision.reason.lower()


def test_decide_escalates_on_missing_evidence():
    config = _config()
    prediction = _prediction(confidence=0.9)
    signals = compute_signals(
        "my order never arrived and its been a week", prediction, [], 0.0, 0.0, config
    )
    decision = decide(signals, prediction, config)
    assert decision.decision == "escalate"


def test_decide_escalates_on_llm_parse_error_regardless_of_other_signals():
    config = _config()
    prediction = _prediction(confidence=0.95)
    evidence = [_evidence(similarity_score=0.95)]
    signals = compute_signals(
        "my order never arrived and its been a week",
        prediction,
        evidence,
        grounding_score=0.0,
        reply_confidence=0.0,
        config=config,
        llm_parse_error=True,
    )
    decision = decide(signals, prediction, config)
    assert decision.decision == "escalate"
    reason_lower = decision.reason.lower()
    assert "structured output" in reason_lower or "parse" in reason_lower or "reply generation" in reason_lower


def test_decide_escalates_on_low_grounding_score():
    config = _config()
    prediction = _prediction(confidence=0.9)
    evidence = [_evidence(similarity_score=0.9)]
    signals = compute_signals(
        "my order never arrived and its been a week",
        prediction,
        evidence,
        grounding_score=0.1,
        reply_confidence=0.9,
        config=config,
    )
    decision = decide(signals, prediction, config)
    assert decision.decision == "escalate"
    assert "grounding" in decision.reason.lower()


def test_decide_escalates_on_evidence_disagreement():
    config = _config(min_evidence_agreement=0.9)
    prediction = _prediction(intent="delivery_delay", confidence=0.9)
    evidence = [
        _evidence("res_1", 0.9, intent="delivery_delay"),
        _evidence("res_2", 0.85, intent="billing_dispute"),
    ]
    signals = compute_signals(
        "my order never arrived and its been a week", prediction, evidence, 0.9, 0.9, config
    )
    decision = decide(signals, prediction, config)
    assert decision.decision == "escalate"
    assert "disagree" in decision.reason.lower()


def test_decide_reason_always_names_the_triggering_signal_not_a_bare_score():
    """Sanity check on the 'not just confidence<threshold' requirement: the
    reason text should be a real explanation, not just a bare number."""
    config = _config()
    prediction = _prediction(confidence=0.2)
    evidence = [_evidence(similarity_score=0.1)]
    signals = compute_signals("short msg here", prediction, evidence, 0.9, 0.9, config)
    decision = decide(signals, prediction, config)
    assert decision.decision == "escalate"
    assert len(decision.reason) > 20
