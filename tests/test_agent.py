import os
from datetime import datetime

import pytest

from src.agent import AgentResult, run_agent
from src.agent.escalation import EscalationConfig
from src.agent.llm import (
    LLMConfigError,
    LLMResponse,
    MockLLMProvider,
    get_llm_provider,
)
from src.agent.prompts import build_user_prompt
from src.agent.reply import ReplyResult, generate_reply
from src.intent.classifier import IntentPrediction
from src.retrieval.search import RetrievalResult


def _evidence(resolution_id="res_1", similarity=0.9, intent="delivery_delay", brand_reply="So sorry, DM us."):
    return RetrievalResult(
        resolution_id=resolution_id,
        customer_message="my order never arrived",
        brand_reply=brand_reply,
        intent=intent,
        similarity_score=similarity,
        conversation_id="c1",
        timestamp=datetime(2021, 1, 1),
    )


def _prediction(intent="delivery_delay", confidence=0.9, reason="test"):
    return IntentPrediction(intent=intent, confidence=confidence, reason=reason)


class StubClassifier:
    """Minimal duck-typed intent classifier: always returns the same
    IntentPrediction, ignoring the input text — enough for testing agent
    orchestration in isolation from any real trained classifier."""

    def __init__(self, prediction: IntentPrediction):
        self._prediction = prediction

    def predict(self, texts):
        return [self._prediction for _ in texts]


class StubSearcher:
    """Minimal duck-typed searcher: returns a fixed evidence list regardless
    of query, optionally filtered by intent (mirroring the real
    ResolutionIndex/TfidfRetrievalBaseline contract)."""

    def __init__(self, evidence: list[RetrievalResult]):
        self._evidence = evidence

    def retrieve(self, query, intent=None, top_k=5):
        pool = self._evidence
        if intent is not None:
            pool = [e for e in pool if e.intent == intent]
        return pool[:top_k]


class EchoBrokenProvider:
    """Test double LLM provider that always returns unparseable garbage,
    to exercise the invalid-JSON path deterministically."""

    name = "broken-test-stub"
    model = "broken-v0"

    def complete(self, system_prompt, user_prompt):
        return LLMResponse(text="not json at all, just prose.", provider=self.name, model=self.model)


class EmptyReplyProvider:
    """Returns syntactically valid JSON but with an empty/missing reply
    field, to exercise that specific invalid-output branch."""

    name = "empty-reply-stub"
    model = "v0"

    def complete(self, system_prompt, user_prompt):
        return LLMResponse(text='{"reply": "", "evidence_ids": [], "grounding_score": 0.5, "confidence": 0.5}',
                            provider=self.name, model=self.model)


# --- LLM provider selection --------------------------------------------------


def test_get_llm_provider_defaults_to_mock_with_no_env(monkeypatch):
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    provider = get_llm_provider()
    assert isinstance(provider, MockLLMProvider)
    assert provider.name == "mock"


def test_get_llm_provider_explicit_mock(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "mock")
    provider = get_llm_provider()
    assert isinstance(provider, MockLLMProvider)


def test_get_llm_provider_explicit_openai_without_key_raises(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(LLMConfigError):
        get_llm_provider("openai")


def test_get_llm_provider_unknown_provider_raises():
    with pytest.raises(LLMConfigError):
        get_llm_provider("not_a_real_provider")


def test_mock_provider_never_claims_to_be_real():
    provider = MockLLMProvider()
    response = provider.complete("system", "user")
    assert response.provider == "mock"


def test_mock_provider_is_deterministic():
    provider = MockLLMProvider()
    prompt = build_user_prompt("my order never arrived", _prediction(), [_evidence()])
    r1 = provider.complete("sys", prompt)
    r2 = provider.complete("sys", prompt)
    assert r1.text == r2.text


def test_mock_provider_differs_for_different_evidence():
    provider = MockLLMProvider()
    prompt_a = build_user_prompt("my order never arrived", _prediction(), [_evidence(similarity=0.9)])
    prompt_b = build_user_prompt("my order never arrived", _prediction(), [_evidence(similarity=0.2)])
    r1 = provider.complete("sys", prompt_a)
    r2 = provider.complete("sys", prompt_b)
    assert r1.text != r2.text


# --- Structured LLM output parsing (reply.py) --------------------------------


def test_generate_reply_parses_well_formed_mock_output():
    provider = MockLLMProvider()
    evidence = [_evidence()]
    result = generate_reply("my order never arrived", _prediction(), evidence, provider)

    assert result.parse_error is False
    assert isinstance(result.reply, str) and result.reply.strip()
    assert 0.0 <= result.grounding_score <= 1.0
    assert 0.0 <= result.confidence <= 1.0
    assert all(eid in {e.resolution_id for e in evidence} for eid in result.evidence_ids)


def test_generate_reply_handles_invalid_json_gracefully():
    result = generate_reply("my order never arrived", _prediction(), [_evidence()], EchoBrokenProvider())
    assert result.parse_error is True
    assert result.reply is None
    assert result.grounding_score == 0.0
    assert result.confidence == 0.0
    assert result.evidence_ids == []
    assert result.raw_llm_text == "not json at all, just prose."


def test_generate_reply_handles_empty_reply_field_as_parse_error():
    result = generate_reply("my order never arrived", _prediction(), [_evidence()], EmptyReplyProvider())
    assert result.parse_error is True
    assert result.reply is None


def test_generate_reply_strips_evidence_ids_not_actually_given():
    class HallucinatingProvider:
        name = "hallucinating-stub"
        model = "v0"

        def complete(self, system_prompt, user_prompt):
            return LLMResponse(
                text='{"reply": "We are on it!", "evidence_ids": ["res_1", "res_totally_made_up"], '
                     '"grounding_score": 0.8, "confidence": 0.8}',
                provider=self.name,
                model=self.model,
            )

    evidence = [_evidence(resolution_id="res_1")]
    result = generate_reply("my order never arrived", _prediction(), evidence, HallucinatingProvider())
    assert result.parse_error is False
    assert result.evidence_ids == ["res_1"]  # the made-up id is silently dropped, not trusted


def test_generate_reply_handles_fenced_json_block():
    class FencedProvider:
        name = "fenced-stub"
        model = "v0"

        def complete(self, system_prompt, user_prompt):
            return LLMResponse(
                text='```json\n{"reply": "Sorry about that!", "evidence_ids": [], '
                     '"grounding_score": 0.4, "confidence": 0.4}\n```',
                provider=self.name,
                model=self.model,
            )

    result = generate_reply("hello", _prediction(), [], FencedProvider())
    assert result.parse_error is False
    assert result.reply == "Sorry about that!"


def test_generate_reply_clamps_out_of_range_scores():
    class OutOfRangeProvider:
        name = "oor-stub"
        model = "v0"

        def complete(self, system_prompt, user_prompt):
            return LLMResponse(
                text='{"reply": "ok", "evidence_ids": [], "grounding_score": 5.0, "confidence": -3.0}',
                provider=self.name,
                model=self.model,
            )

    result = generate_reply("hello", _prediction(), [], OutOfRangeProvider())
    assert result.grounding_score == 1.0
    assert result.confidence == 0.0


# --- Full agent orchestration (run_agent) ------------------------------------


def test_run_agent_high_confidence_strong_evidence_auto_handles():
    classifier = StubClassifier(_prediction(intent="delivery_delay", confidence=0.95))
    searcher = StubSearcher([
        _evidence("res_1", 0.9, "delivery_delay"),
        _evidence("res_2", 0.88, "delivery_delay"),
    ])
    result = run_agent(
        "my order never arrived and its been a week",
        intent_classifier=classifier,
        searcher=searcher,
        llm_provider=MockLLMProvider(),
    )
    assert isinstance(result, AgentResult)
    assert result.decision == "auto_handle"
    assert result.reply is not None
    d = result.as_dict()
    assert set(d.keys()) == {
        "customer_message", "intent", "intent_confidence", "retrieved_evidence",
        "reply", "decision", "reason", "signals",
    }
    assert d["intent"] == "delivery_delay"
    assert len(d["retrieved_evidence"]) == 2
    for ev in d["retrieved_evidence"]:
        assert set(ev.keys()) == {
            "resolution_id", "customer_message", "brand_reply", "intent",
            "similarity_score", "conversation_id", "timestamp",
        }


def test_run_agent_missing_evidence_escalates():
    classifier = StubClassifier(_prediction(intent="delivery_delay", confidence=0.9))
    searcher = StubSearcher([])
    result = run_agent(
        "my order never arrived and its been a week",
        intent_classifier=classifier,
        searcher=searcher,
        llm_provider=MockLLMProvider(),
    )
    assert result.decision == "escalate"
    assert result.signals["evidence_count"] == 0


def test_run_agent_low_similarity_escalates_as_ood():
    classifier = StubClassifier(_prediction(intent="delivery_delay", confidence=0.9))
    searcher = StubSearcher([_evidence("res_1", similarity=0.05, intent="delivery_delay")])
    result = run_agent(
        "some totally unrelated novel message",
        intent_classifier=classifier,
        searcher=searcher,
        llm_provider=MockLLMProvider(),
    )
    assert result.decision == "escalate"
    assert result.signals["ood_flag"] is True


def test_run_agent_high_risk_intent_escalates():
    config = EscalationConfig(high_risk_intents=["billing_dispute"])
    classifier = StubClassifier(_prediction(intent="billing_dispute", confidence=0.95))
    searcher = StubSearcher([_evidence("res_1", 0.95, "billing_dispute")])
    result = run_agent(
        "i was charged twice for the same order",
        intent_classifier=classifier,
        searcher=searcher,
        llm_provider=MockLLMProvider(),
        escalation_config=config,
    )
    assert result.decision == "escalate"
    assert result.signals["risk_flag"] is True


def test_run_agent_ambiguous_intent_escalates():
    config = EscalationConfig(min_intent_confidence=0.5, ambiguous_margin=0.2)
    classifier = StubClassifier(_prediction(intent="delivery_delay", confidence=0.55))
    searcher = StubSearcher([_evidence("res_1", 0.9, "delivery_delay")])
    result = run_agent(
        "my order never arrived and its been a week",
        intent_classifier=classifier,
        searcher=searcher,
        llm_provider=MockLLMProvider(),
        escalation_config=config,
    )
    assert result.decision == "escalate"
    assert result.signals["ambiguous_intent"] is True


def test_run_agent_falls_back_to_unfiltered_search_when_intent_filter_empty():
    classifier = StubClassifier(_prediction(intent="rare_intent", confidence=0.9))
    # No evidence tagged "rare_intent", but plenty tagged "delivery_delay" —
    # the agent should fall back to an unfiltered search rather than
    # reporting zero evidence just because the predicted intent has no
    # corpus coverage.
    searcher = StubSearcher([_evidence("res_1", 0.9, "delivery_delay")])
    result = run_agent(
        "my order never arrived and its been a week",
        intent_classifier=classifier,
        searcher=searcher,
        llm_provider=MockLLMProvider(),
    )
    assert len(result.retrieved_evidence) == 1
    assert result.retrieved_evidence[0].intent == "delivery_delay"  # not silently relabeled


def test_run_agent_llm_parse_failure_forces_escalation():
    classifier = StubClassifier(_prediction(intent="delivery_delay", confidence=0.95))
    searcher = StubSearcher([_evidence("res_1", 0.95, "delivery_delay")])
    result = run_agent(
        "my order never arrived and its been a week",
        intent_classifier=classifier,
        searcher=searcher,
        llm_provider=EchoBrokenProvider(),
    )
    assert result.decision == "escalate"
    assert result.reply is None
    assert result.signals["llm_parse_error"] is True


def test_run_agent_uses_default_providers_when_not_given(monkeypatch):
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    classifier = StubClassifier(_prediction(intent="delivery_delay", confidence=0.95))
    searcher = StubSearcher([_evidence("res_1", 0.9, "delivery_delay")])
    # No llm_provider / escalation_config passed — must not raise even
    # with no environment configured at all.
    result = run_agent(
        "my order never arrived and its been a week",
        intent_classifier=classifier,
        searcher=searcher,
    )
    assert result.decision in ("auto_handle", "escalate")
