"""LLM provider abstraction: OpenAI (production) or a deterministic mock
(offline development, demos, and tests).

Provider selection (`get_llm_provider`) is driven by the `LLM_PROVIDER`,
`OPENAI_API_KEY`, and `OPENAI_MODEL` environment variables. With nothing
configured at all, the agent runs on the mock provider rather than failing
to start — that's what makes "if no API key exists, the system MUST still
run" true out of the box — but an *explicit* `LLM_PROVIDER=openai` with no
key still fails loudly (`LLMConfigError`) rather than silently swapping in
mock, because a misconfiguration should be visible, not papered over.

The mock provider is not a stand-in that imitates what a real LLM would
say. It builds its structured JSON output deterministically from the
prompt's embedded input data using plain, documented rules (see
`_mock_generate_reply_json`), and every `LLMResponse` it returns carries
`provider == "mock"` — nothing downstream can mistake it for a real
completion, and it must never be reported as one.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Optional, Protocol

# Delimiters wrapping the machine-readable JSON payload embedded in the
# user prompt built by src.agent.prompts (reply drafting) and
# src.evaluation.judge (LLM-as-judge scoring). Defined here (not in those
# modules) so the mock provider — the only thing that needs to parse them
# back out — owns them, and callers import them from here rather than the
# other way around. MockLLMProvider dispatches on whichever marker is
# present in the prompt, so one provider instance serves both use cases.
AGENT_INPUT_JSON_START = "<AGENT_INPUT_JSON>"
AGENT_INPUT_JSON_END = "</AGENT_INPUT_JSON>"
JUDGE_INPUT_JSON_START = "<JUDGE_INPUT_JSON>"
JUDGE_INPUT_JSON_END = "</JUDGE_INPUT_JSON>"


class LLMConfigError(RuntimeError):
    """Raised when the requested LLM provider can't actually be
    constructed — e.g. `LLM_PROVIDER=openai` with no `OPENAI_API_KEY`, or
    an unrecognized provider name."""


@dataclass(frozen=True)
class LLMResponse:
    text: str
    provider: str
    model: str


class LLMProvider(Protocol):
    name: str
    model: str

    def complete(self, system_prompt: str, user_prompt: str) -> LLMResponse: ...


def _extract_delimited_json(user_prompt: str, start_marker: str, end_marker: str) -> Optional[dict]:
    start = user_prompt.find(start_marker)
    end = user_prompt.find(end_marker)
    if start == -1 or end == -1 or end <= start:
        return None
    block = user_prompt[start + len(start_marker) : end].strip()
    try:
        parsed = json.loads(block)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _extract_agent_input(user_prompt: str) -> Optional[dict]:
    return _extract_delimited_json(user_prompt, AGENT_INPUT_JSON_START, AGENT_INPUT_JSON_END)


def _extract_judge_input(user_prompt: str) -> Optional[dict]:
    return _extract_delimited_json(user_prompt, JUDGE_INPUT_JSON_START, JUDGE_INPUT_JSON_END)


def _mock_generate_reply_json(user_prompt: str) -> str:
    """Deterministically build the same structured JSON shape a real
    completion is asked for, purely from the embedded input payload — same
    input always produces the same output, no randomness, no network.
    """
    payload = _extract_agent_input(user_prompt)
    if payload is None:
        # No parseable structured input at all — safe, deterministic
        # fallback that clearly signals "nothing to ground this in".
        return json.dumps(
            {
                "reply": "Thanks for reaching out — a member of our team will follow up shortly.",
                "evidence_ids": [],
                "grounding_score": 0.0,
                "confidence": 0.0,
            }
        )

    intent = str(payload.get("intent", "unknown"))
    evidence = payload.get("evidence") or []
    if not isinstance(evidence, list):
        evidence = []

    if not evidence:
        result = {
            "reply": (
                "Thanks for letting us know — we don't have a close match in our history for "
                "this yet, so we're flagging it for a specialist to take a closer look."
            ),
            "evidence_ids": [],
            "grounding_score": 0.0,
            "confidence": 0.15,
        }
        return json.dumps(result)

    # Evidence is expected to already be sorted by similarity (that's the
    # retrieval layer's job); the mock always grounds on the top match.
    top = evidence[0]
    try:
        similarity = float(top.get("similarity_score", 0.0))
    except (TypeError, ValueError):
        similarity = 0.0
    score = round(max(0.0, min(1.0, similarity)), 4)

    used_ids = [str(e["resolution_id"]) for e in evidence[:2] if isinstance(e, dict) and "resolution_id" in e]
    intent_phrase = intent.replace("_", " ")
    reply_text = (
        f"Thanks for reaching out about this {intent_phrase} — we've seen similar cases before "
        "and we're on it. We'll follow up with next steps shortly."
    )

    result = {
        "reply": reply_text,
        "evidence_ids": used_ids,
        "grounding_score": score,
        "confidence": score,
    }
    return json.dumps(result)


def _mock_generate_judge_json(user_prompt: str) -> str:
    """Deterministically build judge-shaped JSON from the embedded judge
    input payload. This is NOT a real semantic quality judgment — it's a
    simple, documented, bounded function of the numeric signals already
    present in the payload (grounding_score, confidence), used so
    `src.evaluation.judge` can be built, tested, and demonstrated without
    an API key. See docs/EVALUATION.md for this limitation in context.
    """
    payload = _extract_judge_input(user_prompt)
    if payload is None:
        return json.dumps(
            {
                "correctness": 3, "groundedness": 3, "relevance": 3, "helpfulness": 3,
                "tone": 3, "hallucination_safety": 3, "escalation_appropriateness": 3, "overall": 3,
                "rationale": "No parseable judge input was found in the prompt; scored neutrally by default.",
            }
        )

    reply = payload.get("agent_reply")
    decision = payload.get("escalation_decision", "escalate")
    grounding = payload.get("grounding_score")
    confidence = payload.get("confidence")
    gold_reference = payload.get("gold_reference")

    try:
        grounding_f = float(grounding) if grounding is not None else 0.0
    except (TypeError, ValueError):
        grounding_f = 0.0
    try:
        confidence_f = float(confidence) if confidence is not None else 0.0
    except (TypeError, ValueError):
        confidence_f = 0.0
    grounding_f = max(0.0, min(1.0, grounding_f))
    confidence_f = max(0.0, min(1.0, confidence_f))

    if not reply:
        scores = {
            "correctness": 3, "groundedness": 3, "relevance": 3,
            "helpfulness": 2, "tone": 3, "hallucination_safety": 5,
            "escalation_appropriateness": 4 if decision == "escalate" else 2,
            "overall": 3,
        }
        rationale = (
            "No reply was drafted (the agent escalated); reply-quality axes are scored "
            "neutrally since there's nothing to evaluate, and hallucination_safety is scored "
            "high because an absent reply cannot contain a fabricated claim."
        )
    else:
        base = round(1 + 4 * ((grounding_f + confidence_f) / 2))
        scores = {
            "correctness": base,
            "groundedness": round(1 + 4 * grounding_f),
            "relevance": base,
            "helpfulness": base,
            "tone": 4,
            "hallucination_safety": round(1 + 4 * grounding_f),
            "escalation_appropriateness": 4,
            "overall": base,
        }
        rationale = (
            f"Deterministic mock scoring derived from grounding_score={grounding_f:.2f} and "
            f"confidence={confidence_f:.2f}; this is NOT a real semantic judgment of the reply's "
            "actual content — see docs/EVALUATION.md."
        )

    if gold_reference:
        rationale += " A gold/reference resolution was provided, but the mock judge does not perform real semantic comparison against it."
    else:
        rationale += " No gold/reference resolution was available for this example."

    scores["rationale"] = rationale
    return json.dumps(scores)


class MockLLMProvider:
    """Deterministic, offline provider for dev/demo/tests. No network, no
    API key, no randomness — the same `(system_prompt, user_prompt)` always
    produces the same `LLMResponse.text`. Always reports `provider="mock"`.

    Dispatches on which delimited JSON block is present in `user_prompt`:
    `AGENT_INPUT_JSON_*` (reply drafting, from `src.agent.prompts`) or
    `JUDGE_INPUT_JSON_*` (judging, from `src.evaluation.judge`) — so one
    instance can serve both callers in the same evaluation run.
    """

    name = "mock"
    model = "mock-deterministic-v1"

    def complete(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        if JUDGE_INPUT_JSON_START in user_prompt:
            text = _mock_generate_judge_json(user_prompt)
        else:
            text = _mock_generate_reply_json(user_prompt)
        return LLMResponse(text=text, provider=self.name, model=self.model)


class OpenAIProvider:
    """Wraps the OpenAI chat completions API.

    Requires the `openai` package and a valid `OPENAI_API_KEY`; both are
    checked eagerly in `__init__` so a misconfiguration fails immediately
    and clearly, rather than surfacing as a confusing error deep inside
    reply generation.
    """

    name = "openai"

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        resolved_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not resolved_key:
            raise LLMConfigError(
                "LLM_PROVIDER=openai requires OPENAI_API_KEY to be set. Set it, or use "
                "LLM_PROVIDER=mock for offline development, demos, and tests."
            )
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise LLMConfigError(
                "The 'openai' package is not installed. Run `pip install openai`, or use "
                "LLM_PROVIDER=mock for offline development, demos, and tests."
            ) from exc

        self.model = model or os.environ.get("OPENAI_MODEL") or "gpt-4o-mini"
        self._client = OpenAI(api_key=resolved_key)

    def complete(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        response = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0,
        )
        text = response.choices[0].message.content or ""
        return LLMResponse(text=text, provider=self.name, model=self.model)


def get_llm_provider(provider_name: Optional[str] = None) -> LLMProvider:
    """Select and construct an LLM provider.

    Resolution order: explicit `provider_name` argument > `LLM_PROVIDER` env
    var > automatic default (`"openai"` if `OPENAI_API_KEY` is set, else
    `"mock"`).
    """
    name = (provider_name or os.environ.get("LLM_PROVIDER") or "").strip().lower()
    if not name:
        name = "openai" if os.environ.get("OPENAI_API_KEY") else "mock"

    if name == "mock":
        return MockLLMProvider()
    if name == "openai":
        return OpenAIProvider()
    raise LLMConfigError(f"Unknown LLM_PROVIDER '{name}'. Supported providers: 'openai', 'mock'.")
