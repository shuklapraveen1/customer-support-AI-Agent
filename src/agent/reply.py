"""Reply generation: build a prompt from retrieved evidence, call the LLM
provider, and parse its structured output into a validated `ReplyResult`.

A malformed or incomplete LLM response is a first-class, expected outcome
here (`parse_error=True`, every score zeroed) rather than an exception that
crashes the whole agent — `src.agent.escalation` is expected to treat a
parse failure as an automatic reason to escalate rather than send an
unvalidated reply.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Optional

from src.agent.llm import LLMProvider
from src.agent.prompts import SYSTEM_PROMPT, build_user_prompt
from src.intent.classifier import IntentPrediction
from src.retrieval.search import RetrievalResult


@dataclass
class ReplyResult:
    reply: Optional[str]
    evidence_ids: list[str] = field(default_factory=list)
    grounding_score: float = 0.0
    confidence: float = 0.0
    parse_error: bool = False
    raw_llm_text: Optional[str] = None
    llm_provider: Optional[str] = None
    llm_model: Optional[str] = None

    def as_dict(self) -> dict:
        """The structured result shape the assignment specifies —
        internal-only fields (`parse_error`, `raw_llm_text`, provider/model)
        are available as attributes but deliberately excluded here."""
        return {
            "reply": self.reply,
            "evidence_ids": self.evidence_ids,
            "grounding_score": self.grounding_score,
            "confidence": self.confidence,
        }


def _clamp01(value: object) -> float:
    try:
        v = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, v))


def _parse_llm_json(text: str) -> Optional[dict]:
    """Extract and parse a JSON object from the LLM's raw text.

    Tolerates a fenced ```json code block or stray leading/trailing prose
    around the object, since real models don't always follow "JSON only"
    perfectly — but does NOT attempt to repair truncated or syntactically
    broken JSON. That is treated as a genuine parse failure, not something
    to silently patch over.
    """
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text[:4].lower() == "json":
            text = text[4:]
        text = text.strip()

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        return None

    candidate = text[start : end + 1]
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _parse_error_result(llm_response) -> ReplyResult:
    return ReplyResult(
        reply=None,
        evidence_ids=[],
        grounding_score=0.0,
        confidence=0.0,
        parse_error=True,
        raw_llm_text=llm_response.text,
        llm_provider=llm_response.provider,
        llm_model=llm_response.model,
    )


def generate_reply(
    customer_message: str,
    intent_prediction: IntentPrediction,
    evidence: list[RetrievalResult],
    llm_provider: LLMProvider,
) -> ReplyResult:
    """Draft a reply grounded in `evidence`, calling `llm_provider` exactly
    once. Never raises on a bad LLM response — returns a `parse_error`
    result instead so the caller can decide what to do (escalate).
    """
    user_prompt = build_user_prompt(customer_message, intent_prediction, evidence)
    llm_response = llm_provider.complete(SYSTEM_PROMPT, user_prompt)

    parsed = _parse_llm_json(llm_response.text)
    if parsed is None:
        return _parse_error_result(llm_response)

    reply_text = parsed.get("reply")
    if not isinstance(reply_text, str) or not reply_text.strip():
        return _parse_error_result(llm_response)

    valid_evidence_ids = {e.resolution_id for e in evidence}
    raw_evidence_ids = parsed.get("evidence_ids")
    if not isinstance(raw_evidence_ids, list):
        raw_evidence_ids = []
    # Defensive: never trust an evidence id the LLM invented that wasn't
    # actually part of what it was given.
    evidence_ids = [str(e) for e in raw_evidence_ids if str(e) in valid_evidence_ids]

    return ReplyResult(
        reply=reply_text.strip(),
        evidence_ids=evidence_ids,
        grounding_score=_clamp01(parsed.get("grounding_score", 0.0)),
        confidence=_clamp01(parsed.get("confidence", 0.0)),
        parse_error=False,
        raw_llm_text=llm_response.text,
        llm_provider=llm_response.provider,
        llm_model=llm_response.model,
    )
