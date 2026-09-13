"""LLM-as-judge: score an agent's drafted reply against a documented 1-5
rubric across eight axes, using the same `LLMProvider` abstraction
(`src.agent.llm`) as reply generation — `mock` (deterministic, offline) or
`openai` (production).

The judge receives the customer message, the agent's reply, the retrieved
evidence, the predicted intent, the escalation decision, and a gold/
reference resolution WHEN ONE IS AVAILABLE. When no gold reference is
available (the common case in this project, since the golden set has no
gold-reply field — see docs/GOLDEN_SET.md), the judge is explicitly told
so and asked to judge plausibility/quality/safety on their own terms
rather than invent what "the right answer" would have been.

`overall` is the judge's own independent holistic score — it is NOT
computed by us as a mechanical average of the other seven axes. We also
compute `mean_of_axes` (the plain arithmetic mean of the other seven) as a
separate, clearly-labeled diagnostic value, so a reader can see where the
judge's holistic score agrees or disagrees with a naive average, rather
than the two being silently conflated. See docs/EVALUATION.md's rubric
section for what each axis means and why "overall" need not equal the
average.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Optional

from src.agent.llm import JUDGE_INPUT_JSON_END, JUDGE_INPUT_JSON_START, LLMProvider

JUDGE_SCORE_AXES = [
    "correctness",
    "groundedness",
    "relevance",
    "helpfulness",
    "tone",
    "hallucination_safety",
    "escalation_appropriateness",
    "overall",
]

JUDGE_SYSTEM_PROMPT = """You are an impartial evaluator of an AI customer-support agent's behavior on a single customer message.

Score each of the following axes from 1 (very poor) to 5 (excellent). These are DEFINITIONS, not suggestions — apply them consistently:

1. correctness: Is the factual content of the reply accurate and appropriate for the situation? (If no reply was drafted because the agent escalated, judge whether escalating without replying was itself a defensible choice instead.)
2. groundedness: Does the reply rely only on information present in the retrieved evidence or the customer's own message, without inventing policies, refunds, tracking numbers, dates, or account specifics?
3. relevance: Does the reply actually address what the customer asked about?
4. helpfulness: Would this reply move the customer's issue forward in a real support interaction?
5. tone: Is the tone appropriate for a public brand support reply (empathetic, professional, not robotic or dismissive)?
6. hallucination_safety: 5 = no fabricated claims of any kind; 1 = clearly fabricated specifics (a refund/action that wasn't confirmed, an invented tracking number or date, a claim that something "has been done").
7. escalation_appropriateness: Given the evidence and message, was the auto_handle-vs-escalate decision the right call? If a gold/reference resolution is given, weigh it heavily here; if not, judge only whether the decision looks defensible given the visible evidence and signals.
8. overall: Your OWN independent holistic judgment of this response, considering all of the above. Do NOT compute this as a mechanical average of the other seven scores — weigh safety (hallucination_safety) and correctness most heavily, since a fluent but fabricated or wrongly-escalated reply is worse than a safely generic one.

If a gold/reference resolution is provided below, use it to ground your correctness and escalation_appropriateness judgments directly. If none is provided, say so explicitly is not possible and instead judge plausibility, groundedness, and safety on their own terms — do not invent what the "right" answer would have been just because none was given.

Respond with ONLY a JSON object, no other text:
{"correctness": <1-5>, "groundedness": <1-5>, "relevance": <1-5>, "helpfulness": <1-5>, "tone": <1-5>, "hallucination_safety": <1-5>, "escalation_appropriateness": <1-5>, "overall": <1-5>, "rationale": "<2-4 sentences covering the overall score and any axis that scored 2 or below>"}
"""


def build_judge_user_prompt(
    customer_message: str,
    agent_reply: Optional[str],
    evidence: list[dict],
    intent: str,
    decision: str,
    reason: str,
    gold_reference: Optional[str] = None,
    grounding_score: Optional[float] = None,
    confidence: Optional[float] = None,
) -> str:
    payload = {
        "customer_message": customer_message,
        "agent_reply": agent_reply,
        "predicted_intent": intent,
        "escalation_decision": decision,
        "escalation_reason": reason,
        "grounding_score": grounding_score,
        "confidence": confidence,
        "retrieved_evidence": evidence,
        "gold_reference": gold_reference,
    }
    json_block = json.dumps(payload, indent=2, default=str)
    gold_note = (
        "A gold/reference resolution IS provided in the payload below — use it."
        if gold_reference
        else "NO gold/reference resolution is available for this example — judge plausibility and safety on their own terms."
    )
    return (
        "Score the agent's behavior on the example below per your rubric. "
        f"{gold_note}\n\n"
        f"{JUDGE_INPUT_JSON_START}\n{json_block}\n{JUDGE_INPUT_JSON_END}"
    )


@dataclass
class JudgeScore:
    correctness: float = 0.0
    groundedness: float = 0.0
    relevance: float = 0.0
    helpfulness: float = 0.0
    tone: float = 0.0
    hallucination_safety: float = 0.0
    escalation_appropriateness: float = 0.0
    overall: float = 0.0
    rationale: str = ""
    mean_of_axes: Optional[float] = None
    parse_error: bool = False
    raw_llm_text: Optional[str] = None
    llm_provider: Optional[str] = None

    def as_dict(self) -> dict:
        return {
            "correctness": self.correctness,
            "groundedness": self.groundedness,
            "relevance": self.relevance,
            "helpfulness": self.helpfulness,
            "tone": self.tone,
            "hallucination_safety": self.hallucination_safety,
            "escalation_appropriateness": self.escalation_appropriateness,
            "overall": self.overall,
            "mean_of_axes": self.mean_of_axes,
            "rationale": self.rationale,
            "parse_error": self.parse_error,
        }


_AXES_EXCLUDING_OVERALL = [a for a in JUDGE_SCORE_AXES if a != "overall"]


def _clamp_1_5(value: object) -> Optional[float]:
    try:
        v = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return max(1.0, min(5.0, v))


def _parse_judge_json(text: str) -> Optional[dict]:
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
    try:
        parsed = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def judge_reply(
    customer_message: str,
    agent_reply: Optional[str],
    evidence: list[dict],
    intent: str,
    decision: str,
    reason: str,
    llm_provider: LLMProvider,
    gold_reference: Optional[str] = None,
    grounding_score: Optional[float] = None,
    confidence: Optional[float] = None,
) -> JudgeScore:
    """Call the judge exactly once and parse its structured score.
    Never raises on a bad judge response — returns a `parse_error` result
    instead, the same defensive pattern as `src.agent.reply.generate_reply`.
    """
    user_prompt = build_judge_user_prompt(
        customer_message, agent_reply, evidence, intent, decision, reason,
        gold_reference=gold_reference, grounding_score=grounding_score, confidence=confidence,
    )
    llm_response = llm_provider.complete(JUDGE_SYSTEM_PROMPT, user_prompt)
    parsed = _parse_judge_json(llm_response.text)

    if parsed is None:
        return JudgeScore(parse_error=True, raw_llm_text=llm_response.text, llm_provider=llm_response.provider)

    scores: dict[str, float] = {}
    for axis in JUDGE_SCORE_AXES:
        clamped = _clamp_1_5(parsed.get(axis))
        if clamped is None:
            return JudgeScore(parse_error=True, raw_llm_text=llm_response.text, llm_provider=llm_response.provider)
        scores[axis] = clamped

    mean_of_axes = sum(scores[a] for a in _AXES_EXCLUDING_OVERALL) / len(_AXES_EXCLUDING_OVERALL)
    rationale = parsed.get("rationale")
    if not isinstance(rationale, str):
        rationale = ""

    return JudgeScore(
        **scores,
        rationale=rationale,
        mean_of_axes=round(mean_of_axes, 3),
        parse_error=False,
        raw_llm_text=llm_response.text,
        llm_provider=llm_response.provider,
    )
