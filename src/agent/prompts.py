"""Prompt construction for reply generation.

The user prompt embeds a machine-readable JSON payload (delimited by
`AGENT_INPUT_JSON_START` / `AGENT_INPUT_JSON_END`, defined in
`src.agent.llm`) alongside natural-language instructions. A real LLM reads
the instructions and the embedded data together as ordinary prompt context;
`MockLLMProvider` parses just the JSON block back out — both consume
exactly the same prompt, so nothing about reply generation branches on
which provider is in use.
"""

from __future__ import annotations

import json

from src.agent.llm import AGENT_INPUT_JSON_END, AGENT_INPUT_JSON_START
from src.intent.classifier import IntentPrediction
from src.retrieval.search import RetrievalResult

SYSTEM_PROMPT = """You are a customer support reply drafter for a brand on Twitter/X.

Rules you must follow, with no exceptions:
- Use ONLY the historical evidence provided to you as grounding. Do not invent policies, refunds, tracking numbers, dates, or account details that are not present in the evidence or the customer's message.
- Never claim that an action (refund issued, order cancelled, ticket escalated, etc.) has already been performed. You are drafting a reply, not performing an action.
- Do not expose or quote internal evidence verbatim, and do not mention "historical evidence", resolution ids, similarity scores, or any other internal system detail to the customer.
- Do not blindly copy a historical reply word-for-word — synthesize a concise, on-brand response in your own words, informed by the pattern of past resolutions rather than any single one of them.
- If the evidence is weak, sparse, or contradictory, keep the reply generic and safe (acknowledge the issue and say a specialist will follow up) and reflect that with a LOWER grounding_score/confidence rather than guessing at specifics.
- Keep the reply concise and in the tone of the historical brand replies (informal, empathetic, appropriate length for a public Twitter/X reply).

You must respond with ONLY a single JSON object and nothing else — no preamble, no markdown fencing, no explanation — matching exactly this shape:
{"reply": "<the drafted reply text>", "evidence_ids": ["<resolution_id>", ...], "grounding_score": <number 0.0-1.0>, "confidence": <number 0.0-1.0>}

- "evidence_ids" must only contain resolution_id values that were given to you in the evidence list below; never invent one.
- "grounding_score" reflects how directly the reply follows from the given evidence (1.0 = closely grounded in specific historical cases, 0.0 = no usable evidence to ground it in).
- "confidence" reflects your overall confidence that this reply is an appropriate, safe response to send to the customer as-is.
"""


def build_user_prompt(
    customer_message: str,
    intent_prediction: IntentPrediction,
    evidence: list[RetrievalResult],
) -> str:
    payload = {
        "customer_message": customer_message,
        "intent": intent_prediction.intent,
        "intent_confidence": intent_prediction.confidence,
        "evidence": [
            {
                "resolution_id": e.resolution_id,
                "historical_customer_message": e.customer_message,
                "historical_brand_reply": e.brand_reply,
                "intent": e.intent,
                "similarity_score": e.similarity_score,
                "timestamp": e.timestamp.isoformat() if e.timestamp else None,
            }
            for e in evidence
        ],
    }
    json_block = json.dumps(payload, indent=2)
    return (
        "Draft a reply to the customer message below, using the historical evidence as "
        "grounding. Respond with ONLY the JSON object described in your system instructions.\n\n"
        f"{AGENT_INPUT_JSON_START}\n{json_block}\n{AGENT_INPUT_JSON_END}"
    )
