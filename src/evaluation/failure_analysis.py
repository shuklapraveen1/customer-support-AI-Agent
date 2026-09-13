"""Categorized failure analysis over evaluation predictions.

Buckets predictions into named failure categories, each carrying real
example ids (and short text snippets — see `example_snippets`) so a
reader can go inspect actual failures (`python -m
src.evaluation.inspect_cases`) rather than trust a summary number blindly.

Categories fall into three groups:

- **System-internal** (always computable, no gold label needed): a parse
  error, out-of-distribution retrieval, insufficient evidence volume, low
  grounding, judge-flagged hallucination.
- **Heuristic, text/signal-based** (always computable, but these are
  deliberately simple proxies, not trained classifiers — each one's
  `why_it_failed`/`hypothesis` text says so explicitly): ambiguous intent,
  multiple questions, account-specific issues, noisy/short tweets,
  sarcasm/anger, semantic retrieval mismatch, conflicting historical
  resolutions, outdated historical response.
- **Gold-dependent** (only computable over examples with the relevant
  gold label present — reports `eligible_count` alongside the failure
  count, so "0 failures" is distinguishable from "0 examples were even
  checkable"): intent misclassification, incorrect escalation.

`build_failure_analysis_report` / `save_failure_analysis` rank every
category by count and surface the top N (default 5) — "automatically
identify the top five failure modes." Each carries its count, percentage,
real examples, a mechanical explanation of why it failed, a hypothesis for
the underlying cause, and a potential fix. None of this is fabricated:
every count comes from the actual `predictions` passed in, and every
hypothesis/fix is written to be checked against real data once it exists,
not asserted as already-verified fact.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

from src.evaluation.metrics import is_labeled, parse_bool_label

LOW_GROUNDING_THRESHOLD = 0.5
LOW_JUDGE_SCORE_THRESHOLD = 2.0
SHORT_MESSAGE_MAX_WORDS = 4
STALE_EVIDENCE_DAYS = 180
ANGRY_KEYWORDS = [
    "ridiculous", "unacceptable", "worst", "never again", "furious",
    "outrageous", "disgusted", "pathetic", "terrible service", "scam",
]
ACCOUNT_SPECIFIC_PHRASES = ["my account", "my order", "my card", "my subscription", "my payment"]


@dataclass
class FailureCategory:
    name: str
    description: str
    why_it_failed: str
    hypothesis: str
    potential_fix: str
    example_ids: list[str] = field(default_factory=list)
    example_snippets: list[dict] = field(default_factory=list)
    eligible_count: Optional[int] = None  # None = "computed over all predictions passed in"

    @property
    def count(self) -> int:
        return len(self.example_ids)

    def percentage(self, denominator: int) -> Optional[float]:
        effective_denominator = self.eligible_count if self.eligible_count is not None else denominator
        if not effective_denominator:
            return None
        return round(100.0 * self.count / effective_denominator, 1)

    def as_dict(self, denominator: int, max_examples: int = 5) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "count": self.count,
            "percentage": self.percentage(denominator),
            "eligible_count": self.eligible_count,
            "why_it_failed": self.why_it_failed,
            "hypothesis": self.hypothesis,
            "potential_fix": self.potential_fix,
            "example_ids": self.example_ids,
            "real_examples": self.example_snippets[:max_examples],
        }


def _add(category: FailureCategory, pred: dict, example_id: str) -> None:
    category.example_ids.append(example_id)
    if len(category.example_snippets) < 10:
        category.example_snippets.append(
            {
                "example_id": example_id,
                "customer_message": pred.get("customer_message"),
                "predicted_intent": pred.get("predicted_intent"),
                "decision": pred.get("decision"),
                "reply": pred.get("reply"),
            }
        )


def _word_count(text: str) -> int:
    return len(text.split()) if text else 0


def _has_multiple_questions(text: str) -> bool:
    return bool(text) and text.count("?") >= 2


def _seems_angry(text: str) -> bool:
    if not text:
        return False
    lowered = text.lower()
    letters = sum(1 for c in text if c.isalpha())
    caps = sum(1 for c in text if c.isupper())
    caps_ratio = (caps / letters) if letters else 0.0
    keyword_hit = any(k in lowered for k in ANGRY_KEYWORDS)
    return keyword_hit or text.count("!") >= 3 or (letters > 5 and caps_ratio > 0.6)


def _mentions_account_specifics(text: str) -> bool:
    if not text:
        return False
    lowered = text.lower()
    has_phrase = any(p in lowered for p in ACCOUNT_SPECIFIC_PHRASES)
    has_number = any(ch.isdigit() for ch in text)
    return has_phrase and has_number


def _evidence_intents(evidence: list[dict]) -> list[str]:
    return [e.get("intent") for e in (evidence or []) if e.get("intent")]


def _is_conflicting_evidence(evidence: list[dict]) -> bool:
    intents = _evidence_intents(evidence)
    return len(set(intents)) > 1


def _parse_timestamp(value: object) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value))
    except ValueError:
        return None


def _corpus_max_timestamp(predictions: list[dict]) -> Optional[datetime]:
    timestamps = []
    for pred in predictions:
        for e in pred.get("retrieved_evidence") or []:
            ts = _parse_timestamp(e.get("timestamp"))
            if ts:
                timestamps.append(ts)
    return max(timestamps) if timestamps else None


def _is_outdated_evidence(evidence: list[dict], reference_time: Optional[datetime]) -> bool:
    if not evidence or reference_time is None:
        return False
    top_ts = _parse_timestamp(evidence[0].get("timestamp"))
    if top_ts is None:
        return False
    return (reference_time - top_ts).days > STALE_EVIDENCE_DAYS


def _new_categories() -> dict[str, FailureCategory]:
    return {
        # --- system-internal --------------------------------------------
        "llm_parse_error": FailureCategory(
            "llm_parse_error",
            "Reply generation did not produce valid structured output; the agent escalated as a result.",
            why_it_failed="The LLM's response couldn't be parsed as the required JSON shape (or was missing a usable 'reply' field).",
            hypothesis="With the mock provider this should be rare/zero by construction; with a real LLM, likely causes are the model adding prose around the JSON beyond what the parser tolerates, or truncation from a token limit.",
            potential_fix="Log the raw (unparsed) LLM text for every parse failure and inspect it directly; consider a stricter response-format API feature (e.g. OpenAI's JSON mode) instead of prompt-only instructions.",
        ),
        "insufficient_evidence": FailureCategory(
            "insufficient_evidence",
            "Zero historical resolutions were retrieved for this message at all.",
            why_it_failed="The retrieval corpus had nothing to return, either because the corpus is small or the predicted intent has no corpus coverage.",
            hypothesis="The TRAIN-split retrieval corpus in this sandbox is built from a tiny synthetic fixture; in a real deployment this would instead indicate the brand+intent combination is genuinely underrepresented in historical data.",
            potential_fix="Grow the TRAIN-split corpus, or broaden the retrieval query (e.g. drop the intent filter, already done automatically by run_agent's fallback) before concluding there's truly no precedent.",
        ),
        "ood_query": FailureCategory(
            "ood_query",
            "Evidence was retrieved, but its similarity to the query was below the out-of-distribution threshold.",
            why_it_failed="The closest historical match still isn't a good match — the message doesn't closely resemble anything the corpus has seen.",
            hypothesis="Either the corpus doesn't cover this kind of issue yet, or the embedder (HashingEmbedder in this sandbox — a lexical-overlap proxy, not real semantics) can't recognize a semantic match phrased with different words.",
            potential_fix="Re-run with a real semantic embedder (sentence-transformers) once available; separately, track OOD rate over time as a signal for corpus coverage gaps.",
        ),
        "low_grounding_reply": FailureCategory(
            "low_grounding_reply",
            f"A reply was drafted but its self-reported grounding_score was below {LOW_GROUNDING_THRESHOLD}.",
            why_it_failed="The drafting LLM itself signaled low confidence that its reply was well-supported by the given evidence.",
            hypothesis="Evidence was present but weak/tangential, so the model produced a generic reply and (correctly) scored it as such.",
            potential_fix="Confirm the escalation policy's min_grounding_score threshold is actually catching these (it should, by design) rather than papering over them with a plausible-sounding but ungrounded reply.",
        ),
        "hallucination": FailureCategory(
            "hallucination",
            f"The LLM judge scored hallucination_safety at or below {LOW_JUDGE_SCORE_THRESHOLD}.",
            why_it_failed="The judge flagged the reply as likely containing a fabricated specific (a policy, refund, tracking number, date, or claimed action not supported by evidence).",
            hypothesis="grounding_score was low but the drafting prompt didn't fully suppress specific-sounding language — models tend to default to concrete, confident phrasing even when instructed to hedge.",
            potential_fix="Tighten the reply prompt's instruction to explicitly generalize/hedge when grounding_score is low, and treat judge-flagged hallucination as its own hard escalation signal, not just a report-time observation.",
        ),
        # --- heuristic, text/signal-based ---------------------------------
        "ambiguous_intent": FailureCategory(
            "ambiguous_intent",
            "Intent confidence landed in the 'ambiguous' band (just above the auto-handle minimum) or was generally low.",
            why_it_failed="The classifier couldn't confidently discriminate between semantically similar intents for this message.",
            hypothesis="All three classifiers in this project are bootstrapped on 1-2 worked examples per intent from configs/intents.yaml (a placeholder taxonomy), not a real labeled training set — low confidence is expected until real training data exists.",
            potential_fix="Fit on a real, larger labeled training set once available; consider a confidence-calibration pass (e.g. temperature scaling) once real validation data exists.",
        ),
        "multiple_questions": FailureCategory(
            "multiple_questions",
            "The message appears to contain more than one distinct question (2+ question marks).",
            why_it_failed="Reply drafting is structured around one predicted intent and one retrieval query, so a multi-part message likely only gets one part addressed.",
            hypothesis="Multi-question messages are structurally under-served by a single-intent, single-query pipeline design.",
            potential_fix="Detect multi-question messages and either split them into sub-queries before drafting, or explicitly instruct the drafting LLM to address every question found in the message.",
        ),
        "account_specific_issue": FailureCategory(
            "account_specific_issue",
            "The message references specific account/order details (e.g. 'my order #1234') that imply the correct answer depends on account state the system can't see.",
            why_it_failed="No component in this pipeline has access to real account/order state — only the historical text corpus.",
            hypothesis="These cases are close to structurally impossible to resolve correctly with a text-only, no-account-access agent; they should escalate close to by definition, not merely due to low confidence.",
            potential_fix="Treat 'account-specific' as its own explicit high-risk category in configs/escalation.yaml (like high_risk_keywords) rather than relying on grounding/confidence to catch it indirectly.",
        ),
        "noisy_or_short_tweet": FailureCategory(
            "noisy_or_short_tweet",
            f"The message is very short ({SHORT_MESSAGE_MAX_WORDS} words or fewer).",
            why_it_failed="There's very little text signal for classification or retrieval to work with.",
            hypothesis="Short/noisy messages are inherently low-information — both the classifier and retrieval degrade when there's little to match against.",
            potential_fix="Route very short/noisy messages to an explicit clarifying-question flow instead of attempting classification/retrieval on them at all (missing_context already partially covers this in escalation.yaml).",
        ),
        "sarcasm_or_anger": FailureCategory(
            "sarcasm_or_anger",
            "The message's tone (via a crude keyword/punctuation/caps-ratio heuristic) suggests frustration or anger.",
            why_it_failed="An angry customer may need human tone-handling regardless of how factually 'answerable' the underlying question is.",
            hypothesis="Keyword/punctuation heuristics are a weak sentiment proxy that will both over- and under-flag real anger; and even correctly-flagged cases get no special handling from the current drafting prompt.",
            potential_fix="Add a dedicated tone/sentiment signal (ideally model-based, not keyword heuristics) as an explicit escalation input, independent of intent/grounding.",
        ),
        "semantic_retrieval_mismatch": FailureCategory(
            "semantic_retrieval_mismatch",
            "The retrieved evidence's own intent tags disagree with the predicted intent for this message.",
            why_it_failed="Classification and retrieval disagreed about what this message is even about.",
            hypothesis="The intent classifier and the retrieval embedder are two independently weak, independently bootstrapped signals in this sandbox — they have no reason to agree with each other on hard cases.",
            potential_fix="Use a shared/consistent representation for classification and retrieval (e.g. fine-tune one embedding for both) so the two signals are consistent by construction rather than independently noisy.",
        ),
        "conflicting_historical_resolutions": FailureCategory(
            "conflicting_historical_resolutions",
            "The top retrieved historical cases don't agree with each other about what kind of issue this is.",
            why_it_failed="Multiple near-equally-similar historical resolutions were retrieved with different intent tags — there's no single clear precedent.",
            hypothesis="The corpus may contain near-duplicate messages that were historically handled inconsistently (different agents, different times, or genuinely different underlying issues that happen to read similarly).",
            potential_fix="Surface conflicting evidence explicitly to a human reviewer rather than silently picking the single top match; consider clustering resolutions by outcome as well as message similarity.",
        ),
        "outdated_historical_response": FailureCategory(
            "outdated_historical_response",
            f"The single most similar historical case on record is more than {STALE_EVIDENCE_DAYS} days older than the most recent evidence timestamp seen in this run.",
            why_it_failed="The brand's actual policy for this kind of issue may have changed since the retrieved precedent was recorded.",
            hypothesis="Retrieval currently optimizes purely for similarity, with no explicit recency preference strong enough to prefer a slightly-less-similar but much fresher precedent.",
            potential_fix="Add an explicit recency decay or minimum-recency filter to retrieval ranking, or tag historical resolutions with a policy-version field once such metadata exists.",
        ),
        # --- gold-dependent ------------------------------------------------
        "intent_misclassified": FailureCategory(
            "intent_misclassified",
            "Predicted intent did not match gold_intent (only computable for labeled examples).",
            why_it_failed="The classifier's top prediction disagreed with the human-labeled correct intent.",
            hypothesis="Same root cause as ambiguous_intent: bootstrapped training data, not a real labeled training set.",
            potential_fix="Same as ambiguous_intent — a real training set is the primary lever here.",
            eligible_count=0,
        ),
        "incorrect_escalation": FailureCategory(
            "incorrect_escalation",
            "The system's auto-handle/escalate decision didn't match the human gold judgment (false auto-handle OR unnecessary escalation; only computable for labeled examples).",
            why_it_failed="The escalation policy's thresholds produced a decision a human labeler disagreed with.",
            hypothesis="configs/escalation.yaml's thresholds are untuned illustrative defaults (see the file's own header) — misses in either direction are expected until real DEV-split tuning happens.",
            potential_fix="Tune configs/escalation.yaml thresholds against DEV-split human judgments once they exist — never against golden/test, per this project's own leakage discipline.",
            eligible_count=0,
        ),
    }


def analyze_failures(predictions: list[dict]) -> list[FailureCategory]:
    """`predictions` is expected to be the same list of dicts written to
    `artifacts/evaluation/predictions.jsonl` — see `run_eval.py` for the
    exact shape. Each dict may or may not have gold_* fields depending on
    labeling progress.
    """
    categories = _new_categories()
    reference_time = _corpus_max_timestamp(predictions)

    for pred in predictions:
        example_id = pred.get("example_id", "unknown")
        signals = pred.get("signals") or {}
        message = pred.get("customer_message") or ""
        evidence = pred.get("retrieved_evidence") or []

        if signals.get("llm_parse_error"):
            _add(categories["llm_parse_error"], pred, example_id)
        if signals.get("evidence_count") == 0:
            _add(categories["insufficient_evidence"], pred, example_id)
        elif signals.get("ood_flag"):
            _add(categories["ood_query"], pred, example_id)
        if pred.get("reply") and pred.get("grounding_score") is not None and pred["grounding_score"] < LOW_GROUNDING_THRESHOLD:
            _add(categories["low_grounding_reply"], pred, example_id)

        judge_score = pred.get("judge_score")
        if judge_score and judge_score.get("hallucination_safety") is not None and judge_score["hallucination_safety"] <= LOW_JUDGE_SCORE_THRESHOLD:
            _add(categories["hallucination"], pred, example_id)

        if signals.get("ambiguous_intent") or (signals.get("intent_confidence") is not None and signals["intent_confidence"] < 0.3):
            _add(categories["ambiguous_intent"], pred, example_id)
        if _has_multiple_questions(message):
            _add(categories["multiple_questions"], pred, example_id)
        if _mentions_account_specifics(message):
            _add(categories["account_specific_issue"], pred, example_id)
        if _word_count(message) <= SHORT_MESSAGE_MAX_WORDS:
            _add(categories["noisy_or_short_tweet"], pred, example_id)
        if _seems_angry(message):
            _add(categories["sarcasm_or_anger"], pred, example_id)
        if evidence and signals.get("evidence_agreement") is not None and signals["evidence_agreement"] < 0.5:
            _add(categories["semantic_retrieval_mismatch"], pred, example_id)
        if _is_conflicting_evidence(evidence):
            _add(categories["conflicting_historical_resolutions"], pred, example_id)
        if _is_outdated_evidence(evidence, reference_time):
            _add(categories["outdated_historical_response"], pred, example_id)

        gold_intent = pred.get("gold_intent")
        if is_labeled(gold_intent):
            categories["intent_misclassified"].eligible_count += 1
            if pred.get("predicted_intent") != gold_intent:
                _add(categories["intent_misclassified"], pred, example_id)

        gold_should_escalate = parse_bool_label(pred.get("gold_should_escalate"))
        if gold_should_escalate is not None:
            decision = pred.get("decision")
            categories["incorrect_escalation"].eligible_count += 1
            if (decision == "auto_handle" and gold_should_escalate) or (decision == "escalate" and not gold_should_escalate):
                _add(categories["incorrect_escalation"], pred, example_id)

    return list(categories.values())


def build_failure_analysis_report(predictions: list[dict], top_n: int = 5) -> dict:
    """Rank every category by count and surface the top `top_n` —
    "automatically identify the top five failure modes." Zero-count
    categories are still listed under `all_categories` for transparency
    (so a reader can see what WASN'T a problem, not just what was), but
    never padded into `top_failure_modes`.
    """
    categories = analyze_failures(predictions)
    denominator = len(predictions)
    ranked = sorted(categories, key=lambda c: (-c.count, c.name))
    top = [c for c in ranked if c.count > 0][:top_n]

    return {
        "n_predictions_analyzed": denominator,
        "top_failure_modes": [c.as_dict(denominator) for c in top],
        "all_categories": [c.as_dict(denominator) for c in ranked],
    }


def save_failure_analysis(
    predictions: list[dict],
    path: str = "artifacts/evaluation/failure_analysis.json",
    top_n: int = 5,
) -> dict:
    report = build_failure_analysis_report(predictions, top_n=top_n)
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, default=str)
    return report
