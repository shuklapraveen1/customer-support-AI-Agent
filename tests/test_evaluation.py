import csv
import json
import os
import sys
from pathlib import Path

import pytest

from src.agent.llm import MockLLMProvider
from src.evaluation.failure_analysis import analyze_failures, build_failure_analysis_report, save_failure_analysis
from src.evaluation.judge import JudgeScore, judge_reply
from src.evaluation.judge_agreement import (
    compute_agreement,
    create_judge_validation_template,
    load_judge_validation_rows,
)
from src.evaluation.metrics import (
    RatioMetric,
    SafeAutomationRow,
    automation_coverage,
    escalation_precision,
    escalation_recall,
    false_auto_handle_rate,
    is_labeled,
    parse_bool_label,
    parse_reply_quality_label,
    safe_automation_coverage,
    safe_automation_coverage_over_labeled_subset,
)
from src.evaluation.retrieval_metrics import evaluate_retrieval_for_report
from src.intent.embedding import HashingEmbedder
from src.retrieval.index import ResolutionRecord
from src.retrieval.search import ResolutionIndex

REPO_ROOT = Path(__file__).parent.parent
FIXTURE = Path(__file__).parent / "fixtures" / "twcs_sample.csv"


# --- metrics.py: label parsing -----------------------------------------


def test_is_labeled():
    assert is_labeled("delivery_delay") is True
    assert is_labeled("UNLABELLED") is False
    assert is_labeled("unlabelled") is False
    assert is_labeled("") is False
    assert is_labeled(None) is False
    assert is_labeled("   ") is False


def test_parse_bool_label():
    assert parse_bool_label("true") is True
    assert parse_bool_label("YES") is True
    assert parse_bool_label("false") is False
    assert parse_bool_label("no") is False
    assert parse_bool_label("UNLABELLED") is None
    assert parse_bool_label("") is None
    assert parse_bool_label("maybe") is None


def test_parse_reply_quality_label():
    score, acceptable = parse_reply_quality_label("4")
    assert score == 4.0 and acceptable is True

    score, acceptable = parse_reply_quality_label("2")
    assert score == 2.0 and acceptable is False

    score, acceptable = parse_reply_quality_label("UNLABELLED")
    assert score is None and acceptable is None

    score, acceptable = parse_reply_quality_label("not a number")
    assert score is None and acceptable is None


# --- metrics.py: RatioMetric ---------------------------------------------


def test_ratio_metric_formats_with_numerator_and_denominator():
    m = RatioMetric("safe automation coverage", 124, 200)
    assert m.value == pytest.approx(0.62)
    assert m.formatted() == "62% safe automation coverage (124/200)"
    assert "124" in m.as_dict()["formatted"] and "200" in m.as_dict()["formatted"]


def test_ratio_metric_zero_denominator_is_na_not_zero():
    m = RatioMetric("escalation precision", 0, 0, note="no labeled examples")
    assert m.value is None
    assert "N/A" in m.formatted()
    assert "0/0" in m.formatted()


# --- metrics.py: escalation metrics --------------------------------------


def test_escalation_precision_recall_only_over_labeled_examples():
    gold = [True, False, None, True]
    pred_escalate = [True, True, True, False]
    precision = escalation_precision(gold, pred_escalate)
    # labeled pairs: (True,True) (False,True) (True,False) -- index 2 excluded (gold None)
    # predicted_pos: (True,True),(False,True) -> tp=1, denom=2
    assert precision.numerator == 1
    assert precision.denominator == 2

    recall = escalation_recall(gold, pred_escalate)
    # gold_pos: (True,True) idx0, (True,False) idx3 -> tp=1 (only idx0), denom=2
    assert recall.numerator == 1
    assert recall.denominator == 2


def test_false_auto_handle_rate():
    gold = [True, False, True, False]
    pred_escalate = [False, False, False, True]
    # auto_handled (pred_escalate False): idx0 gold True(bad), idx1 gold False(ok), idx2 gold True(bad)
    rate = false_auto_handle_rate(gold, pred_escalate)
    assert rate.numerator == 2
    assert rate.denominator == 3


def test_automation_coverage_needs_no_gold():
    pred_escalate = [True, False, False, False]
    cov = automation_coverage(pred_escalate)
    assert cov.numerator == 3
    assert cov.denominator == 4


# --- metrics.py: safe automation coverage --------------------------------


def _row(example_id, decision, pred_intent, gold_intent, gold_escalate, acceptable):
    return SafeAutomationRow(
        example_id=example_id,
        predicted_decision=decision,
        predicted_intent=pred_intent,
        gold_intent=gold_intent,
        gold_should_escalate=gold_escalate,
        reply_quality_acceptable=acceptable,
    )


def test_safe_automation_coverage_counts_only_fully_correct_labeled_rows():
    rows = [
        _row("a", "auto_handle", "delivery_delay", "delivery_delay", False, True),   # satisfies
        _row("b", "auto_handle", "delivery_delay", "billing_dispute", False, True),  # wrong intent
        _row("c", "auto_handle", "delivery_delay", "delivery_delay", True, True),    # gold says should've escalated
        _row("d", "escalate", "delivery_delay", "delivery_delay", False, True),      # not auto-handled
        _row("e", "auto_handle", "delivery_delay", "delivery_delay", False, False),  # reply not acceptable
        _row("f", "auto_handle", "delivery_delay", None, None, None),                # unlabeled
    ]
    full = safe_automation_coverage(rows, total_golden_examples=200)
    assert full.numerator == 1
    assert full.denominator == 200

    labeled = safe_automation_coverage_over_labeled_subset(rows)
    assert labeled.numerator == 1
    assert labeled.denominator == 5  # a,b,c,d,e are labeled; f is not


def test_safe_automation_coverage_all_unlabeled_gives_zero_over_zero():
    rows = [_row("a", "auto_handle", "x", None, None, None)]
    labeled = safe_automation_coverage_over_labeled_subset(rows)
    assert labeled.numerator == 0
    assert labeled.denominator == 0
    assert labeled.value is None


# --- retrieval_metrics.py -------------------------------------------------


def test_evaluate_retrieval_for_report_raises_on_too_few_records():
    records = [
        ResolutionRecord(f"res_{i}", f"message {i}", "reply", "delivery_delay", f"c{i}", None, "BrandA")
        for i in range(3)
    ]
    index = ResolutionIndex.build(records, embedder=HashingEmbedder())
    with pytest.raises(ValueError):
        evaluate_retrieval_for_report(index, records)


def test_evaluate_retrieval_for_report_computes_with_enough_records():
    records = [
        ResolutionRecord(f"res_{i}", f"my order number {i} never arrived please help", "sorry", "delivery_delay", f"c{i}", None, "BrandA")
        for i in range(10)
    ]
    index = ResolutionIndex.build(records, embedder=HashingEmbedder(dimension=128))
    report = evaluate_retrieval_for_report(index, records, seed=42)
    assert 0.0 <= report.recall_at_5 <= 1.0
    assert 0.0 <= report.mrr <= 1.0


# --- judge.py --------------------------------------------------------------


def test_judge_reply_with_mock_provider_no_reply():
    score = judge_reply(
        customer_message="help",
        agent_reply=None,
        evidence=[],
        intent="delivery_delay",
        decision="escalate",
        reason="insufficient evidence",
        llm_provider=MockLLMProvider(),
    )
    assert score.parse_error is False
    assert score.hallucination_safety == 5.0  # mock: absent reply can't hallucinate
    assert score.mean_of_axes is not None


def test_judge_reply_with_mock_provider_with_reply_is_deterministic():
    kwargs = dict(
        customer_message="my order never arrived",
        agent_reply="So sorry about that, we're looking into it.",
        evidence=[{"resolution_id": "res_1", "similarity_score": 0.9}],
        intent="delivery_delay",
        decision="auto_handle",
        reason="all signals passed",
        grounding_score=0.9,
        confidence=0.9,
    )
    s1 = judge_reply(llm_provider=MockLLMProvider(), **kwargs)
    s2 = judge_reply(llm_provider=MockLLMProvider(), **kwargs)
    assert s1.as_dict() == s2.as_dict()
    assert 1.0 <= s1.overall <= 5.0
    assert s1.parse_error is False


def test_judge_reply_handles_invalid_json():
    class BrokenProvider:
        name = "broken"
        model = "v0"

        def complete(self, system_prompt, user_prompt):
            from src.agent.llm import LLMResponse

            return LLMResponse(text="not json", provider=self.name, model=self.model)

    score = judge_reply(
        customer_message="hi", agent_reply="hello", evidence=[], intent="x", decision="auto_handle",
        reason="r", llm_provider=BrokenProvider(),
    )
    assert score.parse_error is True


def test_judge_reply_clamps_out_of_range_scores():
    class OutOfRangeProvider:
        name = "oor"
        model = "v0"

        def complete(self, system_prompt, user_prompt):
            from src.agent.llm import LLMResponse

            payload = {
                "correctness": 10, "groundedness": -2, "relevance": 3, "helpfulness": 3,
                "tone": 3, "hallucination_safety": 3, "escalation_appropriateness": 3,
                "overall": 3, "rationale": "test",
            }
            return LLMResponse(text=json.dumps(payload), provider=self.name, model=self.model)

    score = judge_reply(
        customer_message="hi", agent_reply="hello", evidence=[], intent="x", decision="auto_handle",
        reason="r", llm_provider=OutOfRangeProvider(),
    )
    assert score.correctness == 5.0
    assert score.groundedness == 1.0
    assert score.parse_error is False


def test_judge_reply_mean_of_axes_not_same_object_as_overall():
    """Sanity check for the 'do not simply average without explaining the
    rubric' requirement: mean_of_axes is a distinct, separately computed
    value from the judge's own 'overall', even though for the mock they can
    coincide numerically in simple cases."""
    class DivergentProvider:
        name = "divergent"
        model = "v0"

        def complete(self, system_prompt, user_prompt):
            from src.agent.llm import LLMResponse

            payload = {
                "correctness": 5, "groundedness": 5, "relevance": 5, "helpfulness": 5,
                "tone": 5, "hallucination_safety": 5, "escalation_appropriateness": 5,
                "overall": 1,  # judge holistically disagrees with a naive average
                "rationale": "Despite high individual scores, this reply is unsafe overall.",
            }
            return LLMResponse(text=json.dumps(payload), provider=self.name, model=self.model)

    score = judge_reply(
        customer_message="hi", agent_reply="hello", evidence=[], intent="x", decision="auto_handle",
        reason="r", llm_provider=DivergentProvider(),
    )
    assert score.overall == 1.0
    assert score.mean_of_axes == 5.0
    assert score.overall != score.mean_of_axes


# --- judge_agreement.py -----------------------------------------------------


def test_compute_agreement_pending_when_no_file():
    report = compute_agreement([])
    assert report.status == "pending_human_labels"
    assert "HUMAN VALIDATION PENDING" in report.message


def test_compute_agreement_pending_when_all_unlabelled():
    rows = [
        {"example_id": "a", "human_score": "UNLABELLED", "judge_score": "3"},
        {"example_id": "b", "human_score": "", "judge_score": "4"},
    ]
    report = compute_agreement(rows)
    assert report.status == "pending_human_labels"
    assert report.n_labeled == 0


def test_compute_agreement_computes_real_stats_when_labeled():
    rows = [
        {"example_id": "a", "human_score": "4", "judge_score": "4"},
        {"example_id": "b", "human_score": "2", "judge_score": "3"},
        {"example_id": "c", "human_score": "5", "judge_score": "5"},
        {"example_id": "d", "human_score": "1", "judge_score": "2"},
    ]
    report = compute_agreement(rows)
    assert report.status == "computed"
    assert report.n_labeled == 4
    assert report.exact_agreement == pytest.approx(2 / 4)
    assert report.mean_absolute_difference == pytest.approx((0 + 1 + 0 + 1) / 4)
    assert report.pearson_correlation is not None
    assert -1.0 <= report.pearson_correlation <= 1.0


def test_create_judge_validation_template_writes_unlabelled_human_fields(tmp_path):
    predictions = [
        {
            "example_id": f"golden_{i:04d}",
            "customer_message": f"message {i}",
            "reply": f"reply {i}",
            "retrieved_evidence": [{"resolution_id": "res_1", "similarity_score": 0.8}],
            "judge_score": {"overall": 4.0},
        }
        for i in range(5)
    ]
    out_path = tmp_path / "judge_validation.csv"
    n_written = create_judge_validation_template(predictions, path=str(out_path), n=3, seed=42)
    assert n_written == 3

    rows = load_judge_validation_rows(str(out_path))
    assert len(rows) == 3
    for row in rows:
        assert row["human_score"] == "UNLABELLED"
        assert row["human_notes"] == "UNLABELLED"
        assert row["judge_score"] == "4.0"


def test_create_judge_validation_template_skips_predictions_without_reply(tmp_path):
    predictions = [
        {"example_id": "a", "customer_message": "m", "reply": None, "retrieved_evidence": [], "judge_score": {"overall": 3}},
        {"example_id": "b", "customer_message": "m", "reply": "hi", "retrieved_evidence": [], "judge_score": {"overall": 3}},
    ]
    out_path = tmp_path / "judge_validation.csv"
    n_written = create_judge_validation_template(predictions, path=str(out_path), n=10, seed=1)
    assert n_written == 1


# --- failure_analysis.py -----------------------------------------------------


def test_analyze_failures_system_internal_categories_need_no_gold():
    predictions = [
        {"example_id": "a", "signals": {"llm_parse_error": True, "ood_flag": False, "evidence_count": 1}, "reply": None, "grounding_score": None, "judge_score": None},
        {"example_id": "b", "signals": {"llm_parse_error": False, "ood_flag": True, "evidence_count": 1}, "reply": None, "grounding_score": None, "judge_score": None},
        {"example_id": "c", "signals": {"llm_parse_error": False, "ood_flag": False, "evidence_count": 1}, "reply": "hi", "grounding_score": 0.2, "judge_score": None},
    ]
    categories = {c.name: c for c in analyze_failures(predictions)}
    assert categories["llm_parse_error"].example_ids == ["a"]
    assert categories["ood_query"].example_ids == ["b"]
    assert categories["low_grounding_reply"].example_ids == ["c"]


def test_analyze_failures_judge_flagged_categories():
    predictions = [
        {
            "example_id": "a", "signals": {}, "reply": "hi", "grounding_score": 0.9,
            "judge_score": {"hallucination_safety": 1, "overall": 1},
        },
        {
            "example_id": "b", "signals": {}, "reply": "hi", "grounding_score": 0.9,
            "judge_score": {"hallucination_safety": 5, "overall": 5},
        },
    ]
    categories = {c.name: c for c in analyze_failures(predictions)}
    assert categories["hallucination"].example_ids == ["a"]


def test_analyze_failures_gold_dependent_categories_only_count_labeled():
    predictions = [
        {
            "example_id": "a", "signals": {}, "reply": "hi", "grounding_score": 0.9, "judge_score": None,
            "gold_intent": "delivery_delay", "predicted_intent": "billing_dispute",
            "gold_should_escalate": "UNLABELLED", "decision": "auto_handle",
        },
        {
            "example_id": "b", "signals": {}, "reply": "hi", "grounding_score": 0.9, "judge_score": None,
            "gold_intent": "UNLABELLED", "predicted_intent": "delivery_delay",
            "gold_should_escalate": "true", "decision": "auto_handle",
        },
        {
            "example_id": "c", "signals": {}, "reply": "hi", "grounding_score": 0.9, "judge_score": None,
            "gold_intent": "UNLABELLED", "predicted_intent": "delivery_delay",
            "gold_should_escalate": "false", "decision": "escalate",
        },
    ]
    categories = {c.name: c for c in analyze_failures(predictions)}

    assert categories["intent_misclassified"].eligible_count == 1
    assert categories["intent_misclassified"].example_ids == ["a"]

    # incorrect_escalation covers both false-auto-handle (b) and
    # unnecessary-escalation (c) under one Phase-6-named category.
    assert categories["incorrect_escalation"].eligible_count == 2
    assert set(categories["incorrect_escalation"].example_ids) == {"b", "c"}


def test_analyze_failures_heuristic_categories():
    predictions = [
        {"example_id": "multi_q", "signals": {}, "customer_message": "Where is my order? And why was I charged twice?", "reply": "x", "grounding_score": 0.9, "judge_score": None, "retrieved_evidence": []},
        {"example_id": "account", "signals": {}, "customer_message": "my order #12345 is missing an item", "reply": "x", "grounding_score": 0.9, "judge_score": None, "retrieved_evidence": []},
        {"example_id": "short", "signals": {}, "customer_message": "help me", "reply": "x", "grounding_score": 0.9, "judge_score": None, "retrieved_evidence": []},
        {"example_id": "angry", "signals": {}, "customer_message": "This is absolutely ridiculous!!! Worst service ever!!!", "reply": "x", "grounding_score": 0.9, "judge_score": None, "retrieved_evidence": []},
        {"example_id": "calm", "signals": {}, "customer_message": "Could you please tell me my order status today", "reply": "x", "grounding_score": 0.9, "judge_score": None, "retrieved_evidence": []},
    ]
    categories = {c.name: c for c in analyze_failures(predictions)}

    assert "multi_q" in categories["multiple_questions"].example_ids
    assert "calm" not in categories["multiple_questions"].example_ids

    assert "account" in categories["account_specific_issue"].example_ids
    assert "calm" not in categories["account_specific_issue"].example_ids

    assert "short" in categories["noisy_or_short_tweet"].example_ids
    assert "calm" not in categories["noisy_or_short_tweet"].example_ids

    assert "angry" in categories["sarcasm_or_anger"].example_ids
    assert "calm" not in categories["sarcasm_or_anger"].example_ids


def test_analyze_failures_semantic_mismatch_and_conflicting_evidence():
    predictions = [
        {
            "example_id": "mismatch", "signals": {"evidence_agreement": 0.0}, "customer_message": "some message",
            "reply": "x", "grounding_score": 0.9, "judge_score": None,
            "retrieved_evidence": [{"intent": "billing_dispute", "timestamp": None}],
        },
        {
            "example_id": "conflict", "signals": {"evidence_agreement": 0.5}, "customer_message": "some other message",
            "reply": "x", "grounding_score": 0.9, "judge_score": None,
            "retrieved_evidence": [
                {"intent": "delivery_delay", "timestamp": None},
                {"intent": "billing_dispute", "timestamp": None},
            ],
        },
        {
            "example_id": "clean", "signals": {"evidence_agreement": 1.0}, "customer_message": "another message",
            "reply": "x", "grounding_score": 0.9, "judge_score": None,
            "retrieved_evidence": [{"intent": "delivery_delay", "timestamp": None}],
        },
    ]
    categories = {c.name: c for c in analyze_failures(predictions)}
    assert "mismatch" in categories["semantic_retrieval_mismatch"].example_ids
    assert "clean" not in categories["semantic_retrieval_mismatch"].example_ids
    assert "conflict" in categories["conflicting_historical_resolutions"].example_ids
    assert "clean" not in categories["conflicting_historical_resolutions"].example_ids


def test_analyze_failures_outdated_evidence():
    predictions = [
        {
            "example_id": "stale", "signals": {}, "customer_message": "some message", "reply": "x",
            "grounding_score": 0.9, "judge_score": None,
            "retrieved_evidence": [{"intent": "delivery_delay", "timestamp": "2020-01-01T00:00:00"}],
        },
        {
            "example_id": "fresh", "signals": {}, "customer_message": "some message", "reply": "x",
            "grounding_score": 0.9, "judge_score": None,
            "retrieved_evidence": [{"intent": "delivery_delay", "timestamp": "2021-06-01T00:00:00"}],
        },
    ]
    categories = {c.name: c for c in analyze_failures(predictions)}
    assert "stale" in categories["outdated_historical_response"].example_ids
    assert "fresh" not in categories["outdated_historical_response"].example_ids


def test_build_failure_analysis_report_ranks_top_n():
    predictions = [
        {"example_id": f"short_{i}", "signals": {}, "customer_message": "hi", "reply": "x", "grounding_score": 0.9, "judge_score": None, "retrieved_evidence": []}
        for i in range(5)
    ] + [
        {"example_id": "angry_1", "signals": {}, "customer_message": "This is absolutely ridiculous and completely unacceptable service from you!!!", "reply": "x", "grounding_score": 0.9, "judge_score": None, "retrieved_evidence": []},
    ]
    report = build_failure_analysis_report(predictions, top_n=2)
    assert report["n_predictions_analyzed"] == 6
    assert len(report["top_failure_modes"]) == 2
    # the short-message category (5 hits) should outrank the angry category (1 hit)
    assert report["top_failure_modes"][0]["name"] == "noisy_or_short_tweet"
    assert report["top_failure_modes"][0]["count"] == 5
    assert report["top_failure_modes"][0]["percentage"] == pytest.approx(83.3, abs=0.1)
    assert len(report["top_failure_modes"][0]["real_examples"]) > 0
    assert all("hypothesis" in m and "potential_fix" in m for m in report["top_failure_modes"])


def test_save_failure_analysis_writes_json(tmp_path):
    predictions = [
        {"example_id": "a", "signals": {}, "customer_message": "hi", "reply": "x", "grounding_score": 0.9, "judge_score": None, "retrieved_evidence": []},
    ]
    out_path = tmp_path / "failure_analysis.json"
    save_failure_analysis(predictions, path=str(out_path), top_n=5)
    assert out_path.exists()
    with open(out_path) as fh:
        data = json.load(fh)
    assert "top_failure_modes" in data
    assert "all_categories" in data


# --- run_eval.py: leakage checks + a minimal end-to-end smoke test ----------


def test_run_leakage_checks_detects_train_overlap(tmp_path):
    from src.config import Config, PathsConfig
    from src.evaluation.run_eval import run_leakage_checks

    config = Config(
        paths=PathsConfig(
            raw_csv=str(FIXTURE), processed_dir=str(tmp_path / "processed"), splits_dir=str(tmp_path / "splits")
        )
    )
    # Use a conversation id we know from the fixture split (see test_split.py /
    # test_intent.py precedent) — conversation "4" lands in TRAIN for seed=42.
    golden_rows = [{"conversation_id": "4", "customer_message": "x"}]
    results = run_leakage_checks(config, golden_rows, corpus_records=[])
    train_check = next(r for r in results if r.name == "golden_conversations_not_in_train_split")
    assert train_check.passed is False


def test_run_leakage_checks_passes_for_dev_test_conversations(tmp_path):
    from src.config import Config, PathsConfig
    from src.evaluation.run_eval import run_leakage_checks

    config = Config(
        paths=PathsConfig(
            raw_csv=str(FIXTURE), processed_dir=str(tmp_path / "processed"), splits_dir=str(tmp_path / "splits")
        )
    )
    # conversation "1" lands in DEV for seed=42 (see test_intent.py precedent)
    golden_rows = [{"conversation_id": "1", "customer_message": "x"}]
    results = run_leakage_checks(config, golden_rows, corpus_records=[])
    train_check = next(r for r in results if r.name == "golden_conversations_not_in_train_split")
    assert train_check.passed is True


def test_run_eval_end_to_end_smoke(tmp_path, monkeypatch):
    """Full pipeline smoke test: builds a tiny golden set + corpus in an
    isolated temp directory and runs `run_eval.main()` against it, checking
    every required artifact is produced and the process doesn't crash on a
    fully-unlabeled golden set."""
    from src.evaluation import run_eval

    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    # Minimal config pointing at the real fixture csv.
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        f"""
paths:
  raw_csv: {FIXTURE}
  processed_dir: {tmp_path / "data" / "processed"}
  splits_dir: {tmp_path / "data" / "splits"}
split:
  train: 0.7
  dev: 0.15
  test: 0.15
  seed: 42
"""
    )

    golden_dir = tmp_path / "data" / "golden"
    golden_dir.mkdir(parents=True)
    with open(golden_dir / "golden_set.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "example_id", "conversation_id", "customer_message", "gold_intent",
                "gold_should_escalate", "gold_reason", "gold_reply_quality", "notes",
            ],
        )
        writer.writeheader()
        # conversation "1" is in DEV for this fixture+seed — never TRAIN.
        writer.writerow(
            {
                "example_id": "golden_0001", "conversation_id": "1",
                "customer_message": "My order #123 never arrived, please help!",
                "gold_intent": "UNLABELLED", "gold_should_escalate": "UNLABELLED",
                "gold_reason": "UNLABELLED", "gold_reply_quality": "UNLABELLED", "notes": "",
            }
        )

    sys.argv = [
        "run_eval.py",
        "--config", str(config_path),
        "--intents-config", str(REPO_ROOT / "configs" / "intents.yaml"),
        "--escalation-config", str(REPO_ROOT / "configs" / "escalation.yaml"),
        "--golden-set", str(golden_dir / "golden_set.csv"),
        "--output-dir", str(tmp_path / "artifacts" / "evaluation"),
    ]
    run_eval.main()

    out_dir = tmp_path / "artifacts" / "evaluation"
    assert (out_dir / "predictions.jsonl").exists()
    assert (out_dir / "results.json").exists()
    assert (out_dir / "results.csv").exists()
    assert (out_dir / "confusion_matrix.png").exists()
    assert (out_dir / "report.json").exists()

    with open(out_dir / "report.json") as fh:
        report = json.load(fh)
    assert "safe automation coverage" in report["headline_formatted"]
    assert report["human_judge_agreement"]["status"] == "pending_human_labels"

    with open(out_dir / "predictions.jsonl") as fh:
        lines = fh.readlines()
    assert len(lines) == 3  # 1 golden example x 3 systems
    for line in lines:
        pred = json.loads(line)
        assert pred["conversation_id"] == "1"
        for ev in pred["retrieved_evidence"]:
            assert ev["conversation_id"] != "1"  # no self-leakage
