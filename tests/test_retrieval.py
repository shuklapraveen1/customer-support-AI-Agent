from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

from src.config import Config, PathsConfig
from src.data.conversations import build_conversations, extract_resolution_pairs
from src.data.normalize import load_raw_csv, normalize_dataframe
from src.data.split import split_conversations
from src.intent.classifier import IntentTaxonomy
from src.intent.embedding import HashingEmbedder
from src.retrieval.index import (
    ResolutionRecord,
    build_resolution_corpus,
    load_corpus_parquet,
    save_corpus_parquet,
)
from src.retrieval.search import (
    ResolutionIndex,
    TfidfRetrievalBaseline,
    build_self_retrieval_eval_queries,
    evaluate_retrieval,
)

FIXTURE = Path(__file__).parent / "fixtures" / "twcs_sample.csv"


def _rec(
    resolution_id: str,
    customer_message: str,
    brand_reply: str = "Thanks, we'll look into it.",
    intent: str = "delivery_delay",
    conversation_id: str = "1",
    timestamp: datetime | None = None,
    brand: str = "BrandA_Support",
) -> ResolutionRecord:
    return ResolutionRecord(
        resolution_id=resolution_id,
        customer_message=customer_message,
        brand_reply=brand_reply,
        intent=intent,
        conversation_id=conversation_id,
        timestamp=timestamp,
        brand=brand,
    )


def _synthetic_corpus() -> list[ResolutionRecord]:
    base_time = datetime(2020, 1, 1)
    return [
        _rec("res_1", "my order never arrived and its been a week", "So sorry, DM us your order number.",
             intent="delivery_delay", conversation_id="c1", timestamp=base_time),
        _rec("res_2", "package is still not here after two weeks", "We'll escalate this for you.",
             intent="delivery_delay", conversation_id="c2", timestamp=base_time + timedelta(days=10)),
        _rec("res_3", "where is my delivery it never showed up", "Let us check the tracking for you.",
             intent="delivery_delay", conversation_id="c3", timestamp=base_time + timedelta(days=20)),
        _rec("res_4", "i was charged twice for the same order", "We'll refund the duplicate charge.",
             intent="billing_dispute", conversation_id="c4", timestamp=base_time + timedelta(days=5)),
        _rec("res_5", "please refund me for this purchase", "Refund has been processed.",
             intent="billing_dispute", conversation_id="c5", timestamp=base_time + timedelta(days=15)),
        _rec("res_6", "the app keeps crashing every time i open it", "Please try reinstalling the app.",
             intent="app_technical_issue", conversation_id="c6", timestamp=base_time + timedelta(days=25)),
        _rec("res_7", "login page gives an error every time", "We're looking into this bug.",
             intent="app_technical_issue", conversation_id="c7", timestamp=base_time + timedelta(days=30)),
    ]


# --- Index building ----------------------------------------------------------


def test_resolution_index_builds_from_records():
    records = _synthetic_corpus()
    index = ResolutionIndex.build(records, embedder=HashingEmbedder(dimension=128))
    assert index.vectors.shape[0] == len(records)
    assert index.backend_name in ("faiss", "sklearn")


def test_resolution_index_rejects_empty_corpus():
    with pytest.raises(ValueError):
        ResolutionIndex.build([], embedder=HashingEmbedder())


def test_resolution_index_forced_sklearn_backend():
    records = _synthetic_corpus()
    index = ResolutionIndex.build(records, embedder=HashingEmbedder(), prefer_faiss=False)
    assert index.backend_name == "sklearn"


# --- Search / retrieve --------------------------------------------------------


def test_retrieve_returns_all_required_fields():
    records = _synthetic_corpus()
    index = ResolutionIndex.build(records, embedder=HashingEmbedder(dimension=128))
    results = index.retrieve("my package has not arrived yet", top_k=3)

    assert len(results) > 0
    for r in results:
        assert r.resolution_id
        assert r.customer_message
        assert r.brand_reply
        assert r.intent
        assert isinstance(r.similarity_score, float)
        assert r.conversation_id
        # timestamp is not required to be non-None, but the attribute must
        # always be present and unmodified from the record.
        assert hasattr(r, "timestamp")


def test_retrieve_top_k_limits_result_count():
    records = _synthetic_corpus()
    index = ResolutionIndex.build(records, embedder=HashingEmbedder(dimension=128))

    results_1 = index.retrieve("my order never arrived", top_k=1)
    results_3 = index.retrieve("my order never arrived", top_k=3)
    results_all = index.retrieve("my order never arrived", top_k=100)

    assert len(results_1) == 1
    assert len(results_3) == 3
    assert len(results_all) == len(records)


def test_retrieve_rejects_non_positive_top_k():
    records = _synthetic_corpus()
    index = ResolutionIndex.build(records, embedder=HashingEmbedder())
    with pytest.raises(ValueError):
        index.retrieve("hello", top_k=0)


def test_retrieve_most_similar_result_is_topically_related():
    records = _synthetic_corpus()
    index = ResolutionIndex.build(records, embedder=HashingEmbedder(dimension=128))
    results = index.retrieve("my delivery never came and its late", top_k=1)
    assert results[0].intent == "delivery_delay"


def test_retrieve_low_similarity_is_represented_not_hidden():
    """A query sharing no vocabulary with the corpus should score low, and
    that low score must show up in the result rather than being clamped,
    hidden, or replaced with a fabricated-looking high number."""
    records = _synthetic_corpus()
    index = ResolutionIndex.build(records, embedder=HashingEmbedder(dimension=128))

    close_results = index.retrieve("my order never arrived and its been a week", top_k=1)
    unrelated_results = index.retrieve("zzzz qqqq xylophone bagpipes marmalade", top_k=1)

    assert close_results[0].similarity_score > unrelated_results[0].similarity_score
    # An unrelated bag-of-words query against a hashing embedder should not
    # score anywhere near a near-identical match.
    assert unrelated_results[0].similarity_score < 0.5
    assert close_results[0].similarity_score > 0.9


def test_retrieve_with_intent_filter_only_returns_that_intent():
    records = _synthetic_corpus()
    index = ResolutionIndex.build(records, embedder=HashingEmbedder(dimension=128))
    results = index.retrieve("some issue with my account", intent="billing_dispute", top_k=5)

    assert len(results) > 0
    assert all(r.intent == "billing_dispute" for r in results)


def test_retrieve_with_intent_filter_unknown_intent_returns_empty():
    records = _synthetic_corpus()
    index = ResolutionIndex.build(records, embedder=HashingEmbedder(dimension=128))
    results = index.retrieve("anything", intent="not_a_real_intent", top_k=5)
    assert results == []


def test_retrieve_recency_breaks_exact_similarity_ties():
    """Two records with identical text (and thus identical similarity to
    any query) should be ordered with the more recent one first."""
    older = _rec("res_old", "identical text here", conversation_id="c_old", timestamp=datetime(2020, 1, 1))
    newer = _rec("res_new", "identical text here", conversation_id="c_new", timestamp=datetime(2022, 1, 1))
    index = ResolutionIndex.build([older, newer], embedder=HashingEmbedder(dimension=64))

    results = index.retrieve("identical text here", top_k=2)
    assert results[0].resolution_id == "res_new"
    assert results[1].resolution_id == "res_old"


def test_retrieve_missing_timestamp_is_reported_as_none_not_hidden():
    records = [_rec("res_no_ts", "some message with no timestamp", timestamp=None)]
    index = ResolutionIndex.build(records, embedder=HashingEmbedder())
    results = index.retrieve("some message with no timestamp", top_k=1)
    assert results[0].timestamp is None


# --- Index save/load -----------------------------------------------------


def test_index_save_and_load_round_trip(tmp_path):
    records = _synthetic_corpus()
    embedder = HashingEmbedder(dimension=64)
    index = ResolutionIndex.build(records, embedder=embedder)
    index.save(str(tmp_path / "index"))

    loaded = ResolutionIndex.load(str(tmp_path / "index"), records=records, embedder=embedder)
    assert np.allclose(loaded.vectors, index.vectors)

    original_results = index.retrieve("my order never arrived", top_k=3)
    loaded_results = loaded.retrieve("my order never arrived", top_k=3)
    assert [r.resolution_id for r in original_results] == [r.resolution_id for r in loaded_results]


def test_index_load_rejects_mismatched_record_count(tmp_path):
    records = _synthetic_corpus()
    embedder = HashingEmbedder(dimension=64)
    index = ResolutionIndex.build(records, embedder=embedder)
    index.save(str(tmp_path / "index"))

    with pytest.raises(ValueError):
        ResolutionIndex.load(str(tmp_path / "index"), records=records[:2], embedder=embedder)


# --- Corpus parquet round trip -----------------------------------------------


def test_corpus_parquet_round_trip(tmp_path):
    records = _synthetic_corpus()
    out_path = tmp_path / "resolutions.parquet"
    save_corpus_parquet(records, str(out_path))
    loaded = load_corpus_parquet(str(out_path))

    assert len(loaded) == len(records)
    assert {r.resolution_id for r in loaded} == {r.resolution_id for r in records}
    original_by_id = {r.resolution_id: r for r in records}
    for r in loaded:
        original = original_by_id[r.resolution_id]
        assert r.customer_message == original.customer_message
        assert r.brand_reply == original.brand_reply
        assert r.timestamp == original.timestamp


# --- TF-IDF retrieval baseline ------------------------------------------------


def test_tfidf_baseline_retrieves_and_respects_top_k():
    records = _synthetic_corpus()
    baseline = TfidfRetrievalBaseline(records)
    results = baseline.retrieve("my order never arrived", top_k=2)
    assert len(results) == 2
    assert results[0].intent == "delivery_delay"


def test_tfidf_baseline_intent_filtering():
    records = _synthetic_corpus()
    baseline = TfidfRetrievalBaseline(records)
    results = baseline.retrieve("some technical problem", intent="app_technical_issue", top_k=5)
    assert all(r.intent == "app_technical_issue" for r in results)


def test_tfidf_baseline_rejects_empty_corpus():
    with pytest.raises(ValueError):
        TfidfRetrievalBaseline([])


# --- Retrieval evaluation (proxy Recall@k / MRR) -----------------------------


def test_build_self_retrieval_eval_queries_pairs_with_gold_ids():
    records = _synthetic_corpus()
    queries = build_self_retrieval_eval_queries(records, seed=42)
    gold_ids = {r.resolution_id for r in records}
    assert len(queries) > 0
    for query_text, gold_id in queries:
        assert query_text.strip() != ""
        assert gold_id in gold_ids


def test_evaluate_retrieval_semantic_index_recalls_perturbed_originals():
    records = _synthetic_corpus() * 3  # give the proxy eval enough queries to be meaningful
    records = [
        _rec(f"{r.resolution_id}_{i}", r.customer_message, r.brand_reply, r.intent, f"{r.conversation_id}_{i}", r.timestamp, r.brand)
        for i, r in enumerate(records)
    ]
    index = ResolutionIndex.build(records, embedder=HashingEmbedder(dimension=256))
    eval_queries = build_self_retrieval_eval_queries(records, seed=42, drop_word_prob=0.2)

    report = evaluate_retrieval(index, eval_queries, top_k=5)
    assert 0.0 <= report.recall_at_1 <= report.recall_at_3 <= report.recall_at_5 <= 1.0
    assert 0.0 <= report.mrr <= 1.0
    assert report.n_queries == len(eval_queries)
    assert "proxy" in report.method.lower() or "PROXY" in report.method
    # A lightly-perturbed near-duplicate should usually still find its own
    # original with a decent hashing embedder — a real, computed result,
    # not an assumed one.
    assert report.recall_at_5 > 0.3


def test_evaluate_retrieval_rejects_top_k_below_5():
    records = _synthetic_corpus()
    index = ResolutionIndex.build(records, embedder=HashingEmbedder())
    queries = build_self_retrieval_eval_queries(records, seed=42)
    with pytest.raises(ValueError):
        evaluate_retrieval(index, queries, top_k=3)


def test_evaluate_retrieval_rejects_empty_queries():
    records = _synthetic_corpus()
    index = ResolutionIndex.build(records, embedder=HashingEmbedder())
    with pytest.raises(ValueError):
        evaluate_retrieval(index, [], top_k=5)


# --- Train/test leakage is impossible ----------------------------------------


def test_build_resolution_corpus_only_ever_uses_train_split(tmp_path):
    config = Config(
        paths=PathsConfig(
            raw_csv=str(FIXTURE),
            processed_dir=str(tmp_path / "processed"),
            splits_dir=str(tmp_path / "splits"),
        )
    )

    # Independently reconstruct the full set of resolution pairs (train +
    # dev + test) using the same lower-level primitives, so we can assert
    # the corpus is a strict subset of train and, crucially, contains none
    # of the dev/test pairs.
    df = load_raw_csv(str(FIXTURE))
    tweets, _ = normalize_dataframe(df)
    conversations = build_conversations(tweets)
    pairs = extract_resolution_pairs(conversations)
    conversation_ids = [c.conversation_id for c in conversations]
    assignment = split_conversations(conversation_ids, seed=config.split.seed)

    dev_and_test_ids = set(assignment.dev) | set(assignment.test)
    all_brandA_pairs = [p for p in pairs if p.brand == "BrandA_Support"]
    assert any(p.conversation_id in dev_and_test_ids for p in all_brandA_pairs), (
        "test setup assumption failed: expected at least one BrandA_Support "
        "resolution pair to fall in dev/test for this fixture+seed"
    )

    corpus = build_resolution_corpus(config, "BrandA_Support")
    corpus_conversation_ids = {r.conversation_id for r in corpus}

    assert corpus_conversation_ids.issubset(set(assignment.train))
    assert corpus_conversation_ids.isdisjoint(dev_and_test_ids)


def test_build_resolution_corpus_resolution_ids_are_deterministic(tmp_path):
    config = Config(
        paths=PathsConfig(
            raw_csv=str(FIXTURE),
            processed_dir=str(tmp_path / "processed"),
            splits_dir=str(tmp_path / "splits"),
        )
    )
    corpus_1 = build_resolution_corpus(config, "BrandB_Care")
    corpus_2 = build_resolution_corpus(config, "BrandB_Care")
    assert [r.resolution_id for r in corpus_1] == [r.resolution_id for r in corpus_2]


def test_build_resolution_corpus_tags_intent_when_taxonomy_given(tmp_path):
    config = Config(
        paths=PathsConfig(
            raw_csv=str(FIXTURE),
            processed_dir=str(tmp_path / "processed"),
            splits_dir=str(tmp_path / "splits"),
        )
    )
    raw_taxonomy = {
        "unknown_label": "other",
        "intents": [
            {
                "id": "app_technical_issue",
                "name": "App issue",
                "description": "d",
                "examples": ["the app keeps crashing on login"],
            },
            {
                "id": "delivery_delay",
                "name": "Delivery delay",
                "description": "d",
                "examples": ["my order never arrived please help"],
            },
        ],
    }
    taxonomy = IntentTaxonomy.from_dict(raw_taxonomy)
    corpus = build_resolution_corpus(config, "BrandB_Care", taxonomy=taxonomy)
    assert len(corpus) >= 1
    for record in corpus:
        assert record.intent in taxonomy.allowed_labels


def test_build_resolution_corpus_without_taxonomy_tags_unknown(tmp_path):
    config = Config(
        paths=PathsConfig(
            raw_csv=str(FIXTURE),
            processed_dir=str(tmp_path / "processed"),
            splits_dir=str(tmp_path / "splits"),
        )
    )
    corpus = build_resolution_corpus(config, "BrandB_Care", taxonomy=None)
    assert all(r.intent == "unknown" for r in corpus)
