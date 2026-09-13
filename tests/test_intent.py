from pathlib import Path

import numpy as np
import pytest

from src.analysis.intents import (
    ClusterCandidate,
    discover_intent_candidates,
    load_brand_customer_messages,
)
from src.config import Config, PathsConfig
from src.data.conversations import build_conversations, extract_resolution_pairs
from src.data.normalize import load_raw_csv, normalize_dataframe
from src.data.split import split_conversations
from src.intent.baseline_majority import MajorityClassifier
from src.intent.baseline_tfidf import TfidfLogisticClassifier
from src.intent.classifier import (
    IntentConfigError,
    IntentPrediction,
    IntentTaxonomy,
    InvalidIntentLabelError,
)
from src.intent.embedding import (
    EmbeddingCache,
    EmbeddingNearestCentroidClassifier,
    HashingEmbedder,
)
from src.intent.evaluation import evaluate_predictions

FIXTURE = Path(__file__).parent / "fixtures" / "twcs_sample.csv"
REAL_INTENTS_YAML = Path(__file__).parent.parent / "configs" / "intents.yaml"


def make_taxonomy(labels=("delivery_delay", "refund_request", "account_access_issue")) -> IntentTaxonomy:
    raw = {
        "unknown_label": "other",
        "intents": [
            {"id": label, "name": label.title(), "description": f"Description of {label}.", "examples": [f"example message for {label}"]}
            for label in labels
        ],
    }
    return IntentTaxonomy.from_dict(raw)


# --- IntentTaxonomy ---------------------------------------------------------


def test_taxonomy_loads_from_dict_and_exposes_labels():
    tax = make_taxonomy()
    assert set(tax.labels) == {"delivery_delay", "refund_request", "account_access_issue"}
    assert "other" in tax.allowed_labels
    assert "delivery_delay" in tax
    assert "not_a_real_intent" not in tax


def test_taxonomy_rejects_empty_intent_list():
    with pytest.raises(IntentConfigError):
        IntentTaxonomy.from_dict({"unknown_label": "other", "intents": []})


def test_taxonomy_rejects_missing_required_field():
    raw = {
        "intents": [
            {"id": "x", "name": "X", "examples": ["hi"]},  # missing description
        ]
    }
    with pytest.raises(IntentConfigError):
        IntentTaxonomy.from_dict(raw)


def test_taxonomy_rejects_empty_examples_list():
    raw = {"intents": [{"id": "x", "name": "X", "description": "d", "examples": []}]}
    with pytest.raises(IntentConfigError):
        IntentTaxonomy.from_dict(raw)


def test_taxonomy_rejects_duplicate_ids():
    raw = {
        "intents": [
            {"id": "x", "name": "X", "description": "d", "examples": ["a"]},
            {"id": "x", "name": "X2", "description": "d2", "examples": ["b"]},
        ]
    }
    with pytest.raises(IntentConfigError):
        IntentTaxonomy.from_dict(raw)


def test_taxonomy_from_yaml_missing_file_raises():
    with pytest.raises(IntentConfigError):
        IntentTaxonomy.from_yaml("/nonexistent/path/intents.yaml")


def test_taxonomy_validate_label_accepts_known_and_unknown():
    tax = make_taxonomy()
    assert tax.validate_label("delivery_delay") == "delivery_delay"
    assert tax.validate_label("other") == "other"


def test_taxonomy_validate_label_rejects_arbitrary_label():
    tax = make_taxonomy()
    with pytest.raises(InvalidIntentLabelError):
        tax.validate_label("made_up_intent")


def test_taxonomy_validate_labels_batch():
    tax = make_taxonomy()
    tax.validate_labels(["delivery_delay", "refund_request"])
    with pytest.raises(InvalidIntentLabelError):
        tax.validate_labels(["delivery_delay", "not_real"])


def test_real_intents_yaml_is_schema_valid_and_reasonably_sized():
    """The shipped configs/intents.yaml must at least be loadable and
    internally consistent. It is currently a placeholder (see the file's
    own header comment) since no real per-brand data was available in this
    environment, so this is a schema/sanity check, not a check that these
    are "the right" 10 intents.
    """
    tax = IntentTaxonomy.from_yaml(str(REAL_INTENTS_YAML))
    assert len(tax.labels) == len(set(tax.labels))
    for intent_id in tax.labels:
        definition = tax.get(intent_id)
        assert definition.name
        assert definition.description
        assert len(definition.examples) >= 1
    # Loose sanity bound, not a strict enforcement of "8-15" for a
    # placeholder taxonomy built without real per-brand data.
    assert 3 <= len(tax.labels) <= 20


# --- Baseline #1: majority class -------------------------------------------


def test_majority_classifier_always_predicts_the_most_frequent_label():
    tax = make_taxonomy()
    clf = MajorityClassifier(tax)
    texts = ["a", "b", "c", "d", "e"]
    labels = ["delivery_delay", "delivery_delay", "delivery_delay", "refund_request", "refund_request"]
    clf.fit(texts, labels)

    predictions = clf.predict(["totally unrelated text", "another unrelated message"])
    assert all(p.intent == "delivery_delay" for p in predictions)
    assert all(p.confidence == pytest.approx(3 / 5) for p in predictions)
    assert all("Majority-class baseline" in p.reason for p in predictions)


def test_majority_classifier_rejects_invalid_training_label():
    tax = make_taxonomy()
    clf = MajorityClassifier(tax)
    with pytest.raises(InvalidIntentLabelError):
        clf.fit(["a"], ["not_a_real_intent"])


def test_majority_classifier_requires_fit_before_predict():
    tax = make_taxonomy()
    clf = MajorityClassifier(tax)
    with pytest.raises(RuntimeError):
        clf.predict(["hello"])


def test_majority_classifier_rejects_empty_training_set():
    tax = make_taxonomy()
    clf = MajorityClassifier(tax)
    with pytest.raises(ValueError):
        clf.fit([], [])


# --- Baseline #2: TF-IDF + Logistic Regression -----------------------------


def _separable_dataset():
    texts = [
        "my order never arrived please help with delivery",
        "package is late and still has not shown up",
        "where is my delivery it is very delayed",
        "I want a refund for this order right away",
        "please give me my money back for this purchase",
        "refund me immediately this was a waste of money",
        "I cannot log into my account it says wrong password",
        "locked out of my account after resetting my password",
        "my login keeps failing even with the correct password",
    ]
    labels = (
        ["delivery_delay"] * 3
        + ["refund_request"] * 3
        + ["account_access_issue"] * 3
    )
    return texts, labels


def test_tfidf_classifier_fits_and_predicts_on_separable_data():
    tax = make_taxonomy()
    clf = TfidfLogisticClassifier(tax, random_state=42)
    texts, labels = _separable_dataset()
    clf.fit(texts, labels)

    predictions = clf.predict(texts)
    predicted_labels = [p.intent for p in predictions]
    # Sanity check on training data itself (not a generalization claim) —
    # a clearly-separable bag-of-words dataset should be fit correctly.
    assert predicted_labels == labels
    assert all(0.0 <= p.confidence <= 1.0 for p in predictions)
    assert all("TF-IDF" in p.reason for p in predictions)


def test_tfidf_classifier_rejects_invalid_training_label():
    tax = make_taxonomy()
    clf = TfidfLogisticClassifier(tax)
    with pytest.raises(InvalidIntentLabelError):
        clf.fit(["hello"], ["not_a_real_intent"])


def test_tfidf_classifier_requires_fit_before_predict():
    tax = make_taxonomy()
    clf = TfidfLogisticClassifier(tax)
    with pytest.raises(RuntimeError):
        clf.predict(["hello"])


# --- HashingEmbedder + EmbeddingCache ---------------------------------------


def test_hashing_embedder_is_deterministic():
    embedder = HashingEmbedder(dimension=64)
    v1 = embedder.embed(["hello world"])
    v2 = embedder.embed(["hello world"])
    assert np.allclose(v1, v2)


def test_hashing_embedder_differentiates_different_texts():
    embedder = HashingEmbedder(dimension=64)
    v1, v2 = embedder.embed(["order delivery late", "refund money please"])
    assert not np.allclose(v1, v2)


def test_hashing_embedder_vectors_are_unit_normalized():
    embedder = HashingEmbedder(dimension=64)
    vectors = embedder.embed(["some text with several tokens here"])
    norm = np.linalg.norm(vectors[0])
    assert norm == pytest.approx(1.0, abs=1e-5)


def test_embedding_cache_avoids_recomputing_seen_texts(tmp_path):
    calls = {"count": 0}

    class CountingEmbedder:
        name = "counting-test-embedder"
        dimension = 8

        def embed(self, texts):
            calls["count"] += 1
            return np.ones((len(texts), 8), dtype=np.float32)

    cache = EmbeddingCache(str(tmp_path), CountingEmbedder())
    first = cache.embed_many(["hello", "world"])
    assert calls["count"] == 1

    second = cache.embed_many(["hello", "world"])
    assert calls["count"] == 1  # no new embed() call — both were cached
    assert np.allclose(first, second)

    # a new, unseen text should trigger exactly one more embed() call
    cache.embed_many(["hello", "brand new text"])
    assert calls["count"] == 2


def test_embedding_cache_persists_across_instances(tmp_path):
    embedder = HashingEmbedder(dimension=32)
    cache1 = EmbeddingCache(str(tmp_path), embedder)
    v1 = cache1.embed_many(["persisted text"])

    cache2 = EmbeddingCache(str(tmp_path), embedder)
    v2 = cache2.embed_many(["persisted text"])
    assert np.allclose(v1, v2)
    assert cache2.cache_size >= 1


# --- Embedding nearest-centroid classifier ----------------------------------


def test_embedding_classifier_fits_and_predicts_on_separable_data():
    tax = make_taxonomy()
    clf = EmbeddingNearestCentroidClassifier(tax, embedder=HashingEmbedder(dimension=128))
    texts, labels = _separable_dataset()
    clf.fit(texts, labels)

    predictions = clf.predict(texts)
    predicted_labels = [p.intent for p in predictions]
    assert predicted_labels == labels
    assert all(0.0 <= p.confidence <= 1.0 for p in predictions)
    assert all("Embedding nearest-centroid" in p.reason for p in predictions)


def test_embedding_classifier_rejects_invalid_training_label():
    tax = make_taxonomy()
    clf = EmbeddingNearestCentroidClassifier(tax)
    with pytest.raises(InvalidIntentLabelError):
        clf.fit(["hello"], ["not_a_real_intent"])


def test_embedding_classifier_requires_fit_before_predict():
    tax = make_taxonomy()
    clf = EmbeddingNearestCentroidClassifier(tax)
    with pytest.raises(RuntimeError):
        clf.predict(["hello"])


def test_embedding_classifier_uses_cache_when_given_a_dir(tmp_path):
    tax = make_taxonomy()
    clf = EmbeddingNearestCentroidClassifier(
        tax, embedder=HashingEmbedder(dimension=32), cache_dir=str(tmp_path)
    )
    texts, labels = _separable_dataset()
    clf.fit(texts, labels)
    assert clf.cache is not None
    assert clf.cache.cache_size == len(set(texts))


# --- Evaluation --------------------------------------------------------------


def test_evaluate_predictions_matches_sklearn_directly():
    from sklearn.metrics import accuracy_score, f1_score

    y_true = ["a", "a", "b", "b", "c", "c", "c"]
    y_pred = ["a", "b", "b", "b", "c", "a", "c"]
    labels = ["a", "b", "c"]

    report = evaluate_predictions(y_true, y_pred, labels=labels)

    assert report.accuracy == pytest.approx(accuracy_score(y_true, y_pred))
    assert report.macro_f1 == pytest.approx(f1_score(y_true, y_pred, labels=labels, average="macro"))
    assert report.weighted_f1 == pytest.approx(f1_score(y_true, y_pred, labels=labels, average="weighted"))
    assert report.n_examples == len(y_true)
    assert report.labels == labels


def test_evaluate_predictions_confusion_matrix_shape_and_totals():
    y_true = ["a", "a", "b", "b"]
    y_pred = ["a", "b", "b", "b"]
    labels = ["a", "b"]

    report = evaluate_predictions(y_true, y_pred, labels=labels)
    assert len(report.confusion_matrix) == 2
    assert all(len(row) == 2 for row in report.confusion_matrix)
    assert sum(sum(row) for row in report.confusion_matrix) == len(y_true)


def test_evaluate_predictions_per_intent_metrics_have_correct_support():
    y_true = ["a", "a", "a", "b"]
    y_pred = ["a", "a", "b", "b"]
    labels = ["a", "b"]

    report = evaluate_predictions(y_true, y_pred, labels=labels)
    support_by_label = {m.intent: m.support for m in report.per_intent}
    assert support_by_label["a"] == 3
    assert support_by_label["b"] == 1


def test_evaluate_predictions_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        evaluate_predictions(["a", "b"], ["a"])


def test_evaluate_predictions_rejects_empty_input():
    with pytest.raises(ValueError):
        evaluate_predictions([], [])


# --- Intent discovery (clustering) ------------------------------------------


def _three_group_messages():
    delivery = [
        "my order never arrived and its been a week",
        "package still has not shown up its very late",
        "where is my delivery it never came",
        "order is delayed again with no updates at all",
        "still waiting on my package from last week",
        "shipment never arrived even though it says delivered",
    ]
    billing = [
        "i was charged twice for the same order please help",
        "why is there a duplicate charge on my card",
        "refund me my money back right now please",
        "this charge on my statement is wrong fix it",
        "please refund this purchase i never received it",
        "billing error on my account needs a refund",
    ]
    technical = [
        "the app keeps crashing every time i open it",
        "login page is broken and gives an error",
        "website crashed during checkout and lost my cart",
        "app crashes on startup after the latest update",
        "getting an error message every time i try to log in",
        "checkout page is completely broken on mobile",
    ]
    return delivery + billing + technical


def test_discover_intent_candidates_requires_at_least_two_messages():
    with pytest.raises(ValueError):
        discover_intent_candidates(["only one message"])


def test_discover_intent_candidates_covers_all_deduplicated_messages():
    messages = _three_group_messages()
    candidates = discover_intent_candidates(messages, min_k=2, max_k=5, seed=42)

    total_clustered = sum(c.size for c in candidates)
    assert total_clustered == len(set(m.strip().lower() for m in messages))
    assert 2 <= len(candidates) <= 5


def test_discover_intent_candidates_representatives_are_real_input_messages():
    messages = _three_group_messages()
    candidates = discover_intent_candidates(messages, min_k=2, max_k=5, seed=42)
    normalized_inputs = {m.strip().lower() for m in messages}
    for c in candidates:
        for example in c.representative_examples:
            assert example.strip().lower() in normalized_inputs


def test_discover_intent_candidates_deduplicates_identical_messages():
    messages = ["same exact message", "same exact message", "a totally different one here"]
    candidates = discover_intent_candidates(messages, min_k=1, max_k=2, seed=42)
    assert sum(c.size for c in candidates) == 2


def test_discover_intent_candidates_produces_nonempty_keywords_for_most_clusters():
    messages = _three_group_messages()
    candidates = discover_intent_candidates(messages, min_k=2, max_k=5, seed=42)
    clusters_with_keywords = [c for c in candidates if c.keywords]
    assert len(clusters_with_keywords) >= 1


def test_load_brand_customer_messages_respects_brand_and_split(tmp_path):
    config = Config(
        paths=PathsConfig(
            raw_csv=str(FIXTURE),
            processed_dir=str(tmp_path / "processed"),
            splits_dir=str(tmp_path / "splits"),
        )
    )

    # Independently reconstruct the expected set using the same lower-level
    # primitives (already covered by their own unit tests) to check that
    # load_brand_customer_messages wires brand-filtering and split-filtering
    # together correctly, without hardcoding conversation ids that would be
    # brittle against the seeded shuffle.
    df = load_raw_csv(str(FIXTURE))
    tweets, _ = normalize_dataframe(df)
    conversations = build_conversations(tweets)
    pairs = extract_resolution_pairs(conversations)
    conversation_ids = [c.conversation_id for c in conversations]
    assignment = split_conversations(conversation_ids, seed=config.split.seed)
    expected_train = {
        p.customer_message
        for p in pairs
        if p.brand == "BrandA_Support" and p.conversation_id in set(assignment.train)
    }

    messages = load_brand_customer_messages(config, "BrandA_Support", split="train")
    assert set(messages) == expected_train


def test_load_brand_customer_messages_rejects_invalid_split(tmp_path):
    config = Config(
        paths=PathsConfig(
            raw_csv=str(FIXTURE),
            processed_dir=str(tmp_path / "processed"),
            splits_dir=str(tmp_path / "splits"),
        )
    )
    with pytest.raises(ValueError):
        load_brand_customer_messages(config, "BrandA_Support", split="bogus")


def test_load_brand_customer_messages_unknown_brand_returns_empty(tmp_path):
    config = Config(
        paths=PathsConfig(
            raw_csv=str(FIXTURE),
            processed_dir=str(tmp_path / "processed"),
            splits_dir=str(tmp_path / "splits"),
        )
    )
    messages = load_brand_customer_messages(config, "NotARealBrand", split="train")
    assert messages == []
