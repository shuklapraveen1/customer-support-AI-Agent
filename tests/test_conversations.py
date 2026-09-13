from pathlib import Path

from src.data.conversations import build_conversations, extract_resolution_pairs
from src.data.normalize import load_raw_csv, normalize_dataframe

FIXTURE = Path(__file__).parent / "fixtures" / "twcs_sample.csv"


def _tweets():
    df = load_raw_csv(str(FIXTURE))
    tweets, _ = normalize_dataframe(df)
    return tweets


def test_build_conversations_groups_reply_chains_correctly():
    conversations = build_conversations(_tweets())

    # 8 usable groups: {1,2,3} {4,5} {6,7} {8} {12} {15} {16} {17}
    assert len(conversations) == 8

    by_id = {c.conversation_id: c for c in conversations}
    chain = by_id["1"]
    assert [t.tweet_id for t in chain.turns] == ["1", "2", "3"]
    assert [t.author_type for t in chain.turns] == ["customer", "brand", "customer"]
    assert chain.brand == "BrandA_Support"


def test_conversation_turns_are_chronologically_ordered():
    conversations = build_conversations(_tweets())
    chain = next(c for c in conversations if c.conversation_id == "1")
    timestamps = [t.timestamp for t in chain.turns]
    assert timestamps == sorted(timestamps)


def test_second_brandA_conversation_grouped_separately():
    conversations = build_conversations(_tweets())
    by_id = {c.conversation_id: c for c in conversations}
    chain = by_id["4"]
    assert [t.tweet_id for t in chain.turns] == ["4", "5"]
    assert chain.brand == "BrandA_Support"


def test_brandB_conversation():
    conversations = build_conversations(_tweets())
    by_id = {c.conversation_id: c for c in conversations}
    chain = by_id["6"]
    assert [t.tweet_id for t in chain.turns] == ["6", "7"]
    assert chain.brand == "BrandB_Care"


def test_orphan_reply_becomes_its_own_single_turn_conversation():
    conversations = build_conversations(_tweets())
    orphan = next(c for c in conversations if c.conversation_id == "15")
    assert len(orphan.turns) == 1
    assert orphan.turns[0].in_response_to == "9999"
    # the brand still authored it, so brand is knowable even though the
    # reply-to target doesn't exist in this dataset slice
    assert orphan.brand == "BrandA_Support"


def test_unresolved_customer_message_has_no_brand():
    conversations = build_conversations(_tweets())
    unresolved = next(c for c in conversations if c.conversation_id == "8")
    assert len(unresolved.turns) == 1
    assert unresolved.brand is None


def test_malformed_timestamp_tweet_still_forms_its_own_conversation():
    conversations = build_conversations(_tweets())
    conv = next(c for c in conversations if c.conversation_id == "12")
    assert len(conv.turns) == 1
    assert conv.turns[0].timestamp is None


def test_extract_resolution_pairs_finds_direct_replies_only():
    conversations = build_conversations(_tweets())
    pairs = extract_resolution_pairs(conversations)

    assert len(pairs) == 3
    by_customer_id = {p.customer_tweet_id: p for p in pairs}

    assert by_customer_id["1"].brand_tweet_id == "2"
    assert by_customer_id["1"].brand == "BrandA_Support"
    assert "order #123" in by_customer_id["1"].customer_message

    assert by_customer_id["4"].brand_tweet_id == "5"
    assert by_customer_id["6"].brand_tweet_id == "7"
    assert by_customer_id["6"].brand == "BrandB_Care"


def test_no_pair_from_orphan_reply_or_unresolved_or_short_conversations():
    conversations = build_conversations(_tweets())
    pairs = extract_resolution_pairs(conversations)
    customer_ids_with_pairs = {p.customer_tweet_id for p in pairs}

    assert "8" not in customer_ids_with_pairs  # unresolved
    assert "16" not in customer_ids_with_pairs  # short, no brand
    assert "17" not in customer_ids_with_pairs  # URL-only, no brand


def test_resolution_pairs_carry_both_timestamps_when_available():
    conversations = build_conversations(_tweets())
    pairs = extract_resolution_pairs(conversations)
    pair = next(p for p in pairs if p.customer_tweet_id == "1")
    assert pair.customer_timestamp is not None
    assert pair.brand_timestamp is not None
    assert pair.brand_timestamp >= pair.customer_timestamp
