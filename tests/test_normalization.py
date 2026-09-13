from pathlib import Path

import pandas as pd
import pytest

from src.data.normalize import (
    SchemaError,
    clean_text,
    is_url_only,
    load_raw_csv,
    normalize_dataframe,
    parse_timestamp,
    resolve_columns,
)

FIXTURE = Path(__file__).parent / "fixtures" / "twcs_sample.csv"


def test_clean_text_collapses_whitespace_and_handles_missing():
    assert clean_text("  hello   \n world  ") == "hello world"
    assert clean_text(None) == ""
    assert clean_text(float("nan")) == ""
    assert clean_text("") == ""


def test_clean_text_does_not_alter_content_words():
    text = "Order #123 never arrived @BrandSupport https://example.com!!"
    assert clean_text(text) == text


def test_parse_timestamp_known_format():
    dt = parse_timestamp("Tue Oct 31 22:10:35 +0000 2017")
    assert dt is not None
    assert (dt.year, dt.month, dt.day) == (2017, 10, 31)


def test_parse_timestamp_iso_format():
    dt = parse_timestamp("2020-01-02 03:04:05")
    assert dt is not None
    assert (dt.year, dt.month, dt.day) == (2020, 1, 2)


def test_parse_timestamp_malformed_returns_none():
    assert parse_timestamp("not-a-date") is None
    assert parse_timestamp(None) is None
    assert parse_timestamp(float("nan")) is None
    assert parse_timestamp("") is None


def test_is_url_only():
    assert is_url_only("https://example.com/track/abc123") is True
    assert is_url_only("check this out https://example.com") is False
    assert is_url_only("no links here") is False
    assert is_url_only("") is False


def test_resolve_columns_handles_common_aliases():
    df = pd.DataFrame(
        {
            "id": ["1"],
            "user_id": ["u1"],
            "is_inbound": ["True"],
            "message": ["hi"],
            "timestamp": ["2020-01-01 00:00:00"],
        }
    )
    resolved = resolve_columns(df)
    assert resolved["tweet_id"] == "id"
    assert resolved["author_id"] == "user_id"
    assert resolved["inbound"] == "is_inbound"
    assert resolved["text"] == "message"
    assert resolved["created_at"] == "timestamp"


def test_resolve_columns_is_case_insensitive():
    df = pd.DataFrame({"Tweet_ID": ["1"], "Author_ID": ["a"], "Inbound": ["True"], "TEXT": ["hi"]})
    resolved = resolve_columns(df)
    assert resolved["tweet_id"] == "Tweet_ID"
    assert resolved["author_id"] == "Author_ID"
    assert resolved["text"] == "TEXT"


def test_resolve_columns_missing_required_raises():
    df = pd.DataFrame({"foo": ["bar"]})
    with pytest.raises(SchemaError):
        resolve_columns(df)


def test_load_raw_csv_keeps_ids_as_strings():
    df = load_raw_csv(str(FIXTURE))
    assert df["tweet_id"].dtype == object


def test_normalize_dataframe_against_fixture_counts():
    df = load_raw_csv(str(FIXTURE))
    tweets, report = normalize_dataframe(df)

    assert report.total_rows == 17
    assert report.kept == 12
    assert report.dropped_missing_tweet_id == 1
    assert report.dropped_missing_author_id == 1
    assert report.dropped_missing_inbound == 0
    assert report.dropped_missing_text == 2
    assert report.duplicate_tweet_ids == 1
    assert report.malformed_timestamps == 1

    total_accounted = (
        report.kept
        + report.dropped_missing_tweet_id
        + report.dropped_missing_author_id
        + report.dropped_missing_inbound
        + report.dropped_missing_text
        + report.duplicate_tweet_ids
    )
    assert total_accounted == report.total_rows


def test_duplicate_tweet_id_keeps_first_occurrence():
    df = load_raw_csv(str(FIXTURE))
    tweets, _ = normalize_dataframe(df)
    tweet_1 = next(t for t in tweets if t.tweet_id == "1")
    assert "dup" not in tweet_1.text


def test_malformed_timestamp_row_is_kept_with_none_created_at():
    df = load_raw_csv(str(FIXTURE))
    tweets, _ = normalize_dataframe(df)
    tweet_12 = next(t for t in tweets if t.tweet_id == "12")
    assert tweet_12.created_at is None
    assert tweet_12.malformed_timestamp is True
    assert tweet_12.text == "Any update on my ticket?"


def test_dropped_rows_are_absent_from_output():
    df = load_raw_csv(str(FIXTURE))
    tweets, _ = normalize_dataframe(df)
    ids = {t.tweet_id for t in tweets}
    assert "10" not in ids  # missing text
    assert "11" not in ids  # whitespace-only text
    assert "13" not in ids  # missing author_id
    assert "14" not in ids  # missing tweet_id (would show up as "" if kept)
    assert "" not in ids


def test_raw_text_preserved_even_when_cleaned_differs():
    df = pd.DataFrame(
        {
            "tweet_id": ["1"],
            "author_id": ["a"],
            "inbound": ["True"],
            "text": ["  hello   world  "],
            "created_at": ["2020-01-01 00:00:00"],
        }
    )
    tweets, _ = normalize_dataframe(df)
    assert tweets[0].text == "hello world"
    assert tweets[0].raw_text == "  hello   world  "
