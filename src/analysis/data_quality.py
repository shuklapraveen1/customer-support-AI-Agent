"""Whole-dataset data quality report — independent of any single brand.

Run as:

    python -m src.analysis.data_quality
    python -m src.analysis.data_quality --json-out data/processed/data_quality_report.json
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import asdict, dataclass

from src.config import Config, load_config
from src.data.conversations import build_conversations, extract_resolution_pairs
from src.data.normalize import is_url_only, load_raw_csv, normalize_dataframe


@dataclass
class ConversationLengthStats:
    min: int
    max: int
    mean: float
    median: float
    histogram: dict[str, int]


@dataclass
class DataQualityReport:
    total_rows: int
    duplicate_rows: int
    missing_text: int
    missing_ids: int
    missing_inbound: int
    malformed_timestamps: int
    orphan_replies: int
    usable_tweets: int
    inbound_count: int
    outbound_count: int
    inbound_outbound_ratio: float
    conversations: int
    conversation_length: ConversationLengthStats
    usable_resolution_pairs: int
    short_messages: int
    url_only_messages: int
    brand_distribution: list[tuple[str, int]]

    def as_dict(self) -> dict:
        return asdict(self)


def _length_stats(lengths: list[int]) -> ConversationLengthStats:
    if not lengths:
        return ConversationLengthStats(min=0, max=0, mean=0.0, median=0.0, histogram={})
    s = sorted(lengths)
    n = len(s)
    mean = sum(s) / n
    median = s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2

    hist: Counter[str] = Counter()
    for length in s:
        if length == 1:
            bucket = "1"
        elif length == 2:
            bucket = "2"
        elif length <= 4:
            bucket = "3-4"
        elif length <= 8:
            bucket = "5-8"
        else:
            bucket = "9+"
        hist[bucket] += 1

    return ConversationLengthStats(min=s[0], max=s[-1], mean=round(mean, 2), median=median, histogram=dict(hist))


def run_data_quality(config: Config | None = None) -> DataQualityReport:
    config = (config or load_config()).resolve_paths()
    df = load_raw_csv(config.paths.raw_csv)

    total_rows = len(df)
    duplicate_rows = int(df.duplicated().sum())

    tweets, norm_report = normalize_dataframe(df)
    missing_ids = norm_report.dropped_missing_tweet_id + norm_report.dropped_missing_author_id

    tweet_ids = {t.tweet_id for t in tweets}
    orphan_replies = sum(1 for t in tweets if t.in_response_to and t.in_response_to not in tweet_ids)

    inbound_count = sum(1 for t in tweets if t.inbound)
    outbound_count = sum(1 for t in tweets if not t.inbound)
    ratio = round(inbound_count / outbound_count, 3) if outbound_count else float("inf")

    conversations = build_conversations(tweets)
    pairs = extract_resolution_pairs(conversations)
    lengths = [c.length for c in conversations]

    short_max_words = config.analysis.short_message_max_words
    short_messages = sum(1 for t in tweets if len(t.text.split()) <= short_max_words)
    url_only_messages = sum(1 for t in tweets if is_url_only(t.text))

    brand_counts: Counter[str] = Counter()
    for t in tweets:
        if not t.inbound:
            brand_counts[t.author_id] += 1
    brand_distribution = brand_counts.most_common(20)

    return DataQualityReport(
        total_rows=total_rows,
        duplicate_rows=duplicate_rows,
        missing_text=norm_report.dropped_missing_text,
        missing_ids=missing_ids,
        missing_inbound=norm_report.dropped_missing_inbound,
        malformed_timestamps=norm_report.malformed_timestamps,
        orphan_replies=orphan_replies,
        usable_tweets=len(tweets),
        inbound_count=inbound_count,
        outbound_count=outbound_count,
        inbound_outbound_ratio=ratio,
        conversations=len(conversations),
        conversation_length=_length_stats(lengths),
        usable_resolution_pairs=len(pairs),
        short_messages=short_messages,
        url_only_messages=url_only_messages,
        brand_distribution=brand_distribution,
    )


def _print_report(report: DataQualityReport) -> None:
    d = report.as_dict()
    print("=== Data Quality Report ===")
    for key in (
        "total_rows", "duplicate_rows", "missing_text", "missing_ids",
        "missing_inbound", "malformed_timestamps", "orphan_replies",
        "usable_tweets", "inbound_count", "outbound_count",
        "inbound_outbound_ratio", "conversations", "usable_resolution_pairs",
        "short_messages", "url_only_messages",
    ):
        print(f"  {key}: {d[key]}")

    print("\n  conversation_length:")
    for k, v in d["conversation_length"].items():
        print(f"    {k}: {v}")

    print("\n  brand_distribution (top 20 by outbound tweet count):")
    for brand, count in d["brand_distribution"]:
        print(f"    {brand}: {count}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Whole-dataset data quality report.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    report = run_data_quality(config)
    _print_report(report)

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as fh:
            json.dump(report.as_dict(), fh, indent=2)
        print(f"\nFull report written to {args.json_out}")


if __name__ == "__main__":
    main()
