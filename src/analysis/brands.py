"""Per-brand analysis: which brand in the dataset is actually suitable for
this assignment.

This module never guesses a brand — every number in its report is measured
from the loaded conversations, and `recommend_brands` derives a ranked
shortlist from thresholds in `configs/config.yaml`, not from a hardcoded
brand name. The final choice of brand is still a human judgment call, but
it should be made by reading this report, not by picking a familiar name.

Run as:

    python -m src.analysis.brands
    python -m src.analysis.brands --json-out data/processed/brand_report.json
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass

from src.config import Config, load_config
from src.data.conversations import (
    Conversation,
    ResolutionPair,
    build_conversations,
    extract_resolution_pairs,
)
from src.data.normalize import load_raw_csv, normalize_dataframe

# Deliberately tiny and generic — this is only used to collapse near-duplicate
# complaints into an "issue signature" for the repetition heuristic, not to
# understand meaning. It is not an intent taxonomy.
_STOPWORDS = {
    "the", "a", "an", "to", "is", "my", "i", "and", "for", "of", "in", "on",
    "your", "you", "it", "this", "that", "please", "can", "me", "not", "has",
    "have", "with", "was", "are", "be", "we", "im", "hi", "hey", "hello",
    "just", "so", "at", "do", "did", "no", "yes", "will", "would",
}
_WORD_RE = re.compile(r"[a-z']+")


def _signature_words(text: str, max_words: int = 6) -> tuple[str, ...]:
    """A crude normalized 'issue signature': lowercase content words only, so
    e.g. "My order #123 never arrived, please help!" and
    "My order #456 never arrived, please help!" collapse to the same
    signature and count as the same repeated issue. This is a cheap proxy
    for repetition, not a clustering model — Phase 2+ can replace it with
    embeddings if the signal here looks useful.
    """
    words = [w for w in _WORD_RE.findall(text.lower()) if w not in _STOPWORDS]
    return tuple(words[:max_words])


@dataclass
class BrandStats:
    brand: str
    tweets: int
    conversations: int
    customer_messages: int
    brand_replies: int
    resolution_pairs: int
    repeated_issue_signatures: int
    repetition_ratio: float
    top_repeated_issues: list[tuple[str, int]]

    def as_dict(self) -> dict:
        return asdict(self)


def compute_brand_stats(
    conversations: list[Conversation], pairs: list[ResolutionPair]
) -> list[BrandStats]:
    conv_by_brand: dict[str, list[Conversation]] = defaultdict(list)
    for c in conversations:
        if c.brand:
            conv_by_brand[c.brand].append(c)

    pairs_by_brand: dict[str, list[ResolutionPair]] = defaultdict(list)
    for p in pairs:
        pairs_by_brand[p.brand].append(p)

    stats: list[BrandStats] = []
    for brand, convs in conv_by_brand.items():
        tweets = sum(c.length for c in convs)
        customer_msgs = sum(1 for c in convs for t in c.turns if t.author_type == "customer")
        brand_replies = sum(1 for c in convs for t in c.turns if t.author_type == "brand")
        brand_pairs = pairs_by_brand.get(brand, [])

        sig_counter: Counter[tuple[str, ...]] = Counter()
        for p in brand_pairs:
            sig = _signature_words(p.customer_message)
            if sig:
                sig_counter[sig] += 1
        repeated = {sig: n for sig, n in sig_counter.items() if n > 1}
        repeated_msgs = sum(repeated.values())
        repetition_ratio = repeated_msgs / len(brand_pairs) if brand_pairs else 0.0
        top = sorted(repeated.items(), key=lambda kv: kv[1], reverse=True)[:5]
        top_readable = [(" ".join(sig), n) for sig, n in top]

        stats.append(
            BrandStats(
                brand=brand,
                tweets=tweets,
                conversations=len(convs),
                customer_messages=customer_msgs,
                brand_replies=brand_replies,
                resolution_pairs=len(brand_pairs),
                repeated_issue_signatures=len(repeated),
                repetition_ratio=round(repetition_ratio, 4),
                top_repeated_issues=top_readable,
            )
        )

    stats.sort(key=lambda s: s.resolution_pairs, reverse=True)
    return stats


def recommend_brands(stats: list[BrandStats], config: Config) -> list[dict]:
    """Rank brands by a composite of the volume/quality signals the
    assignment cares about, after filtering out brands too small to build a
    reliable golden eval set from.

    The score and every input to it are reported alongside the ranking, so
    the recommendation is auditable rather than a black box, and a human can
    override it having seen the same numbers.
    """
    min_conv = config.analysis.min_conversations_for_recommendation
    eligible = [
        s for s in stats if s.conversations >= min_conv and s.resolution_pairs >= min_conv
    ]
    if not eligible:
        return []

    max_pairs = max(s.resolution_pairs for s in eligible) or 1
    max_conv = max(s.conversations for s in eligible) or 1

    ranked = []
    for s in eligible:
        volume_score = s.conversations / max_conv
        pairs_score = s.resolution_pairs / max_pairs
        # repetition_ratio is already normalized to 0..1
        composite = 0.4 * pairs_score + 0.3 * volume_score + 0.3 * s.repetition_ratio
        ranked.append(
            {
                "brand": s.brand,
                "composite_score": round(composite, 4),
                "conversations": s.conversations,
                "resolution_pairs": s.resolution_pairs,
                "repetition_ratio": s.repetition_ratio,
                "top_repeated_issues": s.top_repeated_issues,
            }
        )
    ranked.sort(key=lambda r: r["composite_score"], reverse=True)
    return ranked


def run_brand_analysis(config: Config | None = None) -> dict:
    config = (config or load_config()).resolve_paths()
    df = load_raw_csv(config.paths.raw_csv)
    tweets, norm_report = normalize_dataframe(df)
    conversations = build_conversations(tweets)
    pairs = extract_resolution_pairs(conversations)
    stats = compute_brand_stats(conversations, pairs)
    recommendation = recommend_brands(stats, config)
    return {
        "normalization": norm_report.as_dict(),
        "brands": [s.as_dict() for s in stats[: config.analysis.top_brands_to_report]],
        "recommendation": recommendation,
    }


def _print_report(report: dict) -> None:
    print("=== Normalization ===")
    for k, v in report["normalization"].items():
        print(f"  {k}: {v}")

    print(f"\n=== Per-brand stats (top {len(report['brands'])} by resolution pairs) ===")
    print(f"{'brand':<25}{'tweets':>8}{'convs':>8}{'cust_msgs':>11}{'replies':>9}{'pairs':>8}{'rep_ratio':>10}")
    for b in report["brands"]:
        print(
            f"{b['brand']:<25}{b['tweets']:>8}{b['conversations']:>8}"
            f"{b['customer_messages']:>11}{b['brand_replies']:>9}"
            f"{b['resolution_pairs']:>8}{b['repetition_ratio']:>10}"
        )

    print("\n=== Recommended brands (eligible, ranked) ===")
    if not report["recommendation"]:
        print("  No brand met the minimum-conversation/pairs threshold in configs/config.yaml.")
    for r in report["recommendation"][:10]:
        print(
            f"  {r['brand']:<25} score={r['composite_score']:.4f}  "
            f"conversations={r['conversations']}  pairs={r['resolution_pairs']}  "
            f"repetition_ratio={r['repetition_ratio']}"
        )
        if r["top_repeated_issues"]:
            print(f"      repeated issues: {r['top_repeated_issues']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze brands in the raw dataset.")
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    parser.add_argument("--json-out", default=None, help="Optional path to write the full report as JSON")
    args = parser.parse_args()

    config = load_config(args.config)
    report = run_brand_analysis(config)
    _print_report(report)

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2)
        print(f"\nFull report written to {args.json_out}")


if __name__ == "__main__":
    main()
