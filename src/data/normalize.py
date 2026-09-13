"""Normalize raw Twitter Customer Support rows into a single internal schema.

The Kaggle "Customer Support on Twitter" dataset is usually shipped as
`twcs.csv` with columns:

    tweet_id, author_id, inbound, created_at, text,
    response_tweet_id, in_response_to_tweet_id

but different copies/exports rename, reorder, or drop optional columns, so
this module resolves a handful of common aliases before doing anything else.

Cleaning here is deliberately conservative: a row is only dropped when a
field required to identify or place the tweet is unusable (no id, no text,
no parseable inbound flag). Text itself is whitespace-normalized but never
rewritten, truncated, or filtered by content — `raw_text` keeps the original
value so nothing is destroyed that a later phase might need.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import pandas as pd

# --- column aliasing --------------------------------------------------------

COLUMN_ALIASES: dict[str, list[str]] = {
    "tweet_id": ["tweet_id", "id", "tweetid"],
    "author_id": ["author_id", "user_id", "authorid"],
    "inbound": ["inbound", "is_inbound"],
    "text": ["text", "tweet_text", "message", "body"],
    "created_at": ["created_at", "timestamp", "tweet_created_at", "date"],
    "in_response_to": [
        "in_response_to_tweet_id",
        "in_response_to",
        "reply_to",
        "in_reply_to_tweet_id",
    ],
    "response_tweet_id": [
        "response_tweet_id",
        "response_tweet_ids",
        "responses",
    ],
}

# Columns without which a row simply cannot be normalized at all.
REQUIRED_CANONICAL = {"tweet_id", "author_id", "inbound", "text"}

# A handful of known timestamp formats, tried in order before falling back to
# pandas' more permissive (but slower) parser. Anything that fails all of
# them is treated as malformed rather than raising.
_TIMESTAMP_FORMATS = (
    "%a %b %d %H:%M:%S %z %Y",  # canonical twcs.csv format
    "%Y-%m-%d %H:%M:%S%z",
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%dT%H:%M:%S%z",
    "%Y-%m-%dT%H:%M:%SZ",
)

_URL_RE = re.compile(r"https?://\S+")
_MENTION_RE = re.compile(r"@\w+")
_HASHTAG_RE = re.compile(r"#\w+")
_NON_WORD_RE = re.compile(r"[^\w\s]")
_WHITESPACE_RE = re.compile(r"\s+")


class SchemaError(ValueError):
    """Raised when a raw dataframe is missing a required field entirely,
    even after alias resolution."""


def resolve_columns(df: pd.DataFrame) -> dict[str, Optional[str]]:
    """Map canonical field name -> actual column name present in `df` (or
    None if absent). Raises SchemaError if a *required* field has no match.
    """
    lower_cols = {c.lower(): c for c in df.columns}
    resolved: dict[str, Optional[str]] = {}
    for canonical, aliases in COLUMN_ALIASES.items():
        found = None
        for alias in aliases:
            if alias in lower_cols:
                found = lower_cols[alias]
                break
        resolved[canonical] = found

    missing_required = sorted(REQUIRED_CANONICAL - {k for k, v in resolved.items() if v})
    if missing_required:
        raise SchemaError(
            "Raw CSV is missing required column(s) (after alias resolution): "
            f"{missing_required}. Columns present: {list(df.columns)}"
        )
    return resolved


def _is_missing(raw: object) -> bool:
    if raw is None:
        return True
    if isinstance(raw, float) and math.isnan(raw):
        return True
    return False


def clean_text(raw: object) -> str:
    """Whitespace-normalize text without altering its content otherwise.

    Missing values become "". Internal whitespace (including newlines and
    tabs) is collapsed to single spaces and the result is stripped. URLs,
    mentions, hashtags, punctuation, and case are left untouched — cleaning
    here is about malformed structure, not content policy.
    """
    if _is_missing(raw):
        return ""
    text = str(raw)
    return _WHITESPACE_RE.sub(" ", text).strip()


def parse_timestamp(raw: object) -> Optional[datetime]:
    """Best-effort timestamp parse across known formats. None if unparseable."""
    if _is_missing(raw):
        return None
    text = str(raw).strip()
    if not text:
        return None

    for fmt in _TIMESTAMP_FORMATS:
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue

    try:
        parsed = pd.to_datetime(text, errors="raise", utc=False)
        if pd.isna(parsed):
            return None
        return parsed.to_pydatetime()
    except Exception:
        return None


def is_url_only(text: str) -> bool:
    """True if the message contains at least one URL and nothing else once
    URLs, @mentions, #hashtags, and punctuation are stripped out."""
    if not text or not _URL_RE.search(text):
        return False
    stripped = _URL_RE.sub("", text)
    stripped = _MENTION_RE.sub("", stripped)
    stripped = _HASHTAG_RE.sub("", stripped)
    stripped = _NON_WORD_RE.sub("", stripped)
    return not stripped.strip()


def _clean_id(raw: object) -> Optional[str]:
    if _is_missing(raw):
        return None
    text = str(raw).strip()
    if not text or text.lower() == "nan":
        return None
    # Ids sometimes arrive as floats (e.g. "123.0") when a dataframe column
    # was inferred as numeric before reaching us.
    if text.endswith(".0") and text[:-2].replace("-", "").isdigit():
        text = text[:-2]
    return text


def _split_ids(raw: object) -> list[str]:
    """`response_tweet_id` can carry multiple comma/pipe-separated ids."""
    if _is_missing(raw):
        return []
    parts = re.split(r"[,\|]", str(raw))
    out = []
    for part in parts:
        cid = _clean_id(part)
        if cid:
            out.append(cid)
    return out


def _parse_inbound(raw: object) -> Optional[bool]:
    if isinstance(raw, bool):
        return raw
    if _is_missing(raw):
        return None
    text = str(raw).strip().lower()
    if text in ("true", "t", "1", "yes"):
        return True
    if text in ("false", "f", "0", "no"):
        return False
    return None


@dataclass
class NormalizedTweet:
    tweet_id: str
    author_id: str
    inbound: bool
    text: str
    raw_text: str
    created_at: Optional[datetime]
    created_at_raw: Optional[str]
    in_response_to: Optional[str]
    response_tweet_ids: list[str] = field(default_factory=list)
    malformed_timestamp: bool = False


@dataclass
class NormalizationReport:
    total_rows: int = 0
    kept: int = 0
    dropped_missing_tweet_id: int = 0
    dropped_missing_author_id: int = 0
    dropped_missing_inbound: int = 0
    dropped_missing_text: int = 0
    duplicate_tweet_ids: int = 0
    malformed_timestamps: int = 0

    def as_dict(self) -> dict:
        return dict(self.__dict__)


def normalize_dataframe(
    df: pd.DataFrame,
) -> tuple[list[NormalizedTweet], NormalizationReport]:
    """Turn a raw dataframe into a deduplicated list of `NormalizedTweet`.

    Drop rules (each counted, never silent):
      - no usable tweet_id      -> dropped (nothing can reference this row)
      - duplicate tweet_id      -> first occurrence kept, later ones dropped
      - no usable author_id     -> dropped (can't tell customer from brand)
      - inbound not parseable   -> dropped
      - text empty after clean  -> dropped

    A malformed timestamp does NOT drop the row; `created_at` is set to None
    and `malformed_timestamp=True`, because conversation order can often
    still be recovered from the reply graph (see `src.data.conversations`).
    """
    cols = resolve_columns(df)
    report = NormalizationReport(total_rows=len(df))
    seen_ids: set[str] = set()
    tweets: list[NormalizedTweet] = []

    for _, row in df.iterrows():
        tweet_id = _clean_id(row.get(cols["tweet_id"])) if cols["tweet_id"] else None
        if not tweet_id:
            report.dropped_missing_tweet_id += 1
            continue
        if tweet_id in seen_ids:
            report.duplicate_tweet_ids += 1
            continue

        author_id = _clean_id(row.get(cols["author_id"])) if cols["author_id"] else None
        if not author_id:
            report.dropped_missing_author_id += 1
            continue

        inbound = _parse_inbound(row.get(cols["inbound"])) if cols["inbound"] else None
        if inbound is None:
            report.dropped_missing_inbound += 1
            continue

        raw_text_val = row.get(cols["text"]) if cols["text"] else None
        raw_text = "" if _is_missing(raw_text_val) else str(raw_text_val)
        text = clean_text(raw_text_val)
        if not text:
            report.dropped_missing_text += 1
            continue

        created_raw = row.get(cols["created_at"]) if cols["created_at"] else None
        created_at = parse_timestamp(created_raw)
        malformed_ts = (not _is_missing(created_raw)) and str(created_raw).strip() != "" and created_at is None
        if malformed_ts:
            report.malformed_timestamps += 1

        in_response_to = (
            _clean_id(row.get(cols["in_response_to"])) if cols["in_response_to"] else None
        )
        response_ids = (
            _split_ids(row.get(cols["response_tweet_id"])) if cols["response_tweet_id"] else []
        )

        seen_ids.add(tweet_id)
        tweets.append(
            NormalizedTweet(
                tweet_id=tweet_id,
                author_id=author_id,
                inbound=inbound,
                text=text,
                raw_text=raw_text,
                created_at=created_at,
                created_at_raw=(str(created_raw) if not _is_missing(created_raw) else None),
                in_response_to=in_response_to,
                response_tweet_ids=response_ids,
                malformed_timestamp=malformed_ts,
            )
        )

    report.kept = len(tweets)
    return tweets, report


def load_raw_csv(path: str) -> pd.DataFrame:
    """Load the raw CSV with every column kept as string/object.

    Reading everything as `str` up front (rather than letting pandas infer
    dtypes) avoids pandas silently turning an id like `"123"` into the float
    `123.0` before our own cleaning gets a chance to run.
    """
    return pd.read_csv(path, dtype=str, keep_default_na=True, na_values=["", "NaN", "nan"])
