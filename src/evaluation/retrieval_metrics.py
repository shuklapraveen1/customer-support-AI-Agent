"""Retrieval evaluation for the Phase 5 report.

A thin, deliberately non-duplicating wrapper around
`src.retrieval.search`'s proxy self-retrieval evaluation (Recall@1/3/5,
MRR). There is still no human relevance judgment linking a golden customer
message to "the one correct historical precedent" — the golden set schema
(see docs/GOLDEN_SET.md) doesn't include a gold-evidence field, and
building that judgment is exactly the kind of hand-labeling effort that's
out of scope here. So retrieval quality is reported via the same documented
proxy introduced in Phase 3, not a new, undocumented one — "MRR where
defensible" is this proxy MRR, and its defensibility limits are stated
plainly rather than glossed over.
"""

from __future__ import annotations

from typing import Optional

from src.retrieval.index import ResolutionRecord
from src.retrieval.search import (
    RetrievalEvalReport,
    build_self_retrieval_eval_queries,
    evaluate_retrieval,
)

MIN_RECORDS_FOR_MEANINGFUL_EVAL = 5


def evaluate_retrieval_for_report(
    searcher,
    records: list[ResolutionRecord],
    seed: int = 42,
    sample_size: Optional[int] = None,
) -> RetrievalEvalReport:
    """Run the proxy Recall@1/3/5 + MRR self-retrieval check against
    `searcher` (a `ResolutionIndex` or `TfidfRetrievalBaseline`).

    Raises rather than silently returning a report built from too little
    data — a handful of records isn't enough for Recall@5 to mean
    anything, and reporting a number anyway would look more authoritative
    than it is.
    """
    if len(records) < MIN_RECORDS_FOR_MEANINGFUL_EVAL:
        raise ValueError(
            f"Only {len(records)} records in the retrieval corpus — too few for a "
            f"meaningful Recall@5/MRR proxy (need at least {MIN_RECORDS_FOR_MEANINGFUL_EVAL}). "
            "This is expected against the tiny synthetic fixture; re-run against the real "
            "dataset for a trustworthy number."
        )
    eval_queries = build_self_retrieval_eval_queries(records, seed=seed, sample_size=sample_size)
    return evaluate_retrieval(searcher, eval_queries, top_k=5)
