"""Reconstruct conversation threads and customer <-> brand resolution pairs
from normalized tweets.

A conversation is a weakly-connected component of the reply graph formed by
`in_response_to` / `response_tweet_ids` links. Turns are ordered by
timestamp when every tweet in the group has one; otherwise ordering falls
back to a BFS walk of the reply graph, so a handful of malformed timestamps
in an otherwise well-formed thread doesn't scramble the whole conversation.

A resolution pair here is *evidence* of how a brand has replied to a similar
message before — it is not a guarantee that the reply actually resolved the
customer's issue (this dataset has no satisfaction signal), and downstream
retrieval/drafting code must treat it that way.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from src.data.normalize import NormalizedTweet

AuthorType = str  # "customer" | "brand"


@dataclass
class Turn:
    tweet_id: str
    author_type: AuthorType
    author_id: str
    timestamp: Optional[datetime]
    text: str
    in_response_to: Optional[str]


@dataclass
class Conversation:
    conversation_id: str
    brand: Optional[str]
    turns: list[Turn] = field(default_factory=list)

    @property
    def length(self) -> int:
        return len(self.turns)


@dataclass
class ResolutionPair:
    conversation_id: str
    brand: str
    customer_tweet_id: str
    brand_tweet_id: str
    customer_message: str
    brand_reply: str
    customer_timestamp: Optional[datetime]
    brand_timestamp: Optional[datetime]


def _union_find_groups(tweets: list[NormalizedTweet]) -> list[list[NormalizedTweet]]:
    """Weakly-connected components of the reply graph, using only links that
    point at another tweet actually present in `tweets` (a link to a tweet
    outside the loaded set is an orphan reply, not a connection)."""
    by_id = {t.tweet_id: t for t in tweets}
    parent: dict[str, str] = {t.tweet_id: t.tweet_id for t in tweets}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for t in tweets:
        if t.in_response_to and t.in_response_to in by_id:
            union(t.tweet_id, t.in_response_to)
        for rid in t.response_tweet_ids:
            if rid in by_id:
                union(t.tweet_id, rid)

    groups: dict[str, list[NormalizedTweet]] = defaultdict(list)
    for t in tweets:
        groups[find(t.tweet_id)].append(t)
    return list(groups.values())


def _order_group(group: list[NormalizedTweet]) -> list[NormalizedTweet]:
    by_id = {t.tweet_id: t for t in group}
    ids_in_group = set(by_id)

    have_ts = [t for t in group if t.created_at is not None]
    if len(have_ts) == len(group):
        return sorted(group, key=lambda t: (t.created_at, t.tweet_id))

    # Mixed/degraded case: walk the reply graph breadth-first from its
    # root(s), using timestamps only to break ties within the same graph
    # level. This keeps a thread coherent even when some hops have no usable
    # timestamp at all.
    parents = {
        t.tweet_id: t.in_response_to if t.in_response_to in ids_in_group else None
        for t in group
    }
    children: dict[str, list[str]] = defaultdict(list)
    for tid, parent_id in parents.items():
        if parent_id:
            children[parent_id].append(tid)

    roots = [tid for tid, parent_id in parents.items() if parent_id is None]
    roots.sort(key=lambda tid: (by_id[tid].created_at or datetime.max, tid))

    ordered: list[NormalizedTweet] = []
    seen: set[str] = set()
    queue: deque[str] = deque(roots)
    while queue:
        tid = queue.popleft()
        if tid in seen:
            continue
        seen.add(tid)
        ordered.append(by_id[tid])
        kids = sorted(
            children.get(tid, []), key=lambda c: (by_id[c].created_at or datetime.max, c)
        )
        for kid in reversed(kids):
            queue.appendleft(kid)

    # Anything unreached (e.g. a reply cycle, which shouldn't occur in this
    # dataset but must not crash the pipeline) is appended, sorted.
    missing = [t for t in group if t.tweet_id not in seen]
    missing.sort(key=lambda t: (t.created_at or datetime.max, t.tweet_id))
    return ordered + missing


def build_conversations(tweets: list[NormalizedTweet]) -> list[Conversation]:
    """Group tweets into conversations and order each conversation's turns.

    `brand` is the author_id of the first outbound (non-inbound) tweet found
    in the ordered conversation, or None if the conversation contains no
    brand turn at all (an unresolved / unanswered customer message).
    """
    groups = _union_find_groups(tweets)
    conversations: list[Conversation] = []

    for group in groups:
        ordered = _order_group(group)
        conversation_id = ordered[0].tweet_id

        brand = None
        for t in ordered:
            if not t.inbound:
                brand = t.author_id
                break

        turns = [
            Turn(
                tweet_id=t.tweet_id,
                author_type="customer" if t.inbound else "brand",
                author_id=t.author_id,
                timestamp=t.created_at,
                text=t.text,
                in_response_to=t.in_response_to,
            )
            for t in ordered
        ]
        conversations.append(Conversation(conversation_id=conversation_id, brand=brand, turns=turns))

    conversations.sort(key=lambda c: c.conversation_id)
    return conversations


def extract_resolution_pairs(conversations: list[Conversation]) -> list[ResolutionPair]:
    """Pairs of (customer turn, brand turn) where the brand turn *directly*
    replies to that specific customer turn — not merely the next turn in
    list order, which would be wrong for branching threads.

    Only conversations with an identified brand are considered, since a pair
    without a known brand can't be attributed for retrieval or analysis.
    """
    pairs: list[ResolutionPair] = []
    for conv in conversations:
        if not conv.brand:
            continue
        by_id = {t.tweet_id: t for t in conv.turns}
        for turn in conv.turns:
            if turn.author_type != "brand" or not turn.in_response_to:
                continue
            customer_turn = by_id.get(turn.in_response_to)
            if customer_turn is None or customer_turn.author_type != "customer":
                continue
            pairs.append(
                ResolutionPair(
                    conversation_id=conv.conversation_id,
                    brand=conv.brand,
                    customer_tweet_id=customer_turn.tweet_id,
                    brand_tweet_id=turn.tweet_id,
                    customer_message=customer_turn.text,
                    brand_reply=turn.text,
                    customer_timestamp=customer_turn.timestamp,
                    brand_timestamp=turn.timestamp,
                )
            )
    return pairs
