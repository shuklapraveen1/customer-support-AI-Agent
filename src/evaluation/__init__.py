"""Rigorous evaluation for the Hiver support agent.

This package never fabricates a human label or a human/judge-agreement
number. Every metric that depends on a gold label degrades to an explicit
"pending" status — reported with its numerator and denominator, never a
bare percentage — when that label hasn't actually been filled in by a
human yet. See docs/GOLDEN_SET.md and docs/EVALUATION.md for the full
methodology and its documented limitations.
"""
