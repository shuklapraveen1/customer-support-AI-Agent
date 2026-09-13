"""Historical resolution retrieval.

Builds a searchable corpus of past customer -> brand resolution pairs from
TRAIN-split conversations only (`src.retrieval.index`), and finds similar
past resolutions for a new customer message (`src.retrieval.search`).

No reply generation happens anywhere in this package — retrieval only
surfaces historical evidence for a later drafting phase to use (and treat
as evidence, not guaranteed-correct precedent).
"""
