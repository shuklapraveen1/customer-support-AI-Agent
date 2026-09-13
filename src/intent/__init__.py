"""Intent classification: a shared taxonomy-constrained interface, two
baselines (majority-class, TF-IDF + Logistic Regression), and an embedding
nearest-centroid classifier.

No reply generation or LLM calls happen anywhere in this package — that is
a later phase. Every classifier here only ever returns labels that are
present in `configs/intents.yaml` (or the configured unknown/fallback
label); see `src.intent.classifier.IntentTaxonomy`.
"""
