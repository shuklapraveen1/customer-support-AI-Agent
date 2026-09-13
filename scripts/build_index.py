#!/usr/bin/env python
"""Build the historical resolution corpus and search index for a brand.

    python scripts/build_index.py --brand AmazonHelp
    python scripts/build_index.py --brand AmazonHelp --embedder hashing   # explicit offline fallback

Writes:
    artifacts/resolutions.parquet   — the TRAIN-split-only resolution corpus
    artifacts/index/vectors.npy     — cached embeddings for that corpus
    artifacts/index/meta.json       — embedder/backend metadata

Then runs the retrieval self-check (a proxy Recall@k/MRR — see
`src.retrieval.search.build_self_retrieval_eval_queries`) for both the
semantic index and the TF-IDF baseline, and prints both.

Does not draft any replies or call an LLM.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config
from src.intent.classifier import IntentTaxonomy
from src.intent.embedding import EmbeddingUnavailableError, HashingEmbedder, SentenceTransformerEmbedder
from src.retrieval.index import build_resolution_corpus, save_corpus_parquet
from src.retrieval.search import (
    ResolutionIndex,
    TfidfRetrievalBaseline,
    build_self_retrieval_eval_queries,
    evaluate_retrieval,
)


def _build_embedder(name: str):
    if name == "hashing":
        print("Using HashingEmbedder — an explicit offline/dev fallback, NOT the production "
              "semantic embedder. Results built with it should not be reported as if they came "
              "from sentence-transformers/all-MiniLM-L6-v2.")
        return HashingEmbedder()

    try:
        return SentenceTransformerEmbedder()
    except EmbeddingUnavailableError as exc:
        print(f"Could not load the production embedder: {exc}")
        print("Re-run with --embedder hashing to explicitly opt into the offline fallback instead.")
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the resolution corpus and search index for a brand.")
    parser.add_argument("--brand", required=True)
    parser.add_argument("--config", default=None)
    parser.add_argument("--intents-config", default=None, help="Path to configs/intents.yaml")
    parser.add_argument("--embedder", choices=["sentence-transformer", "hashing"], default="sentence-transformer")
    parser.add_argument("--corpus-out", default=None, help="Defaults to artifacts/resolutions.parquet")
    parser.add_argument("--index-out", default=None, help="Defaults to artifacts/index")
    parser.add_argument("--no-faiss", action="store_true", help="Force the sklearn backend even if FAISS is available")
    parser.add_argument("--eval-sample-size", type=int, default=None)
    args = parser.parse_args()

    config = load_config(args.config).resolve_paths()
    corpus_path = args.corpus_out or str(Path("artifacts") / "resolutions.parquet")
    index_dir = args.index_out or str(Path("artifacts") / "index")

    try:
        taxonomy = IntentTaxonomy.from_yaml(args.intents_config)
    except Exception as exc:
        print(f"Could not load intent taxonomy ({exc}); records will be tagged intent='unknown'.")
        taxonomy = None

    print(f"Building resolution corpus for brand='{args.brand}' from TRAIN-split conversations only ...")
    records = build_resolution_corpus(config, args.brand, taxonomy=taxonomy)
    print(f"  {len(records)} historical resolution pairs")
    if not records:
        print("No resolution pairs found for this brand in the train split — nothing to index.")
        sys.exit(1)

    save_corpus_parquet(records, corpus_path)
    print(f"  wrote {corpus_path}")

    embedder = _build_embedder(args.embedder)
    print(f"Embedding corpus with '{embedder.name}' ...")
    index = ResolutionIndex.build(
        records, embedder, cache_dir=str(Path(index_dir) / "embedding_cache"), prefer_faiss=not args.no_faiss
    )
    print(f"  backend: {index.backend_name}")
    index.save(index_dir)
    print(f"  wrote {index_dir}/vectors.npy and meta.json")

    print("\nRunning proxy retrieval self-check (see docstring for what this does and doesn't measure) ...")
    eval_queries = build_self_retrieval_eval_queries(records, seed=config.split.seed, sample_size=args.eval_sample_size)
    if len(eval_queries) < 5:
        print(f"  only {len(eval_queries)} eval queries available — too few for a meaningful proxy score, skipping.")
    else:
        semantic_report = evaluate_retrieval(index, eval_queries, top_k=5)
        print(f"  [semantic index]  {semantic_report.as_dict()}")

        tfidf_baseline = TfidfRetrievalBaseline(records)
        tfidf_report = evaluate_retrieval(tfidf_baseline, eval_queries, top_k=5)
        print(f"  [tfidf baseline]  {tfidf_report.as_dict()}")


if __name__ == "__main__":
    main()
