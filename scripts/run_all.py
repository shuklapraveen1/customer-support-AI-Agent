#!/usr/bin/env python
"""Run the entire pipeline end-to-end, from raw data to evaluation report.

    python scripts/run_all.py --brand <brand>
    python scripts/run_all.py --brand <brand> --fast

Runs, in order:

  1. prepare data           (scripts/prepare_data.py: normalize -> conversations -> pairs -> split)
  2. analyze/select brand    (src.analysis.brands — informational; skipped if --brand is given)
  3. construct conversations (done as part of step 1; reported here for visibility)
  4. load intent taxonomy    (configs/intents.yaml)
  5. train baselines         (majority + TF-IDF, bootstrapped on the taxonomy's own examples —
                               same approach used everywhere else in this repo; a quick
                               training-set self-check is printed as a sanity check, not a
                               real accuracy claim)
  6. build retrieval index   (scripts/build_index.py --brand <brand>)
  7. run evaluation          (python -m src.evaluation.run_eval)
  8. generate report artifacts (failure_analysis.json is written by run_eval itself;
                               this step just points you at everything produced)

`--fast` subsamples the raw dataset to `--sample-size` rows (default 2000)
before step 1, uses the offline `HashingEmbedder` (skips attempting to
download a sentence-transformer model), and samples a smaller golden set
(`--fast-golden-size`, default 20) — all specifically so the *entire*
pipeline, including evaluation, reproduces headline results in well under
15 minutes on a normal laptop once the real dataset is available. Every
embedding computed anywhere in this run goes through the same
`EmbeddingCache` used elsewhere in this repo, so re-running (e.g. fast
first, then for real) never recomputes an embedding for a message it's
already seen.

This sandbox's raw dataset is a tiny synthetic fixture (the real Kaggle
dataset couldn't be downloaded here — see README), so `--fast`'s
subsampling has nothing to actually subsample in a demo run; it's still
exercised and tested (see `tests/test_run_all.py`) so it's ready for the
real dataset.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

_IMPORT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_IMPORT_ROOT))
CWD = Path.cwd()

DEFAULT_SAMPLE_SIZE = 2000
DEFAULT_GOLDEN_SIZE = 200
FAST_GOLDEN_SIZE = 20


def _run(cmd: list[str], env: dict) -> None:
    print(f"\n$ {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=str(CWD), env=env)
    if result.returncode != 0:
        print(f"Command failed with exit code {result.returncode}: {' '.join(cmd)}")
        sys.exit(result.returncode)


def maybe_subsample_raw_csv(raw_csv_path: str, sample_size: int, seed: int) -> str:
    """If `raw_csv_path` has more than `sample_size` rows, write a seeded
    random subsample next to it and return the subsample's path.
    Otherwise returns `raw_csv_path` unchanged. Subsampling rows can orphan
    some `in_response_to`/`response_tweet_id` links across the cut — the
    normalization/conversation pipeline already treats those as orphan
    replies rather than crashing (see src/data/conversations.py), so this
    is a safe, if slightly lossy, way to trade completeness for speed.
    """
    import pandas as pd

    df = pd.read_csv(raw_csv_path, dtype=str)
    if len(df) <= sample_size:
        print(f"  raw CSV has {len(df)} rows (<= sample size {sample_size}) — using it as-is, no subsampling needed.")
        return raw_csv_path

    sampled = df.sample(n=sample_size, random_state=seed)
    out_path = Path(raw_csv_path).parent / f"{Path(raw_csv_path).stem}_fast_sample.csv"
    sampled.to_csv(out_path, index=False)
    print(f"  subsampled {len(df)} rows -> {sample_size} rows -> {out_path}")
    return str(out_path)


def step_prepare_data(env: dict, config_path: str | None) -> None:
    print("\n=== Step 1-3: prepare data (normalize, build conversations, extract pairs, split) ===")
    cmd = [sys.executable, "scripts/prepare_data.py"]
    if config_path:
        cmd += ["--config", config_path]
    _run(cmd, env)


def step_select_brand(env: dict, brand: str | None, config_path: str | None) -> str:
    print("\n=== Step 2: analyze/select brand ===")
    if brand:
        print(f"  --brand={brand} given explicitly; skipping automatic selection.")
        return brand

    print("  No --brand given: running brand analysis for visibility (see src.analysis.brands).")
    cmd = [sys.executable, "-m", "src.analysis.brands"]
    if config_path:
        cmd += ["--config", config_path]
    _run(cmd, env)
    print(
        "\n  No brand was auto-selected — pick one from the report above and re-run with "
        "--brand <brand>. Brand selection is a human judgment call informed by this report, "
        "not an automatic decision (see the project's architecture notes)."
    )
    sys.exit(1)


def step_load_taxonomy(intents_config: str | None):
    print("\n=== Step 4: load intent taxonomy ===")
    from src.intent.classifier import IntentTaxonomy

    taxonomy = IntentTaxonomy.from_yaml(intents_config)
    print(f"  Loaded {len(taxonomy.labels)} intents from configs/intents.yaml: {taxonomy.labels}")
    return taxonomy


def step_train_baselines(taxonomy) -> None:
    print("\n=== Step 5: train baselines (bootstrapped on configs/intents.yaml's own examples) ===")
    from src.evaluation.metrics import intent_classification_report
    from src.intent.baseline_majority import MajorityClassifier
    from src.intent.baseline_tfidf import TfidfLogisticClassifier

    texts, labels = [], []
    for intent_id in taxonomy.labels:
        definition = taxonomy.get(intent_id)
        for example in definition.examples:
            texts.append(example)
            labels.append(intent_id)

    for name, classifier in (
        ("majority", MajorityClassifier(taxonomy)),
        ("tfidf", TfidfLogisticClassifier(taxonomy)),
    ):
        classifier.fit(texts, labels)
        predictions = [p.intent for p in classifier.predict(texts)]
        report = intent_classification_report(labels, predictions, labels=taxonomy.labels)
        print(
            f"  {name}: fit on {len(texts)} bootstrap examples; TRAINING-SET self-accuracy "
            f"{report.accuracy:.2f} (a sanity check that fitting works, NOT a generalization claim)."
        )


def step_build_index(env: dict, brand: str, embedder: str, config_path: str | None) -> None:
    print("\n=== Step 6: build retrieval index ===")
    cmd = [sys.executable, "scripts/build_index.py", "--brand", brand, "--embedder", embedder]
    if config_path:
        cmd += ["--config", config_path]
    _run(cmd, env)


def step_ensure_golden_set(env: dict, brand: str, golden_size: int, config_path: str | None) -> None:
    golden_path = CWD / "data" / "golden" / "golden_set.csv"
    print("\n=== Ensuring a golden set exists ===")
    if golden_path.exists():
        print(f"  {golden_path} already exists — leaving it as-is (never overwriting existing labels).")
        return
    print(f"  No golden set found — sampling one now (target size {golden_size}).")
    cmd = [
        sys.executable, "scripts/create_golden_template.py",
        "--brand", brand, "--target-size", str(golden_size),
    ]
    if config_path:
        cmd += ["--config", config_path]
    _run(cmd, env)


def step_run_evaluation(env: dict, config_path: str | None) -> None:
    print("\n=== Step 7-8: run evaluation + generate report artifacts ===")
    cmd = [sys.executable, "-m", "src.evaluation.run_eval"]
    if config_path:
        cmd += ["--config", config_path]
    _run(cmd, env)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the entire Hiver support-agent pipeline end-to-end.")
    parser.add_argument("--brand", default=None, help="Brand to build everything for. Omit to see the brand report first.")
    parser.add_argument("--fast", action="store_true", help="Subsample the dataset and use a smaller golden set for a quick run.")
    parser.add_argument("--sample-size", type=int, default=DEFAULT_SAMPLE_SIZE, help="Row subsample size in --fast mode.")
    parser.add_argument("--golden-size", type=int, default=None, help="Golden set target size (default: 200, or 20 in --fast mode).")
    parser.add_argument("--config", default=None)
    parser.add_argument("--intents-config", default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    start = time.time()
    env = dict(os.environ)

    raw_csv_path = env.get("RAW_CSV_PATH", str(CWD / "data" / "raw" / "twcs.csv"))
    if not Path(raw_csv_path).exists():
        print(f"No raw dataset found at {raw_csv_path}.")
        print("Download the Kaggle 'Customer Support on Twitter' dataset and place it there first — see README.")
        sys.exit(1)

    if args.fast:
        print(f"=== FAST MODE: subsampling to at most {args.sample_size} rows, using the offline embedder ===")
        sampled_path = maybe_subsample_raw_csv(raw_csv_path, args.sample_size, seed=args.seed)
        env["RAW_CSV_PATH"] = sampled_path

    golden_size = args.golden_size or (FAST_GOLDEN_SIZE if args.fast else DEFAULT_GOLDEN_SIZE)
    embedder = "hashing"  # always offline/fast in this sandbox; see README for sentence-transformer usage

    step_prepare_data(env, args.config)
    brand = step_select_brand(env, args.brand, args.config)
    taxonomy = step_load_taxonomy(args.intents_config)
    step_train_baselines(taxonomy)
    step_build_index(env, brand, embedder, args.config)
    step_ensure_golden_set(env, brand, golden_size, args.config)
    step_run_evaluation(env, args.config)

    elapsed = time.time() - start
    print(f"\n=== Done in {elapsed:.1f}s ===")
    print("Artifacts:")
    print("  artifacts/resolutions.parquet")
    print("  artifacts/index/")
    print("  artifacts/evaluation/{predictions.jsonl,results.json,results.csv,confusion_matrix.png,report.json,failure_analysis.json}")
    print("\nNext steps:")
    print("  python -m src.evaluation.inspect_cases --n 20")
    print("  python -m src.demo --message '...'")


if __name__ == "__main__":
    main()
