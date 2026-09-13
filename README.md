# Hiver Support Agent

An AI customer-support agent for the Kaggle **Customer Support on
Twitter** dataset (`thoughtvector/customer-support-on-twitter`): classify
an incoming message's intent, retrieve similar historically-resolved
cases as grounding evidence, draft a reply, and decide auto-handle vs.
escalate — evaluated by a harness built to be at least as rigorous as the
system itself.

> **Read this before anything else.** The real Kaggle dataset could not be
> downloaded in this build environment — `kaggle.com` returns
> `host_not_allowed` from this sandbox's network egress, confirmed
> directly. As a direct consequence, **no real brand, no real intent
> taxonomy, no real training data, and no real human labels exist
> anywhere in this repository.** Every component below is fully built,
> tested, and demonstrated end-to-end against a tiny synthetic fixture
> (`tests/fixtures/twcs_sample.csv`, 17 rows) instead. Every command in
> this README runs successfully today against that fixture; none of the
> numbers it prints should be read as real performance — see
> [`REPORT.md`](REPORT.md) Section 9 for exactly why, and re-run these
> same commands against the real dataset once it's available for real
> numbers.

## 1. What was built

A full pipeline, in seven phases, each with its own tests:

| Phase | What it does | Key modules |
|---|---|---|
| 1. Data foundation | Normalize raw tweets, reconstruct conversations, conversation-level train/dev/test split, brand & data-quality analysis | `src/data/`, `src/analysis/brands.py`, `src/analysis/data_quality.py` |
| 2. Intent classification | Cluster-based intent discovery, a configurable taxonomy, 3 classifiers (majority / TF-IDF / embedding), classification metrics | `src/analysis/intents.py`, `src/intent/` |
| 3. Retrieval | TRAIN-split-only historical resolution corpus, FAISS/sklearn semantic search + TF-IDF baseline, proxy retrieval evaluation | `src/retrieval/` |
| 4. Agent | LLM provider abstraction (mock + OpenAI), grounded reply drafting, a 12-rule multi-signal escalation policy | `src/agent/` |
| 5. Evaluation | Stratified golden-set sampling, LLM-as-judge, human/judge agreement, headline "safe automation coverage" metric, leakage checks | `src/evaluation/`, `scripts/create_golden_template.py` |
| 6. Reporting & demo | Automated top-5 failure-mode analysis, case inspection CLI, interactive demo, one-command fast pipeline, this documentation | `src/evaluation/failure_analysis.py`, `src/evaluation/inspect_cases.py`, `src/demo.py`, `scripts/run_all.py` |

**182 tests pass** (`pytest`), none requiring an API key or the real
dataset.

## 2. Why

Twitter customer support is high-volume, repetitive, and mostly low-risk
— a good candidate for automation — but a wrong automated reply
(fabricated refund, wrong policy, tone-deaf response to an angry
customer) is worse than no automation at all. So the design goal
throughout is **conservative, evidence-grounded automation with an
honest escalation boundary**, not maximum automation rate. See
[`REPORT.md`](REPORT.md) Section 2 for the full framing, and
[`docs/DECISIONS.md`](docs/DECISIONS.md) for the specific engineering
tradeoffs made in service of that goal.

## 3. Selected brand

**Not yet selected from real data.** `python -m src.analysis.brands`
recommends brands from measured volume/quality signals rather than
guessing (see Phase 1) — but with no real dataset available, there's
nothing real to recommend from. `configs/intents.yaml` and
`configs/escalation.yaml` both ship as clearly-labeled placeholders (see
their own header comments) built to a valid schema so the rest of the
pipeline has something to run against. Once real data is available:

```bash
python -m src.analysis.brands              # pick a brand from real volume/quality stats
python scripts/run_all.py --brand <brand>   # everything else follows from there
```

## 4. Dataset setup

Download the Kaggle **Customer Support on Twitter** dataset and place the
CSV at:

```
data/raw/twcs.csv
```

The loader tolerates common column-naming variants (see
`src/data/normalize.py`'s `COLUMN_ALIASES`) rather than assuming an exact
schema. `data/raw/`, `data/processed/`, `data/splits/`, `artifacts/`, and
`data/golden/` are all git-ignored (aside from `.gitkeep`) since
everything in them is regenerable from the raw CSV.

## 5. Installation

**macOS / Linux:**

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**Windows (PowerShell):**

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

**Windows (cmd.exe):**

```cmd
python -m venv .venv
.venv\Scripts\activate.bat
pip install -r requirements.txt
```

Optional extras (not required by the test suite, which stays fully
offline via `HashingEmbedder` / `MockLLMProvider`):

```bash
pip install sentence-transformers   # real semantic embeddings (needs model-download network access)
pip install openai                  # real LLM provider (needs OPENAI_API_KEY)
```

## 6. Configuration

Everything is YAML + environment-variable overrides, no hardcoded paths:

| File | Controls |
|---|---|
| `configs/config.yaml` | Raw/processed/splits paths, train/dev/test ratios, seed (`42`), analysis thresholds |
| `configs/intents.yaml` | The intent taxonomy (**placeholder** — see file header) |
| `configs/escalation.yaml` | Escalation policy thresholds (**placeholder, untuned** — see file header) |
| `.env` (copy from `.env.example`) | `LLM_PROVIDER`, `OPENAI_API_KEY`, `OPENAI_MODEL`, path overrides |

With nothing configured at all, the agent runs on the deterministic
`MockLLMProvider` — no API key is required for anything in this repo to
run, including the full evaluation pipeline. Config loaders fall back to
built-in defaults when a file is missing (`configs/config.yaml`,
`configs/escalation.yaml`); `configs/intents.yaml` must exist (there's no
sensible default taxonomy to fall back to).

## 7. 15-minute reproduction

```bash
python scripts/run_all.py --brand <brand> --fast
```

Runs the entire pipeline — prepare data, (informational) brand analysis,
load the taxonomy, train baselines, build the retrieval index, ensure a
golden set exists, run evaluation, write report artifacts — subsampling
the raw dataset (`--sample-size`, default 2000 rows) and using the
offline embedder throughout so it stays fast. Against the tiny fixture in
this sandbox it completes in **under 6 seconds**; the subsampling and
caching are specifically what keep it well under 15 minutes once the real
dataset is available. See `docs/DECISIONS.md` #12 and #14 for the
tradeoffs behind fast mode's design.

```bash
# Windows: identical command, just from an activated venv shell
python scripts\run_all.py --brand <brand> --fast
```

## 8. Evaluation

Step by step (what `run_all.py --fast` automates):

```bash
python scripts/prepare_data.py
python -m src.analysis.brands
python scripts/create_intent_candidates.py --brand <brand>       # optional: regenerate the taxonomy from real data
python scripts/build_index.py --brand <brand> --embedder hashing  # or --embedder sentence-transformer
python scripts/create_golden_template.py --brand <brand> --target-size 200
# --- hand-label data/golden/golden_set.csv here; see docs/GOLDEN_SET.md ---
python -m src.evaluation.run_eval
python -m src.evaluation.inspect_cases --n 20
```

`run_eval.py` runs a majority baseline, a TF-IDF baseline, and the final
system through the identical pipeline, scores the final system's replies
with an LLM judge, runs automated leakage checks (aborting if any hard
check fails), and writes `artifacts/evaluation/{predictions.jsonl,
results.json, results.csv, confusion_matrix.png, report.json,
failure_analysis.json}`. Full methodology, the headline metric's exact
definition, and every documented limitation: **[`docs/EVALUATION.md`](docs/EVALUATION.md)**.
Golden-set sampling/labeling procedure: **[`docs/GOLDEN_SET.md`](docs/GOLDEN_SET.md)**.

## 9. Demo

```bash
python -m src.demo                                            # interactive
python -m src.demo --message "My refund still hasn't arrived"  # one-shot
```

Prints `Customer / Intent / Confidence / Historical evidence / Draft
reply / Decision / Reason` for the message, using whatever retrieval
corpus exists at `artifacts/resolutions.parquet` (run
`scripts/build_index.py` first for real evidence) and the mock LLM
provider by default (`--llm-provider openai` with `OPENAI_API_KEY` set
for a real completion).

## 10. Results

**Not yet evaluated on real data** — see [`REPORT.md`](REPORT.md) for the
full report, including the actual (fixture-scale, demo-only) numbers this
pipeline currently produces, the real statistics it can honestly report
about the synthetic fixture, and a dedicated section on exactly what
would be misleading about treating any current number as a real result.
Headline metric definition and current status:

```
0% safe automation coverage (0/2)   — 2 examples total, 0 labeled; see REPORT.md §9
```

## 11. Limitations

- **No real dataset, brand, taxonomy, training data, or human labels** —
  the root cause behind everything else on this list. See the callout at
  the top of this README.
- **All three intent classifiers are bootstrapped** on `configs/intents.yaml`'s
  own 1-2 worked examples per intent, not a real labeled training set —
  expect (and don't be surprised by) low, noisy confidence.
- **Retrieval uses `HashingEmbedder`** (a lexical-overlap proxy) by
  default in this sandbox, not real semantic embeddings — `sentence-transformers/all-MiniLM-L6-v2`
  requires network access this sandbox doesn't have.
- **Retrieval evaluation is a documented proxy** (self-retrieval on a
  synthetically-perturbed query), not human relevance judgment.
- **The LLM judge and reply-drafting LLM share a provider** — self-grading
  bias is flagged, not corrected (see `docs/DECISIONS.md` #11).
- **Escalation thresholds are untuned illustrative defaults** — never
  calibrated against real dev-set human judgments (see `configs/escalation.yaml`'s
  header and `docs/EVALUATION.md`'s leakage-check section).
- **The golden set is fully unlabeled** and demo-scale (2 examples, not
  150-250) — see `docs/GOLDEN_SET.md`.

## 12. Tests

```bash
pytest
```

**182 tests, all passing, none requiring an API key or the real
dataset.** Every phase's tests use either the shared synthetic fixture
(`tests/fixtures/twcs_sample.csv`) or small hand-built in-memory data, and
always `HashingEmbedder`/`MockLLMProvider` rather than anything requiring
network access. A few slower, subprocess-based end-to-end smoke tests
(`tests/test_run_all.py`, `tests/test_demo.py`) are marked `@pytest.mark.slow`
and still run in a few seconds each against the fixture; skip them with
`pytest -m "not slow"` if desired.

```bash
pytest -q                # full suite (~10s)
pytest -m "not slow"     # skip subprocess-based end-to-end tests
pytest tests/test_agent.py tests/test_escalation.py   # a single phase
```

## Repository layout

```
src/
  config.py                     # typed, path-safe config loader
  data/                          # normalize, reconstruct conversations, conversation-level split
  analysis/                       # brands.py, data_quality.py, intents.py (cluster discovery)
  intent/                          # taxonomy, 3 classifiers, embeddings/cache, classification eval
  retrieval/                        # TRAIN-only resolution corpus, semantic/TF-IDF search, retrieval eval
  agent/                              # LLM provider, prompts, reply generation, escalation policy, run_agent()
  evaluation/                          # metrics, judge, judge agreement, failure analysis, run_eval, inspect_cases
  demo.py                                # interactive/one-shot CLI demo
scripts/
  prepare_data.py                # normalize -> conversations -> pairs -> split
  create_intent_candidates.py     # cluster a brand's train-split messages into candidate intents
  build_index.py                   # build the resolution corpus + search index
  create_golden_template.py         # stratified-sample an UNLABELLED golden set
  run_all.py                          # orchestrates everything, with --fast subsampling
configs/
  config.yaml / intents.yaml / escalation.yaml
artifacts/
  resolutions.parquet, index/, evaluation/   # all regenerable, git-ignored
data/
  raw/, processed/, splits/, golden/          # all regenerable except hand-labeled golden CSVs, git-ignored
docs/
  GOLDEN_SET.md, EVALUATION.md, DECISIONS.md
tests/
  test_*.py (182 tests), fixtures/twcs_sample.csv
REPORT.md
```

## Design notes worth knowing before reading the code

- **Conversation-level splitting everywhere.** No function in this repo
  can split individual tweets across train/dev/test — splitting operates
  on `conversation_id` from `src/data/split.py` onward, and the retrieval
  corpus builder (`build_resolution_corpus`) doesn't even accept a split
  argument, guaranteeing it can only ever read TRAIN.
  `run_eval.py` re-verifies this with automated leakage checks before
  evaluating anything.
- **Cleaning is conservative.** A row is only dropped when a required
  field (id, author, inbound flag, non-empty text) is truly unusable. A
  malformed timestamp is kept (flagged, `created_at=None`), never dropped
  or guessed at — timestamps are reported as-is everywhere downstream,
  including when missing (see `docs/DECISIONS.md` #13).
- **Nothing pretends to be something it isn't.** `MockLLMProvider` always
  reports `provider="mock"`; the golden-set template always says
  `"UNLABELLED"` rather than a guessed label; placeholder configs say so
  in their own headers; every ratio metric reports its numerator and
  denominator, never a bare percentage; every metric that needs a gold
  label degrades to an honest "pending" status rather than a fabricated
  number.
- **Historical replies are evidence, not truth.** The reply-drafting
  prompt explicitly forbids copying a historical reply verbatim and
  forbids inventing policies/refunds/tracking numbers/dates/actions —
  retrieval surfaces a *pattern*, drafting has to synthesize from it.
- **Escalation is a named rule ladder, not a single threshold** — see
  `docs/DECISIONS.md` #7 for the full ordered list and why.

For anything not covered here, `docs/EVALUATION.md` and
`docs/DECISIONS.md` go into far more depth than a README should.
