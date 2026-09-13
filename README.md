# Hiver Support Agent — Customer Support on Twitter

An AI support agent built on the Kaggle **Customer Support on Twitter**
dataset (`thoughtvector/customer-support-on-twitter`). Given a brand, it
will eventually: classify incoming customer messages into a small,
data-derived set of intents, draft a reply grounded in that brand's own
historical resolutions, and decide auto-handle vs. escalate with a stated
reason — evaluated rigorously against a hand-labeled golden set, an
LLM-judge, and two baselines.

**This repository is now at Phase 5: data foundation + intent
discovery/classification + historical resolution retrieval + the AI
support agent + rigorous evaluation.** What exists today is normalization,
conversation reconstruction, conversation-level splitting, brand/quality
analysis (Phase 1); intent-cluster discovery, a configurable taxonomy,
three classifiers, and classification evaluation (Phase 2); a
TRAIN-split-only historical resolution corpus with semantic + TF-IDF
search and retrieval evaluation (Phase 3); an end-to-end agent — classify
→ retrieve → draft a grounded reply via an LLM (or a deterministic offline
mock) → decide auto-handle vs. escalate via a multi-signal policy (Phase
4); and a full evaluation harness — a stratified golden-set sampler, three
systems (majority/TF-IDF/final) evaluated head-to-head on intent,
retrieval, reply quality (via an LLM judge), and escalation metrics, a
headline "safe automation coverage" metric that always reports its
numerator and denominator, automated leakage checks, and honest
"pending"/"N/A" reporting everywhere a human label doesn't exist yet
(Phase 5).

> **No real human labels exist in this repository.** `data/golden/golden_set.csv`
> ships fully `UNLABELLED` — see "Golden evaluation set" below for exactly
> why, and `docs/GOLDEN_SET.md` for how to label it once real data is
> available.

> **The real Kaggle dataset could not be downloaded in this environment** —
> `kaggle.com` is not reachable from this sandbox (network egress is
> allow-listed and returns `host_not_allowed` for it). Everything in this
> repo is built, tested, and demonstrated against a tiny synthetic fixture
> instead (see "Getting the dataset" and "How intents are discovered"
> below for exactly what that means for `configs/intents.yaml`).

> Evaluation matters more than UI in this project. Phase 1 accordingly has
> no UI at all — everything is a script or a module you run from the
> command line.

## Repository layout

```
src/
  config.py               # typed, path-safe config loader (configs/config.yaml + env overrides)
  data/
    normalize.py           # raw CSV -> NormalizedTweet, with column aliasing and drop accounting
    conversations.py        # reply-graph reconstruction -> Conversation, ResolutionPair
    split.py                # conversation-level train/dev/test split (SEED=42)
  analysis/
    brands.py                # per-brand stats + data-driven brand recommendation
    data_quality.py          # whole-dataset data quality report
    intents.py                # intent-cluster discovery over a brand's TRAIN-split messages
  intent/
    classifier.py             # IntentTaxonomy (configs/intents.yaml) + shared fit/predict interface
    baseline_majority.py       # baseline #1: majority-class
    baseline_tfidf.py          # baseline #2: TF-IDF + Logistic Regression
    embedding.py                # embedders (sentence-transformer + offline hashing fallback), cache, nearest-centroid classifier
    evaluation.py                # accuracy / macro+weighted F1 / per-intent P-R-F1 / confusion matrix
  retrieval/
    index.py                    # TRAIN-split-only historical resolution corpus (ResolutionRecord)
    search.py                    # ResolutionIndex (FAISS/sklearn semantic search), TfidfRetrievalBaseline, retrieval eval
  agent/
    llm.py                        # LLM provider abstraction: OpenAI (production) + deterministic mock
    prompts.py                     # grounded-reply prompt construction
    reply.py                        # calls the LLM, parses/validates its structured JSON output
    escalation.py                    # multi-signal auto-handle vs. escalate policy
    __init__.py                       # run_agent(): the single end-to-end entrypoint
  evaluation/
    metrics.py                          # intent (wraps Phase 2)/escalation metrics incl. headline "safe automation coverage"
    retrieval_metrics.py                 # thin wrapper around Phase 3's proxy retrieval eval
    judge.py                              # LLM-as-judge: 8-axis 1-5 rubric scoring
    judge_agreement.py                     # human vs judge agreement (exact/MAD/Pearson/weighted kappa)
    failure_analysis.py                     # named, example-id-carrying failure categories
    run_eval.py                              # orchestrator: `python -m src.evaluation.run_eval`
scripts:
  prepare_data.py           # end-to-end: normalize -> conversations -> pairs -> split -> write outputs
  create_intent_candidates.py  # CLI: cluster a brand's train-split messages into candidate intents
  build_index.py                # CLI: build the resolution corpus + search index for a brand
  create_golden_template.py      # CLI: stratified-sample an UNLABELLED golden set template
configs/
  config.yaml               # all paths, split ratios, analysis thresholds
  intents.yaml               # the intent taxonomy (currently a placeholder — see below)
  escalation.yaml             # escalation policy thresholds (currently a placeholder — see below)
artifacts/
  resolutions.parquet       # the built historical resolution corpus (regenerable, git-ignored)
  index/                     # cached embeddings + search index metadata (regenerable, git-ignored)
  evaluation/                 # results.json/.csv, predictions.jsonl, confusion_matrix.png, report.json (regenerable, git-ignored)
data/
  golden/
    golden_set.csv              # the golden evaluation set (UNLABELLED template until hand-labeled)
    judge_validation.csv         # human/LLM-judge agreement sample (UNLABELLED template)
docs/
  GOLDEN_SET.md              # sampling procedure, label definitions, labeling/disagreement workflow
  EVALUATION.md               # full evaluation methodology, headline metric, documented limitations
tests/
  test_normalization.py
  test_conversations.py
  test_split.py
  test_intent.py
  test_retrieval.py
  test_escalation.py
  test_agent.py
  test_evaluation.py
  fixtures/
    twcs_sample.csv         # tiny synthetic fixture — tests only, never evaluation data
```

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # optional in Phase 1 — nothing here requires a key yet
```

## Getting the dataset

Download the Kaggle **Customer Support on Twitter** dataset
(`thoughtvector/customer-support-on-twitter`) and place the CSV at:

```
data/raw/twcs.csv
```

The loader does not assume an exact column layout — see "Column aliasing"
below — but it does expect a single CSV with one row per tweet.

`data/raw/`, `data/processed/`, and `data/splits/` are git-ignored (aside
from `.gitkeep`) since the raw dataset and all derived artifacts are
regenerable from it; nothing about the pipeline depends on committing them.

## Running the pipeline

```bash
# 1. Normalize, reconstruct conversations, extract resolution pairs, split, and write outputs
python scripts/prepare_data.py

# 2. Whole-dataset data quality report
python -m src.analysis.data_quality
python -m src.analysis.data_quality --json-out data/processed/data_quality_report.json

# 3. Per-brand stats and a data-driven brand shortlist
python -m src.analysis.brands
python -m src.analysis.brands --json-out data/processed/brand_report.json

# 4. Discover candidate intent clusters for a brand (train-split messages only)
python scripts/create_intent_candidates.py --brand <brand>

# 5. Build the historical resolution corpus + search index for a brand
python scripts/build_index.py --brand <brand>                    # production embedder
python scripts/build_index.py --brand <brand> --embedder hashing # explicit offline fallback
```

`scripts/prepare_data.py` writes:

- `data/processed/conversations.jsonl` — one reconstructed conversation per line
- `data/processed/resolution_pairs.jsonl` — customer→brand resolution pairs (evidence, not verified outcomes)
- `data/processed/normalization_report.json` — row-level drop accounting
- `data/splits/{train,dev,test}_conversation_ids.json` — the conversation-level split

## How intents are discovered

`python scripts/create_intent_candidates.py --brand <brand>` runs the
discovery workflow in `src/analysis/intents.py`:

1. **Sample**: pull that brand's customer messages from resolution pairs,
   restricted to conversations in the **train** split only. Discovery never
   looks at dev/test messages — the taxonomy is a modeling decision made
   before training, and defining intents by peeking at eval-set messages
   would leak eval information into the label space before evaluation even
   starts, the same way an eval-derived feature would.
2. **Represent**: embed the (deduplicated) messages — either with the
   production `SentenceTransformerEmbedder`
   (`sentence-transformers/all-MiniLM-L6-v2`) or, offline, with the
   deterministic `HashingEmbedder`.
3. **Group**: run k-means over a range of k (default 8–15, matching the
   assignment's guideline), and pick k by silhouette score — the range sets
   the assignment's guidance, but the actual number of clusters is chosen
   by what best separates *this brand's* messages, not hardcoded.
4. **Describe each cluster**: top TF-IDF keywords (fit separately from
   whatever embedding did the clustering, so keywords stay interpretable
   even when clustering used opaque embeddings) and the messages closest to
   the cluster centroid as representative examples.
5. **Name**: a cheap heuristic (top 3 keywords) by default;
   `src.analysis.intents.name_cluster_with_llm` is a deliberately
   unimplemented hook for optional LLM-assisted naming in a later phase —
   no LLM calls happen anywhere in Phase 2.

This produces **candidates** at
`data/processed/intent_candidates_<brand>.json` — it does not write
`configs/intents.yaml`. Finalizing the taxonomy is a human step: read the
candidate clusters' keywords and representative examples, then merge,
split, rename, or drop clusters into the `intents:` list in
`configs/intents.yaml`, generally landing around 8–15 intents but letting
what the data actually supports determine the exact count.

## How the taxonomy is finalized

`configs/intents.yaml` is the single source of truth for legal intent
labels. Every classifier in `src/intent/` is built against an
`IntentTaxonomy` loaded from this file, and every one of them routes both
training labels and predicted labels through
`IntentTaxonomy.validate_label()` — a label outside the file's `intents:`
list (or the configured `unknown_label`) raises `InvalidIntentLabelError`
rather than silently being accepted. That's what "the classifier must
reject arbitrary labels" means in practice: it's enforced in the shared
base class, not left to each classifier to remember.

**Current status: `configs/intents.yaml` is a placeholder.** This sandbox
has no network access to `kaggle.com`, so the real "Customer Support on
Twitter" dataset could not be downloaded here, no brand has been chosen
from real data, and therefore no real per-brand taxonomy could be derived.
The 10 intents currently in the file are a hand-written, generic set
covering common retail/consumer Twitter-support issues (delivery delays,
refunds, billing disputes, account access, app bugs, cancellations,
product defects, general complaints, compliments) — written so the
classifier code has a valid, schema-correct taxonomy to develop and test
against. **They are not measured from any brand's actual tweets and must
not be reported as if they were.** The file's own header comment repeats
this and gives the exact steps to regenerate it for real once the dataset
is available:

```bash
python -m src.analysis.brands                                   # pick a brand from real data
python scripts/create_intent_candidates.py --brand <brand>       # cluster its train-split messages
# hand-curate data/processed/intent_candidates_<brand>.json into configs/intents.yaml
```

## Why Banking77 is not used as the actual taxonomy

Banking77 is a public dataset of banking-customer-support utterances with a
77-intent taxonomy, and it's a reasonable **methodological** reference for
what a well-scoped, flat, mutually-distinguishable intent taxonomy looks
like structurally. It is never used here as the actual label set: this
project's data is general Twitter customer support for whichever brand is
selected from the Kaggle dataset — not banking — so Banking77's specific
labels (`activate_my_card`, `card_not_working`, foreign-currency-transfer
questions, etc.) simply don't describe that brand's real support issues.
Any taxonomy this project's classifiers actually use has to come out of
clustering that brand's own customer messages (the workflow above), never
off-the-shelf labels imported from an unrelated domain.

## How to train/evaluate the classifiers

There's no single "train.py" yet (that lands with the full evaluation
harness in a later phase), but every classifier shares the same interface
and can be driven directly:

```python
from src.intent.classifier import IntentTaxonomy
from src.intent.baseline_majority import MajorityClassifier
from src.intent.baseline_tfidf import TfidfLogisticClassifier
from src.intent.embedding import EmbeddingNearestCentroidClassifier, HashingEmbedder
from src.intent.evaluation import evaluate_predictions

taxonomy = IntentTaxonomy.from_yaml("configs/intents.yaml")

# train_texts / train_labels / dev_texts / dev_labels come from your
# hand-labeled golden set once it exists (a later phase) — split at the
# conversation level using src.data.split, same as everything else, so a
# classifier is never evaluated on a conversation it was trained on.

clf = TfidfLogisticClassifier(taxonomy).fit(train_texts, train_labels)
predictions = clf.predict(dev_texts)

report = evaluate_predictions(
    y_true=dev_labels,
    y_pred=[p.intent for p in predictions],
    labels=taxonomy.labels,
)
print(report.accuracy, report.macro_f1, report.weighted_f1)
for m in report.per_intent:
    print(m.intent, m.precision, m.recall, m.f1, m.support)
```

Swap `TfidfLogisticClassifier` for `MajorityClassifier` (baseline #1) or
`EmbeddingNearestCentroidClassifier` (pass a real
`SentenceTransformerEmbedder(...)` for production use, or the default
offline `HashingEmbedder()` for dev/tests) — all three return the same
`IntentPrediction(intent, confidence, reason)` shape and go through the
same `evaluate_predictions()` call. `evaluate_predictions` computes
accuracy, macro F1, weighted F1, per-intent precision/recall/F1, and a
confusion matrix directly from `sklearn.metrics` against whatever
predictions and labels you actually pass in — nothing here is estimated or
hardcoded.

`EmbeddingNearestCentroidClassifier(taxonomy, embedder=..., cache_dir=...)`
persists embeddings to disk (keyed by embedder name + text hash) so
re-running against the same messages never re-embeds them — pass a
`cache_dir` in any real run.

## How historical resolution retrieval works

`python scripts/build_index.py --brand <brand>` builds everything Phase 3
adds:

1. **Corpus (`src/retrieval/index.py`)**: `build_resolution_corpus` pulls
   customer→brand resolution pairs for the brand, restricted to
   conversations in the **TRAIN split — unconditionally**. The function
   doesn't even accept a `split` argument; there is no parameter to flip
   that would let dev/test conversations into the retrieval corpus. This is
   deliberate: retrieval evidence is what a later drafting phase is allowed
   to reference, and if it could contain dev/test resolutions, evaluating
   against those splits would leak the answer key into the evidence set.
   Each record gets a deterministic `resolution_id` (a hash of
   conversation/tweet ids) so it stays stable across rebuilds, and an
   `intent` tag predicted by a classifier bootstrapped on nothing but the
   worked examples in `configs/intents.yaml` — **a heuristic, not a gold
   label**; see the module docstring. The corpus is written to
   `artifacts/resolutions.parquet`.

2. **Embedding**: each record's `customer_message` is embedded (production:
   `SentenceTransformerEmbedder` /
   `sentence-transformers/all-MiniLM-L6-v2`; offline/dev: `HashingEmbedder`
   — same two implementations from Phase 2, reused here). Embeddings go
   through the same `EmbeddingCache` as Phase 2, keyed by embedder name +
   text hash, so re-running the build never re-embeds unchanged messages —
   cached under `artifacts/index/embedding_cache/`.

3. **Index (`src/retrieval/search.py`)**: `ResolutionIndex` wraps the
   vectors in a FAISS `IndexFlatIP` over L2-normalized vectors (exact
   cosine similarity) when FAISS is importable, and transparently falls
   back to `sklearn.neighbors.NearestNeighbors` (cosine metric) when it
   isn't — matching "FAISS is preferred if practical; if it causes
   portability problems, use sklearn." Which one actually ran is recorded
   on `index.backend_name`, not hidden. `index.save()` only ever persists
   the raw vectors (`artifacts/index/vectors.npy`) plus a small
   `meta.json` — never FAISS's own binary index format — and rebuilds
   whichever backend is available at `load()` time, which is what makes
   `artifacts/index/` portable across machines regardless of whether FAISS
   is installed on either end.

4. **Retrieve**:

   ```python
   from src.retrieval.index import load_corpus_parquet
   from src.retrieval.search import ResolutionIndex
   from src.intent.embedding import HashingEmbedder  # or SentenceTransformerEmbedder

   records = load_corpus_parquet("artifacts/resolutions.parquet")
   embedder = HashingEmbedder()
   index = ResolutionIndex.load("artifacts/index", records=records, embedder=embedder)

   results = index.retrieve("my order never arrived", intent="delivery_delay", top_k=5)
   for r in results:
       print(r.resolution_id, r.similarity_score, r.timestamp, r.brand_reply)
   ```

   Every result carries `resolution_id`, `customer_message`,
   `brand_reply` (the historical evidence itself), `intent`,
   `similarity_score`, `conversation_id`, and `timestamp` — the timestamp
   is always present in the result (even when `None`), never dropped or
   substituted. Passing `intent=` restricts candidates to that intent
   before ranking; omit it to search the whole corpus. Ranking is primarily
   by similarity score; exact-tied similarity scores are broken by
   recency (most recent first) — see `_rank_and_slice` — so "prefer
   semantically similar, same intent, reasonably recent" holds without
   ever letting recency override a genuine semantic difference or without
   ever hiding an old timestamp to make evidence look fresher than it is.

5. **TF-IDF baseline**: `TfidfRetrievalBaseline` implements the identical
   `retrieve(query, intent=None, top_k=5)` interface using
   `TfidfVectorizer` + cosine similarity instead of embeddings — no
   semantic representation at all. This is what the embedding index has to
   beat, and it's also the retrieval half of the "simple end-to-end
   baseline" a later phase will assemble.

### Retrieval evaluation — and its honest limitation

Recall@1/3/5 and MRR need a "gold" answer per query: for a given customer
message, which historical resolution was the *correct* one to retrieve?
**This dataset has no such labels**, and hand-labeling them is exactly the
kind of work that belongs in the golden-eval-set phase, not here. Building
that judgment now would either be fabricated or would smuggle a
not-yet-built deliverable in early, so `evaluate_retrieval` instead
implements an explicit, narrower **proxy**:

`build_self_retrieval_eval_queries` takes each corpus record, derives a
corrupted paraphrase-like query from it (randomly drops ~30% of its words,
seeded and deterministic), and pairs that corrupted query with the
`resolution_id` of the exact record it came from. `evaluate_retrieval` then
asks: *given this corrupted version of a real message, can the index find
its own original again?* — computing Recall@1/3/5 and MRR from real
`retrieve()` calls, never estimated.

**What this proxy does and does not tell you**: a high score means the
embedder/index combination is robust to superficial rewording of a message
it has already seen the "clean" version of. It does **not** mean the
system finds the best historical precedent for a genuinely novel complaint
the corpus has nothing like — that requires human relevance judgment, which
this proxy explicitly does not provide. `RetrievalEvalReport.method` states
this in the report itself so the numbers are never presented as if they
were real IR relevance metrics. `scripts/build_index.py` runs this proxy
for both the semantic index and the TF-IDF baseline and prints both side by
side.

## How the agent works (Phase 4)

`src.agent.run_agent(customer_message, intent_classifier, searcher, ...)` is
the single entrypoint. It runs, in order:

```
classify intent  →  retrieve historical evidence  →  draft a grounded reply  →  decide auto-handle vs. escalate
```

```python
from src.intent.classifier import IntentTaxonomy
from src.intent.baseline_tfidf import TfidfLogisticClassifier
from src.retrieval.index import load_corpus_parquet
from src.retrieval.search import ResolutionIndex
from src.intent.embedding import HashingEmbedder  # or SentenceTransformerEmbedder
from src.agent import run_agent

taxonomy = IntentTaxonomy.from_yaml("configs/intents.yaml")
classifier = TfidfLogisticClassifier(taxonomy).fit(train_texts, train_labels)

records = load_corpus_parquet("artifacts/resolutions.parquet")
index = ResolutionIndex.load("artifacts/index", records=records, embedder=HashingEmbedder())

result = run_agent("my order never arrived, please help!", classifier, index)
print(result.as_dict())
```

With no other arguments, `run_agent` calls `get_llm_provider()` and
`load_escalation_config()` internally — **this runs with zero
configuration**: no API key needed, no config files required on disk
(both have built-in defaults). That's what makes "if no API key exists,
the system MUST still run" true by construction rather than by a special
code path bolted on for tests.

### LLM provider (`src/agent/llm.py`)

Two providers, selected by the `LLM_PROVIDER` environment variable (or the
explicit `provider_name` argument to `get_llm_provider`):

- **`mock`** (`MockLLMProvider`) — deterministic, offline, no network, no
  API key. Given the same prompt it always returns the same output, built
  by simple documented rules from the structured input embedded in the
  prompt (see `_mock_generate_reply_json`), never by imitating what a real
  model would say. Every response it returns has `provider == "mock"` —
  nothing downstream can mistake it for a real completion, and it must
  never be reported as one.
- **`openai`** (`OpenAIProvider`) — wraps the OpenAI chat completions API.
  Requires `OPENAI_API_KEY` (and optionally `OPENAI_MODEL`, defaulting to
  `gpt-4o-mini`) plus the `openai` package (`pip install
  hiver-support-agent[openai]` or `pip install openai`).

Selection logic in `get_llm_provider()`: explicit argument > `LLM_PROVIDER`
env var > automatic default (`openai` if `OPENAI_API_KEY` is set, else
`mock`). An *explicit* `LLM_PROVIDER=openai` with no key still raises
`LLMConfigError` immediately rather than silently substituting mock — a
misconfiguration should be visible, not papered over; only the *unset*
default is designed to fail open onto mock.

### Reply generation (`src/agent/prompts.py`, `src/agent/reply.py`)

The prompt (`SYSTEM_PROMPT` + `build_user_prompt`) explicitly instructs the
model not to invent policies, refunds, tracking numbers, dates, or account
details; not to claim an action was performed; not to expose internal
evidence (resolution ids, similarity scores) to the customer; not to copy
a historical reply verbatim; and to respond with only the JSON object
`{"reply", "evidence_ids", "grounding_score", "confidence"}`.

`generate_reply()` parses that JSON defensively:

- Tolerates a fenced ` ```json ` block or stray prose around the object,
  but a genuinely broken/truncated response is a real parse failure, not
  something silently patched over.
- Any `evidence_ids` the model returns that weren't actually in the
  evidence it was given are dropped — the model can't get an id "for free"
  into the record just by naming it.
- `grounding_score`/`confidence` are clamped to `[0, 1]`.
- A parse failure (`ReplyResult.parse_error = True`) returns `reply=None`
  and all scores at `0.0` instead of raising — this is deliberately a
  normal, handled outcome that flows straight into the escalation policy
  (`llm_parse_error` always forces `escalate`, before any other signal is
  even checked), rather than crashing the whole pipeline on one bad
  completion.

### Escalation policy (`src/agent/escalation.py`, `configs/escalation.yaml`)

**Not a single `confidence < threshold` check.** `decide()` walks an
ordered list of named rules — first match wins, and that match's reason
becomes the stated reason:

1. **Safety** — message matches a configured safety/self-harm keyword →
   escalate, unconditionally, before anything else is even checked.
2. **LLM parse failure** — the reply drafter didn't return valid structured
   output → escalate.
3. **Out-of-distribution** — top retrieval similarity below
   `ood_similarity_threshold` → escalate, reason states plainly that
   historical evidence is insufficient.
4. **High risk** — predicted intent is in `high_risk_intents`, or the
   message matches a `high_risk_keywords` phrase (defense-in-depth against
   intent misclassification for legal/security/irreversible-account-action
   requests) → escalate.
5. **Insufficient evidence volume** — fewer than `min_evidence_count`
   historical precedents retrieved → escalate.
6. **Missing context** — message shorter than `min_message_words` (a crude
   proxy for "can't safely act on this without more detail") → escalate.
7. **Low intent confidence** — below `min_intent_confidence` → escalate.
8. **Ambiguous intent** — confidence *just barely* clears
   `min_intent_confidence` (within `ambiguous_margin` of it) → escalate.
   This is what makes the policy more than plain thresholding on the
   intent-confidence axis alone: "barely passing" and "clearly passing"
   are treated differently.
9. **Low retrieval similarity** — below `min_retrieval_similarity` (a
   softer bar than the OOD floor above) → escalate.
10. **Evidence disagreement** — the fraction of retrieved evidence sharing
    the predicted intent is below `min_evidence_agreement` (a proxy for
    "the retrieved precedents don't obviously agree with what we think
    this is about" / conflicting historical resolutions) → escalate.
11. **Low grounding** — the reply's own `grounding_score` is below
    `min_grounding_score` → escalate.
12. Otherwise → **`auto_handle`**.

The returned `signals` dict includes the four keys the assignment calls
out explicitly (`intent_confidence`, `retrieval_score`, `evidence_count`,
`risk_flag`) plus the others the policy actually used
(`ood_flag`, `grounding_score`, `reply_confidence`, `evidence_agreement`,
`ambiguous_intent`, `missing_context`, `safety_flag`, `llm_parse_error`) —
every signal that could have triggered escalation is visible in the
output, not just the one that did.

**Threshold provenance — read before trusting these numbers.**
`configs/escalation.yaml`'s thresholds are illustrative engineering
defaults, not tuned against any dev-set human judgment: this sandbox has
no real, human-labeled dev data (the real Kaggle dataset couldn't be
downloaded here — same `kaggle.com` network restriction as previous
phases). The file's own header comment repeats this and describes the
intended tuning procedure once real data exists: sweep thresholds against
**dev-split** predictions only, then freeze them and evaluate on **test**
exactly once. Choosing final thresholds by looking at test-set performance
would turn evaluation into further tuning and invalidate the test numbers
as an estimate of real-world performance — the same discipline this
project has applied to every other split-sensitive step.

### Agent output shape

```json
{
  "customer_message": "...",
  "intent": "...",
  "intent_confidence": 0.0,
  "retrieved_evidence": [
    {"resolution_id": "...", "customer_message": "...", "brand_reply": "...",
     "intent": "...", "similarity_score": 0.0, "conversation_id": "...", "timestamp": "..."}
  ],
  "reply": "...",
  "decision": "auto_handle | escalate",
  "reason": "...",
  "signals": { "intent_confidence": 0.0, "retrieval_score": 0.0, "evidence_count": 0, "risk_flag": false, "...": "..." }
}
```

`AgentResult.as_dict()` produces exactly this shape; the `AgentResult`
object itself also carries `intent_reason` and the full `RetrievalResult`
objects (with their own `.as_dict()`) for anything that wants more detail
than the spec'd output includes.

## Golden evaluation set

```bash
python scripts/create_golden_template.py --brand <brand> --target-size 200
```

Stratified-samples customer messages from **DEV+TEST split conversations
only** (never TRAIN) into `data/golden/golden_set.csv`, covering common/
rare predicted intents, ambiguous confidence, short/long/noisy messages,
missing-context cases, and messages the current system would/wouldn't
auto-handle. Every `gold_intent` / `gold_should_escalate` / `gold_reason` /
`gold_reply_quality` cell is written as the literal string `"UNLABELLED"`
— **the script never uses an LLM to guess what a human annotator would
write**, and never invents a label. Full sampling methodology, label
definitions, the labeling procedure, and how to handle ambiguity/
disagreement are in **`docs/GOLDEN_SET.md`** — read it before labeling.

**Current status**: this sandbox has no network access to `kaggle.com`
(same restriction as previous phases), so `data/golden/golden_set.csv`
ships with only the handful of examples the tiny synthetic fixture can
produce, not the real 150–250 (target 200). It is still 100% `UNLABELLED`.

## Running the evaluation

```bash
python -m src.evaluation.run_eval
```

Runs a majority-class baseline, a TF-IDF baseline, and the "final" system
(embedding classifier + semantic retrieval) through the same `run_agent`
pipeline over the golden set, scores the final system's replies with an
LLM judge, runs automated leakage checks (aborting if any hard check
fails), and writes every required artifact to `artifacts/evaluation/`:
`predictions.jsonl`, `results.json`, `results.csv`,
`confusion_matrix.png`, `report.json`. It also creates
`data/golden/judge_validation.csv` (target: 30 rows) on first run, for a
human/LLM-judge agreement check.

**The headline metric is "safe automation coverage"**: the fraction of
golden examples that were auto-handled, had the correct intent, had an
acceptable reply, and agreed with gold's escalation call — always reported
with its numerator and denominator (e.g. `"62% safe automation coverage
(124/200)"`), never a bare percentage. Full methodology — including the
per-axis LLM-judge rubric, the retrieval-proxy limitation, why "overall"
isn't computed as a plain average, human/judge agreement mechanics, and
every documented limitation of these numbers — is in
**`docs/EVALUATION.md`**; read it before trusting any number this command
prints.

**Current status**: with a fully `UNLABELLED` golden set, every
label-dependent metric reports an honest `"pending_human_labels"` /
`"N/A (0/0)"` status rather than a fabricated number — this is expected
and correct, not a bug. Once the real dataset is available and
`docs/GOLDEN_SET.md`'s labeling procedure has been followed, the same
command will report real numbers.

## Running the tests

```bash
pytest
```

Tests run **without an API key and without the real dataset**. The Phase 1
tests use the tiny synthetic fixture at `tests/fixtures/twcs_sample.csv`,
which exercises every cleaning/edge case (duplicates, missing text, missing
ids, malformed timestamps, whitespace-only text, orphan replies, short
messages, URL-only messages, multi-brand data) on 17 rows. That fixture
exists only to make the pipeline testable in isolation and must never be
used as an evaluation or golden-set source.

The Phase 2 (`test_intent.py`) and Phase 3 (`test_retrieval.py`) tests use
that same fixture for brand/split wiring checks (including an explicit
leakage test that fails if any dev/test conversation ever ends up in the
retrieval corpus), plus small hand-written in-memory text/record datasets
for classifier and retrieval behavior — and always the offline
`HashingEmbedder`, never `SentenceTransformerEmbedder`, so nothing in the
suite needs network access or downloads a model. FAISS is exercised
directly when installed; the sklearn fallback path is exercised explicitly
via `prefer_faiss=False`, so both backends are covered regardless of which
one a given machine actually has available.

The Phase 4 tests (`test_escalation.py`, `test_agent.py`) never use
`OpenAIProvider` — every test either uses `MockLLMProvider` directly or a
small hand-written stub LLM/classifier/searcher double (e.g. one that
always returns unparseable text, to test the invalid-JSON path
deterministically). `test_agent.py` covers structured-output parsing
(well-formed, invalid JSON, empty reply field, fenced code blocks,
hallucinated evidence ids, out-of-range scores), the full `run_agent`
orchestration (high-confidence auto-handle, missing evidence, low
similarity/OOD, high-risk intent, ambiguous intent, LLM parse failure
forcing escalation, and running with zero configuration at all).
`test_escalation.py` covers every individual signal and rule in `decide()`
in isolation, plus the real `configs/escalation.yaml` for schema/sanity.

The Phase 5 tests (`test_evaluation.py`) cover: label parsing (`UNLABELLED`
handling), every `RatioMetric`/escalation/safe-automation-coverage
calculation against small hand-built inputs (including the "all unlabeled
→ 0/0, not a fabricated number" case), the judge's deterministic mock
scoring (including a case where `overall` deliberately diverges from the
plain average of the other axes), human/judge agreement in both its
`"pending_human_labels"` and `"computed"` states, failure-analysis
category bucketing, and — via `run_leakage_checks` called directly — that
a known TRAIN-split conversation id is correctly rejected while a known
DEV-split id passes. `test_run_eval_end_to_end_smoke` runs the full
`run_eval.main()` pipeline against an isolated temp directory (a one-row
golden set built from the fixture, real `configs/intents.yaml` and
`configs/escalation.yaml`) and asserts every required output file is
produced and no example ever retrieves its own conversation as evidence.

## Design notes for this phase

**Column aliasing.** Different exports of the Kaggle dataset rename or
reorder columns. `src/data/normalize.py` resolves canonical field names
(`tweet_id`, `author_id`, `inbound`, `text`, `created_at`,
`in_response_to`, `response_tweet_id`) against a small alias table before
doing anything else, and raises a clear `SchemaError` naming the columns
actually present if a required field truly can't be found.

**Cleaning is conservative.** A row is only dropped when a field required
to identify or place the tweet is unusable: missing/duplicate `tweet_id`,
missing `author_id`, an unparseable `inbound` flag, or empty text after
whitespace normalization. A malformed timestamp does **not** drop the row —
`created_at` is set to `None` and the row is flagged, because conversation
order can usually still be recovered from the reply graph. Every drop
reason is counted in the `NormalizationReport`, never silently discarded.
Text content itself (URLs, mentions, punctuation, case) is never rewritten
— `raw_text` is preserved alongside the whitespace-cleaned `text`.

**Conversations are reply-graph components, not tweet-adjacency.**
`build_conversations` unions tweets via `in_response_to` /
`response_tweet_id` links (ignoring links to tweets outside the loaded set —
those are counted as orphan replies in the data quality report, not treated
as connections) and orders each resulting group chronologically, falling
back to a graph BFS when timestamps are missing so a few malformed
timestamps can't scramble an otherwise coherent thread.

**Resolution pairs are evidence, not verified outcomes.** A pair is
extracted only when a brand turn is a *direct* reply (via `in_response_to`)
to a specific customer turn — not just "next in the list," which would be
wrong for branching threads. Twitter data carries no satisfaction signal,
so a resolution pair means "this is how the brand replied to something
like this before," not "this reply solved the problem." Anything built on
top of these pairs (retrieval, drafting) must keep treating them that way.

**Splitting is conversation-level and seeded.** `split_conversations` never
splits the turns of one conversation across train/dev/test — the unit of
splitting is `conversation_id`. IDs are de-duplicated and sorted before a
seeded (`SEED=42`) shuffle, so the split is reproducible regardless of the
order conversations happen to be discovered in, and rounding remainders
always fall to `test` so no conversation is ever silently dropped.

**Brand selection is data-driven, not assumed.** `src.analysis.brands`
computes real per-brand volume/quality numbers (tweets, conversations,
customer messages, brand replies, usable resolution pairs, and a repeated-
issue-signature heuristic) and only *recommends* brands that clear
configurable minimum thresholds (`configs/config.yaml` →
`analysis.min_conversations_for_recommendation`). The composite score and
every input to it are printed, so the eventual brand choice is auditable
rather than a black box — and it's still a human decision made after
reading this report, not an automatic pick.

**No absolute paths.** Every path in `configs/config.yaml` is resolved
against the current working directory (or an explicit base) at run time via
`Config.resolve_paths()`. Overriding `RAW_CSV_PATH`, `PROCESSED_DIR`,
`SPLITS_DIR`, or `SPLIT_SEED` as environment variables (see
`.env.example`) works without editing the checked-in yaml.

## What's explicitly NOT here yet

- A brand chosen from the real dataset, and a taxonomy/retrieval
  corpus/escalation thresholds/golden set actually built, labeled, and
  tuned from it (blocked on network access to `kaggle.com`; see above)
- Real, human-filled labels anywhere: `data/golden/golden_set.csv` and
  `data/golden/judge_validation.csv` are both fully-tooled, schema-correct
  `UNLABELLED` templates — the labeling itself hasn't happened (see
  `docs/GOLDEN_SET.md`)
- Every metric that depends on those labels: real intent accuracy/F1 on
  the golden set, real escalation precision/recall/false-auto-handle-rate,
  real safe automation coverage, and real human/LLM-judge agreement all
  currently report honest "pending" statuses rather than numbers
- Real relevance-judged retrieval evaluation (only the documented proxy
  exists — see `docs/EVALUATION.md`)
- Escalation thresholds tuned against real dev-set human judgments (only
  illustrative defaults exist — see `configs/escalation.yaml`'s header and
  `docs/EVALUATION.md`'s "no data leakage" section)
- Any actual LLM API call in this sandbox — `OpenAIProvider` is
  implemented and tested for its *configuration* behavior (missing key,
  missing package) but never actually invoked, since there's no key to
  invoke it with here; every reply and every judge score in this repo's
  demo artifacts comes from `MockLLMProvider`; `name_cluster_with_llm` and
  `SentenceTransformerEmbedder`'s model download are likewise still
  explicitly unimplemented/unused

These are later phases, described in the project's architecture proposal.
