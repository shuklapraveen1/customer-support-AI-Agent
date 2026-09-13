# Evaluation methodology

## Status right now — read this first

Every number this evaluation pipeline can currently produce is either (a)
a mechanical/self-consistency check that needs no gold label, or (b) an
honest **"pending"** placeholder, because `data/golden/golden_set.csv`
ships fully unlabeled in this sandbox (see `docs/GOLDEN_SET.md` for why —
no network access to the real dataset, so no real hand-labeling was
possible). Nothing in `artifacts/evaluation/` should be read as a real
performance measurement yet. This document describes what the pipeline
computes, how, and — just as importantly — what it deliberately does NOT
claim to measure.

## Running it

```bash
python -m src.evaluation.run_eval
```

Runs three systems through the identical `run_agent(...)` pipeline over
every row in `data/golden/golden_set.csv`:

| System | Intent classifier | Retrieval |
|---|---|---|
| `majority_baseline` | `MajorityClassifier` | `TfidfRetrievalBaseline` |
| `tfidf_baseline` | `TfidfLogisticClassifier` | `TfidfRetrievalBaseline` |
| `final_system` | `EmbeddingNearestCentroidClassifier` | `ResolutionIndex` (semantic, FAISS/sklearn) |

All three share the same LLM provider, escalation config, and golden set,
so the *only* things that differ between them are classification and
retrieval sophistication — isolating exactly what those two components
contribute to the full pipeline's behavior. All three classifiers are fit
**only** on `configs/intents.yaml`'s own worked examples (the same
bootstrap approach used in Phases 2–3) — never on anything from
`golden_set.csv` — which is what makes "evaluation labels are not used
during model fitting" true by construction (see "No data leakage" below).

The LLM judge (`src.evaluation.judge`) only scores `final_system`'s
replies — running it on all three would triple judging cost for baselines
whose reply quality isn't really the point of comparison.

## Metrics

### Intent (reuses `src.intent.evaluation`, built in Phase 2)

Accuracy, macro F1, weighted F1, per-class precision/recall/F1, and a
confusion matrix (`artifacts/evaluation/confusion_matrix.png`, for
`final_system` only) — computed only over golden rows with a non-empty
`gold_intent`. If zero rows are labeled, this whole section reports
`{"status": "pending_human_labels"}` instead of a fabricated 0% or 100%.

### Retrieval (reuses `src.retrieval.search`, built in Phase 3)

Recall@1/3/5 and MRR via the **same documented proxy** introduced in Phase
3: each corpus record is turned into a synthetically word-dropped query,
and we check whether retrieval can still find that record's own original
entry. **This is not human relevance judgment** — a high score means the
embedder/index is robust to superficial rewording of a message it has
already seen the clean version of; it does not mean retrieval finds the
best historical precedent for a genuinely novel complaint. The golden set
schema has no gold-evidence field (deliberately — building real relevance
judgments is exactly the kind of hand-labeling effort out of scope for
this phase), so there is currently no way to compute a non-proxy retrieval
metric at all. "MRR where defensible" in the assignment is this proxy MRR;
its defensibility limit is stated here rather than glossed over.

### Reply quality (LLM judge, `src.evaluation.judge`)

Eight axes, each scored 1–5 by the judge itself (see the rubric embedded
in `JUDGE_SYSTEM_PROMPT`): **correctness, groundedness, relevance,
helpfulness, tone, hallucination_safety, escalation_appropriateness,
overall**. `run_eval.py` reports the mean of each axis across every
successfully-judged `final_system` reply, alongside `n_scored`/`n_total`
so a partial judging run is visible as such.

**On not simply averaging scores**: `overall` is the judge's own
independent holistic score, generated in the same call as the other seven
— we do not compute it in code as `mean(other seven)`. Separately, we do
compute that plain mean ourselves and expose it as `mean_of_axes`,
specifically so a reader can see where the two diverge (e.g. a reply that
scores well on every individual axis but the judge still holistically
flags as unsafe overall) rather than silently treating "overall" and "the
average" as interchangeable. `test_judge_reply_mean_of_axes_not_same_object_as_overall`
in `tests/test_evaluation.py` exercises exactly this divergence.

**Gold/reference resolution**: when a golden example's `conversation_id`
happens to have an actual historical brand reply recorded elsewhere in the
full dataset (not the train-only retrieval corpus — see "No data leakage"
below for why that distinction matters), it's passed to the judge as a
reference. When none exists, the judge is told so explicitly and asked to
judge plausibility/safety on their own terms rather than invent what "the
right answer" would have been.

**Self-grading bias**: in this sandbox, reply drafting and judging both
run on the same `MockLLMProvider` (and in a real run with `LLM_PROVIDER=
openai`, they'd likely use the same model/configuration too, unless a
different `--llm-provider` were deliberately supplied to the judge). This
is a real limitation: a model (or a fixed deterministic function, in the
mock's case) grading its own output tends to agree with itself more than
an independent grader would. Nothing in this pipeline corrects for this —
it's flagged here and in `report.json`'s "what could be misleading" list
so it isn't mistaken for independent validation. Independent validation is
what the human/judge agreement check below is for.

### Escalation (`src.evaluation.metrics`)

- **Escalation precision** — of examples the system escalated, how many
  gold says should have escalated (labeled examples only).
- **Escalation recall** — of examples gold says should have escalated,
  how many the system actually escalated (labeled examples only).
- **False auto-handle rate** — of examples the system auto-handled, how
  many gold says should have escalated instead. This is the single most
  safety-relevant number: it's the rate of the one failure mode that
  actually reaches a customer without human review.
- **Automation coverage** — fraction of ALL golden examples the system
  chose to auto-handle. Needs no gold label at all (it's about system
  behavior, not correctness) and is always computable.
- **Safe automation coverage** — see below.

Every one of these is a `RatioMetric`: it always carries its numerator and
denominator and formats as e.g. `"62% safe automation coverage (124/200)"`
— never a bare percentage.

## The headline metric: safe automation coverage

**Definition** (`src.evaluation.metrics.safe_automation_coverage`): of all
golden examples, the fraction that were (1) auto-handled, (2) had the
correct predicted intent, (3) had an acceptable (`gold_reply_quality` ≥ 4)
reply, and (4) didn't contradict gold's escalation call (i.e.
`gold_should_escalate` was `False`).

**Two denominators are reported, and `run_eval.py` picks between them
explicitly rather than always defaulting to one:**

- `safe_automation_coverage_full_denominator` — numerator over the FULL
  golden set size (e.g. 200), as the assignment's example format
  (`"62% safe automation coverage (124/200)"`) literally specifies.
- `safe_automation_coverage_labeled_subset` — numerator over just the
  examples that actually have all three required gold labels filled in.

**Why both exist**: while labeling is incomplete, the full-denominator
version is actively misleading — an unlabeled example can never satisfy
the numerator's criteria (we can't confirm "correct intent" or "acceptable
reply" without the label), so as labeling ramps up from 0 to 200 examples,
the full-denominator number would silently *look like it's getting worse*
purely from more of the denominator becoming checkable, not from the
system actually degrading. `run_eval.py`'s headline picks
`safe_automation_coverage_labeled_subset` whenever at least one example is
labeled, and only falls back to the full-denominator version (reporting
`"N/A (0/0)"`, honestly) when literally nothing is labeled yet — which is
this sandbox's current, real state.

**If this metric can't be computed honestly at all** (as is currently the
case — 0 labeled examples), `run_eval.py` still reports it, but as
`"N/A (0/0)"` with an explanatory note, never as a fabricated percentage.

## Human / LLM-judge agreement

`data/golden/judge_validation.csv` (target: 30 rows, created automatically
by `run_eval.py` on its first run from a random sample of `final_system`
predictions) has columns `example_id`, `customer_message`, `agent_reply`,
`evidence`, `human_score`, `human_notes`, `judge_score`. `human_score`/
`human_notes` are written as the literal placeholder `"UNLABELLED"` — a
human fills these in by hand, independently, ideally without seeing
`judge_score` first (same blind-labeling discipline as
`docs/GOLDEN_SET.md`).

`src.evaluation.judge_agreement.compute_agreement` then reports:

- **Exact agreement** — fraction where `round(human_score) ==
  round(judge_score)`.
- **Mean absolute difference** — average `|human_score - judge_score|`.
- **Pearson correlation** — linear association between the two score
  series (undefined, reported as `None`, if either series has zero
  variance).
- **Weighted Cohen's kappa** (quadratic weights, via
  `sklearn.metrics.cohen_kappa_score`) — appropriate here because both
  scores are ordinal (1–5), not nominal categories, so a human=3/judge=4
  disagreement should count as "closer" than human=1/judge=5;
  quadratic-weighted kappa reflects that, plain (unweighted) kappa would
  not.

**If no `human_score` has been filled in** — the current, real state —
this reports `status: "pending_human_labels"` and the literal message
`"HUMAN VALIDATION PENDING"`, never a fabricated agreement number. This is
checked in code (`test_compute_agreement_pending_when_all_unlabelled`),
not just documented.

## No data leakage

`run_eval.py` runs automated checks before evaluating anything, and
**aborts** (`sys.exit(1)`) if either of the first two fails:

1. **Golden conversations are not in the TRAIN split** — recomputed fresh
   from the raw CSV with the configured seed, not cached/assumed.
2. **Golden conversations are not in the retrieval corpus** — the corpus
   is TRAIN-only by construction (`src.retrieval.index.
   build_resolution_corpus` doesn't even accept a `split` argument — see
   Phase 3), so this check is close to redundant with #1, but is verified
   directly rather than assumed transitively.
3. **No golden example is returned as its own retrieval evidence** — a
   per-example defensive check run after evaluation, over the actual
   evidence returned for every prediction. This should be structurally
   impossible given #1/#2, but "should be impossible given other
   invariants" is exactly the kind of assumption worth checking directly.
4. **Thresholds were not tuned on the golden set** — this is fundamentally
   a **process guarantee**, not something a script can verify after the
   fact (nothing in a config file's bytes records how its values were
   chosen). `run_leakage_checks` reports this as a `passed: null` "process
   check" with a reminder of what to attest to, rather than pretending to
   have automated something that can't be. The actual attestation: the
   current `configs/escalation.yaml` values are hand-picked engineering
   defaults chosen **before this sandbox had any golden set at all** (see
   the file's own header comment, written in Phase 4) — they were
   mechanically incapable of being tuned on golden-set outcomes that
   didn't exist yet, though obviously that's a weaker guarantee than "we
   tuned on dev and never looked at test," which is the correct procedure
   once real dev/golden data exists.
5. **Evaluation labels are not used during model fitting** — made
   mechanically true rather than merely attested: `build_systems()` in
   `run_eval.py` fits all three classifiers only on `configs/intents.yaml`'s
   worked examples, and never reads `golden_set.csv` before or during
   fitting. Reported as `passed: True` with that explanation.

The gold-reference passed to the judge (see "Reply quality" above) is a
deliberate, documented exception worth calling out explicitly: it's
sourced from the *full* dataset (all splits), not just train, because it's
used only inside a judge prompt at evaluation time — never for training a
classifier or as retrieval evidence. This is standard reference-based
grading, not leakage, but the distinction is easy to blur, hence spelling
it out here.

## Outputs

`artifacts/evaluation/`:

- **`predictions.jsonl`** — one line per `(system, golden example)` pair:
  `system`, `example_id`, `conversation_id`, `customer_message`,
  `gold_intent`/`gold_should_escalate`/`gold_reply_quality` (or `null` if
  unlabeled), `predicted_intent`, `confidence`, `retrieved_evidence` (full
  `resolution_id`/similarity/timestamp/etc. per item), `reply`,
  `grounding_score`, `decision`, `reason`, `signals`, and `judge_score`
  (only populated for `final_system`).
- **`results.json`** — per-system intent/retrieval/escalation/judge
  reports, each either `"status": "computed"` with real numbers or an
  honest pending/insufficient-data status.
- **`results.csv`** — the same escalation + intent-accuracy numbers
  flattened into one row per `(system, metric)` for quick scanning/
  spreadsheet import.
- **`confusion_matrix.png`** — `final_system`'s intent confusion matrix,
  or (currently) a plot that says `PENDING` in place of a heatmap, since
  there's nothing to plot with zero labeled examples. The file always
  exists, as required; its content honestly reflects the data available.
- **`report.json`** — the headline number, the "what could be misleading
  about this number" list (mirroring this project's running theme from
  earlier phases), leakage check results, human/judge agreement status,
  and per-system failure analysis.

## Failure analysis (`src.evaluation.failure_analysis`)

Named categories, each carrying the actual `example_id`s involved (never
just a count):

- **Always computable** (no gold needed): `llm_parse_error`,
  `out_of_distribution`, `low_grounding_reply`,
  `judge_flagged_hallucination_risk`, `judge_flagged_low_overall`.
- **Only computable over labeled examples**: `intent_misclassified`,
  `false_auto_handle`, `unnecessary_escalation` — each reports its own
  `eligible_count` (how many golden examples actually had the relevant
  gold label) alongside its failure count, so "0 failures" is
  distinguishable from "0 examples were even checkable."

## Known limitations, stated plainly

- **No real dataset** → no real brand, no real taxonomy, no real
  training data, no real golden labels. Every number this pipeline
  currently produces is either a mechanical self-consistency check or an
  honest "pending" status — see the top of this document.
- **Bootstrapped classifiers**: all three intent classifiers are trained
  on ~1-2 worked examples per intent from `configs/intents.yaml`, not a
  real labeled corpus. Expect (and don't be surprised by) weak, low-
  confidence predictions in any demo run against the synthetic fixture.
- **Retrieval evaluation is a proxy**, not human relevance judgment (see
  above).
- **Self-grading**: the reply-drafting and judging LLM calls currently
  share a provider/configuration (see above).
- **Small-sample noise**: with 2 golden examples in the shipped demo run,
  every ratio metric is extremely noisy; none of the specific numbers in
  a demo `artifacts/evaluation/` run should be quoted as if they meant
  anything beyond "the pipeline executed successfully."
