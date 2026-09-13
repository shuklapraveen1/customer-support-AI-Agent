# Decision log

Fifteen non-obvious decisions made while building this project, in the
order the assignment listed the required topics. Each includes what was
decided, why, and what it costs — a decision without a stated tradeoff
usually means the tradeoff wasn't looked at hard enough.

---

**1. Conversation-level split, not tweet-level**

*Decision*: `src/data/split.py` splits on `conversation_id`; no function
anywhere in this repo accepts a way to split individual tweets.

*Reason*: A conversation's turns are causally linked (a reply only makes
sense next to what it's replying to). Splitting turns of the same
conversation across train/test would leak conversational context — and
the specific historical resolution — across the split boundary.

*Tradeoff*: Conversations vary a lot in length, so the split is exact at
the conversation level but only approximate at the tweet/row level; a
brand with unusually long conversations could shift the effective
row-count ratio away from 70/15/15.

---

**2. Brand selection is a human decision, never automatic**

*Decision*: `src.analysis.brands` ranks and recommends candidate brands
but never picks one; `scripts/run_all.py` exits and asks for `--brand`
if none is given.

*Reason*: The brand choice has downstream consequences (taxonomy content,
corpus size, risk profile) that deserve a human glance at the actual
report, especially since the composite recommendation score's weights
(0.4 pairs / 0.3 volume / 0.3 repetition) are a reasonable but ultimately
arbitrary choice.

*Tradeoff*: `run_all.py` isn't a single fully-hands-off command on a fresh
checkout — it requires one manual decision partway through.

---

**3. All three intent classifiers are bootstrapped on the taxonomy's own examples, not gold labels**

*Decision*: `MajorityClassifier`, `TfidfLogisticClassifier`, and
`EmbeddingNearestCentroidClassifier` are all fit on nothing but the 1-2
worked examples per intent already in `configs/intents.yaml` — never on
`golden_set.csv`.

*Reason*: No real hand-labeled intent training set exists in this
sandbox. Bootstrapping this way makes "evaluation labels are not used
during model fitting" mechanically true by construction, not just a rule
to remember to follow.

*Tradeoff*: Every classifier will have weak, sometimes near-arbitrary
confidence until a real training set replaces the bootstrap — visible
directly in the `ambiguous_intent` failure category, not hidden.

---

**4. Banking77 excluded as the actual taxonomy**

*Decision*: Banking77 is referenced in `src/analysis/intents.py`'s
docstring only as a methodological example of a well-scoped taxonomy; it
is never used as this project's real intent labels.

*Reason*: Banking77 is banking-specific; its labels (card issues,
foreign-currency transfers) don't describe general Twitter customer
support for an arbitrary consumer brand.

*Tradeoff*: Without a ready-made validated label set to fall back on, the
real taxonomy has to be earned from clustering the chosen brand's own
data — which, per this sandbox's dataset-access constraints, hasn't
happened yet (`configs/intents.yaml` ships as a hand-written placeholder).

---

**5. The retrieval corpus is TRAIN-split only, unconditionally**

*Decision*: `build_resolution_corpus()` doesn't accept a `split`
parameter at all.

*Reason*: Removing the parameter removes the possibility of a caller
accidentally pointing it at dev/test — the leakage-safety guarantee lives
in the function's signature, not in caller discipline.

*Tradeoff*: Less flexible if a future need arises for a dev-split corpus
(e.g. for threshold tuning against dev) — that would require a
deliberate new function, which is the intended friction, not an oversight.

---

**6. Historical replies are evidence, never guaranteed-correct**

*Decision*: The reply-drafting system prompt instructs the model to
synthesize a reply "informed by the pattern of past resolutions," never
to copy one verbatim, and the golden-set schema deliberately has no
"gold reply" field for the model to be handed and parrot.

*Reason*: Twitter data carries no signal that a historical reply actually
resolved the customer's issue — treating it as verified truth would be
false confidence dressed up as grounding.

*Tradeoff*: `grounding_score` is a self-reported model confidence, not a
hard guarantee. The pipeline still has to escalate defensively rather
than assume "retrieved evidence" means "correct answer."

---

**7. Escalation is a 12-rule ordered ladder, not `confidence < threshold`**

*Decision*: `src/agent/escalation.py`'s `decide()` checks safety
keywords, parse errors, out-of-distribution retrieval, high-risk
intents/keywords, evidence volume, message length, intent confidence, an
"ambiguous-but-technically-passing" band, retrieval similarity, evidence/
intent agreement, and reply grounding — in that order, first match wins.

*Reason*: A single confidence threshold can't distinguish "confidently
wrong because of a legal threat" from "confidently right about a routine
question" — the assignment explicitly asked for more than one signal.

*Tradeoff*: Twelve thresholds/lists in `configs/escalation.yaml` to
configure and eventually tune, versus one number. Rule *order* is itself
a design choice with interaction effects (e.g. whether OOD or high-risk
gets checked first matters when both apply to the same message) that
aren't independently justified per pair.

---

**8. Out-of-distribution detection is a hard floor, separate from the soft similarity minimum**

*Decision*: `ood_similarity_threshold` (always escalate below it) is
distinct from, and stricter than, `min_retrieval_similarity` (one signal
among several, checked later in the ladder).

*Reason*: Below a certain similarity, there is no legitimate historical
basis for a reply at all — this should escalate unconditionally rather
than being outvoted by an otherwise-confident intent prediction.

*Tradeoff*: Two similarity thresholds to explain and tune instead of one,
and their relative ordering (`ood < min_retrieval`) is itself an untested
assumption pending real similarity-score distributions.

---

**9. Safe automation coverage reports two denominators, not one**

*Decision*: `src.evaluation.metrics` computes both
`safe_automation_coverage_full_denominator` (against the whole golden
set) and `safe_automation_coverage_labeled_subset` (against only labeled
examples); `run_eval.py`'s headline picks whichever is non-empty.

*Reason*: While labeling is incomplete, the full-denominator version
would make the number look like it's getting *worse* purely because more
of the denominator becomes checkable over time — a measurement artifact,
not a real regression.

*Tradeoff*: Two numbers to explain instead of one. A reader skimming
`report.json` could still mistake the labeled-subset version for a final
answer once only a handful of examples are labeled — the "what could be
misleading" list exists specifically to counter this, but only if it's
actually read.

---

**10. The golden set's sampling strata come from the system's own predictions, never from gold labels**

*Decision*: `scripts/create_golden_template.py` uses the current system's
intent/confidence/decision to bucket candidates for representative
sampling, but writes every `gold_*` field as `"UNLABELLED"` and tells the
annotator explicitly not to anchor on the system's suggestion.

*Reason*: Sampling needs *some* signal to guarantee coverage of rare/
ambiguous/escalate-worthy cases; the system's own (imperfect) predictions
are a defensible source for that without contaminating the actual labels.

*Tradeoff*: If the system has a systematic blind spot (e.g. it never
predicts some rare intent), stratifying off its own predictions could
under-sample exactly the cases that would reveal that blind spot. A
fully independent stratification (e.g. off raw keyword clusters) might
surface more surprises, at the cost of being less structured.

---

**11. The LLM judge and the reply-drafting LLM currently share a provider — flagged, not fixed**

*Decision*: `run_eval.py` doesn't force a different provider/model for
judging vs. drafting; both default to the same `--llm-provider`.

*Reason*: Only one real provider (`openai`) is implemented, and this
sandbox has no API key to use it with — enforcing "use a different judge
model" here would be theater without a second real option to point at.

*Tradeoff*: Self-grading bias is real and currently uncorrected —
documented prominently in `docs/EVALUATION.md` and in `report.json`'s
misleading-number list, but documentation isn't the same as fixing it.

---

**12. Fast mode subsamples rows, accepting orphaned reply links**

*Decision*: `scripts/run_all.py --fast` randomly subsamples raw CSV rows,
rather than subsampling whole conversations.

*Reason*: Row-level subsampling is simpler and faster, and the
normalization/conversation-reconstruction code already treats a broken
`in_response_to` link as an "orphan reply" (its own singleton
conversation) rather than crashing — degraded-but-safe behavior already
existed from Phase 1, so fast mode gets it for free.

*Tradeoff*: A fast-mode run has an inflated proportion of single-turn
"conversations" and a lower resolution-pair yield than a full run — its
numbers are useful for verifying the pipeline runs correctly and quickly,
not for comparing against a full run's results.

---

**13. Timestamps are preserved as-is, including missing, never inferred**

*Decision*: A malformed timestamp doesn't drop a row (it's kept with
`created_at=None` and a flag); a missing timestamp is preserved as `None`/
`null` all the way through retrieval results, never backfilled with a
guess (e.g. "now" or the corpus median).

*Reason*: Silently inferring a timestamp would misrepresent
recency-sensitive decisions (retrieval's recency tiebreak, the "outdated
historical response" failure heuristic) with fabricated confidence.

*Tradeoff*: Both the recency tiebreak and the outdated-evidence heuristic
degrade to "no information" for records with missing timestamps,
understating how outdated some evidence might actually be — safer than
guessing, but not free of cost.

---

**14. One shared embedding cache, keyed by content hash, everywhere**

*Decision*: `EmbeddingCache` keys strictly on `sha256(text)` + embedder
name, used identically across intent classification, retrieval-corpus
building, and golden-set stratification.

*Reason*: The same message text is genuinely embedded multiple times
across different pipeline stages; one shared cache format means none of
that work repeats — directly relevant to the 15-minute fast-mode budget.

*Tradeoff*: The cache invalidates only on embedder *name*, not on a
hash/version of the embedder's actual weights — swapping in an updated
checkpoint under the same model name would silently serve stale
embeddings unless the cache directory is cleared by hand.

---

**15. Baselines are majority + TF-IDF, not two variations of the same idea**

*Decision*: The two required baselines are a majority-class classifier
(ignores text entirely) and TF-IDF+LogisticRegression (a real but simple
text classifier), each paired with the shared `TfidfRetrievalBaseline` for
retrieval — not, say, two different embedding models.

*Reason*: A floor (majority) and a serious-but-simple text-aware
competitor (TF-IDF) together make the "final system" demonstrate it beats
both "no signal at all" and "text signal without semantics" — two
embedding variants would both cluster near the final system and be a
less informative comparison.

*Tradeoff*: TF-IDF and the final embedding classifier share almost no
architecture, so when they disagree it's hard to attribute the gap to
"semantics helped" versus "the specific bootstrap examples happened to
favor one representation." A real ablation would need more controlled
comparisons than three architecturally-unrelated systems.
