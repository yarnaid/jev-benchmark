# Question model v2: multi-label category, 0–100 scores, new questions

Sub-project **A** of the 2026-09-25 improvement request (A → B → C, see §11). It builds on
`2026-09-24-jev-benchmark-design.md`; everything not mentioned here is unchanged.

**Revision 2 (2026-09-25, before implementation).** Every column keeps returning probabilities that **sum to
1**. The multi-label set is *read* from that distribution: a label applies when its probability is at least
`threshold × max(p)`. Revision 1 had asked for independent per-label probabilities; this replaces it. As a
result:
- Jev keeps its native `choice` primitive (no unfolding into `noul`s);
- the LLM schema, the LLM prompts and the embeddings classifier are unchanged;
- runs made before v2 stay comparable on `category` and `sentiment`.

## 1. Intent

- An email can belong to **several categories at once** (for example marketing *and* phishing). Every column
  returns a probability per category (summing to 1). The rule for labels:
  - the probabilities are normalized by the most probable category, so the top category becomes 1.0;
  - every category whose normalized value reaches the **editable threshold** (default 80%) is a positive
    label.

  The top category is therefore always a label.
- Every ordered question (urgency, importance, sentiment, confidentiality) also gets a **0–100 score** that a
  manager can read at a glance.
- The question set gains seven questions that matter for automated email processing and risk.

**Success criteria**

1. `config/questions.toml` is `email-triage-v2` (§2) and loads.
2. All four columns (Jev, Anthropic, OpenAI, embeddings) produce category distributions from which label sets
   are read, and the generator produces a multi-label reference.
3. The comparison report and the Explorer show multi-label metrics at a threshold that can be changed in the
   UI without a re-run.
4. Runs made with v1 stay comparable on every question whose option ids are unchanged. That includes
   `category` (choice → multi) and `sentiment` (choice → score).
5. Gates: ruff, pyright (0 errors), default suite < 5 s, coverage ≥ 95%, `node --check`, `node --test`.
6. A paid real smoke on **10 emails** (§10) is run only after explicit user approval.

**Out of scope:** everything listed in §11 (sub-projects B and C).

## 2. Question set `email-triage-v2`

Existing questions keep their ids, types and option ids unless listed below. Order in the file = display order.

| # | id | type | status |
|---|---|---|---|
| 1 | `category` | **multi** ×20, `threshold = 0.8` | changed (was choice) |
| 2 | `urgency` | score ×5 | unchanged |
| 3 | `importance` | score ×4 | unchanged |
| 4 | `sentiment` | **score** ×3 | changed (was choice); the option order is already lowest first |
| 5 | `confidentiality` | score ×4 | new |
| 6 | `needs_reply` | noul | unchanged |
| 7 | `action_required` | noul | unchanged |
| 8 | `deadline` | noul | new |
| 9 | `attachment_review` | noul | new |
| 10 | `delegatable` | noul | new |
| 11 | `escalation` | noul | new |
| 12 | `skippable` | noul | unchanged |
| 13 | `should_delete` | noul | unchanged |
| 14 | `is_automated` | noul | new |
| 15 | `llm_safe` | noul | unchanged |
| 16 | `malicious` | noul | unchanged |
| 17 | `impersonation` | noul | new |
| 18 | `sensitive_data` | noul | unchanged |

Wording of changed and new questions:

- **`category`**: "Which categories describe this email? Several categories can apply at once, for example
  marketing and phishing; judge each category independently." Options: the same 20 ids and descriptions as v1.
- **`sentiment`**: "How positive or negative is the overall tone of the sender?"
  `negative` → `neutral` → `positive`, v1 descriptions kept.
- **`confidentiality`**: "How confidential is the information in this email, as a data-handling
  classification?"
  - `public`: "Could be published without harm"
  - `internal`: "Meant for the organization or the recipient; little harm if it leaked"
  - `confidential`: "Business, personal or financial details whose leak would cause harm"
  - `restricted`: "Highly sensitive: credentials, legal matters, health data or strategic plans whose leak
    would cause serious harm"
- **`deadline`**: "Does the email state or imply a specific deadline, due date or expiry that the recipient
  must meet?"
  - yes: "A concrete date, time or time limit for the recipient's response or action is given"
  - no: "No deadline, due date or expiry applies to the recipient"
- **`attachment_review`**: "Does the email ask the recipient to open, review, sign or approve an attached or
  linked document?"
  - yes: "An attached or linked document must be opened, reviewed, signed or approved"
  - no: "No document needs to be opened, reviewed, signed or approved"
- **`delegatable`**: "Could an assistant or AI agent fully handle this email on the recipient's behalf,
  without needing the recipient's own judgement or authority?"
  - yes: "Routine: it can be filed, answered or acted on by following standard rules"
  - no: "It needs the recipient's personal judgement, authority, knowledge or relationships"
- **`escalation`**: "Should this email be escalated to a manager or a specialist team such as legal,
  security, finance or HR?"
  - yes: "It raises a risk, dispute, incident or decision beyond the recipient's normal remit"
  - no: "The recipient can handle it within their normal remit, or it needs no handling"
- **`is_automated`**: "Was this email sent by an automated system or a bulk-mailing tool rather than written
  by a person for this recipient?"
  - yes: "Sent automatically or in bulk: notifications, alerts, receipts, newsletters or campaigns"
  - no: "Written by a person for this recipient or conversation"
- **`impersonation`**: "Does the sender pretend to be a person, executive, organization or brand they are
  not?"
  - yes: "The sender falsely claims an identity, for example a spoofed brand, a fake executive or a look-alike
    domain"
  - no: "The sender appears to be who they claim to be"

## 3. Types and invariants

### 3.1 `questions.py`

- `MultiQuestion(type="multi", threshold: float = 0.8)`, with the validator `0 < threshold <= 1`. It joins
  `AnyQuestion` and the `Question` discriminated union.
- **Hard answers** (`HardAnswer = str | list[str]`) come from the generator reference and from human labels.
  A multi question accepts a single option id (a one-label set; old v1 references are strings) or a
  non-empty list of unique option ids.
- `hard_distribution(question, answer)` turns a hard answer into its distribution, or `None` when the answer
  is invalid for the question:
  - `one_hot` for choice, score and noul;
  - for multi, the **uniform distribution over the labels**, e.g. `["spam", "phishing"]` →
    `{spam: 0.5, phishing: 0.5, others: 0}`. Normalizing it by its max gives back exactly the same label
    set.
- `with_threshold(questions, threshold)` replaces every multi question's threshold; `None` keeps them.
- `render_questions` gets a kind hint for `multi`: "multi-label: one or more options can apply".
- `compatible()` becomes shape-based. Choice, score and multi answers are all distributions over their
  options, so any two of them are compatible when their option ids are equal and in the same order. A noul is
  compatible only with a noul. This is what keeps v1 runs comparable on `category` and `sentiment`. The
  threshold is a reading parameter, not part of compatibility.

### 3.2 Invariant (unchanged)

Every answer from every source is a distribution summing to 1: choice, score and multi alike, with `noul` as
`{yes, no}`. For a multi question the **applied labels** are the options with `p >= threshold × max(p)`
(ties at the boundary are applied). The top option is always applied, so a valid answer never has an empty
label set.

### 3.3 Score (0–100)

`metrics/distributions.py::score_0_100(matrix) = expected_level(matrix) / (k − 1) × 100`. Hard labels go
through the same function. This is the only implementation.

## 4. Columns

- **Jev** (`classifiers/jev.py`): a multi question is sent as a native `choice` (instructions + criteria)
  and parsed as a `choice` answer. It is the same call and the same primitive as today, so Jev's recommended
  usage is kept. Only the type sent and expected differs (`multi` → `choice`).
- **Chat LLMs:** no change. The schema already asks for one number per option; the parser already normalizes;
  the prompts already say the probabilities of one question sum to 1. The `category` instructions tell every
  model that several categories can apply.
- **Embeddings:** no change (softmax over the options). With a softmax distribution the relative rule means:
  a label applies iff `cos_best − cos_label ≤ τ · ln(1 / t)` (≈ 0.011 at τ = 0.05, t = 0.8).
- `jev_output_reserve` (1000) stays. The paid smoke (§10) reports Jev's real output tokens.

## 5. Generation, labels

- **Answers type:** `GeneratorOutput.answers` and `Email.reference_answers` become `dict[str, str | list[str]]`.
  Old `str` values still load.
- **Schema:** for multi, `{"type": "array", "items": {"type": "string", "enum": [...]}}`. There is no
  `minItems`, because enforcement by every generator provider under strict mode is not verified.
- **Parse:** a multi answer must be a valid hard answer (§3.1). Otherwise the output is unusable and goes
  through the existing same-model retry (`max_attempts = 3`).
- **Generator system prompt:** "choosing exactly one option id per question" → "choosing exactly one option id
  per question, or every option id that applies (at least one) for a multi-label question".
- **Category trait:** keeps stratifying over the 20 options; the trait is the email's primary category.
  `count_mismatches`: for a list answer the trait is contradicted iff the requested value is **not in** the
  list.
- **Label store and route:** `store/labels.py` and `PUT /api/labels/{id}` accept `str | list[str]`.
  - A list must be non-empty with unique valid ids.
  - `null` or `[]` clears the label.
  - A list for a non-multi question is a 400.

## 6. Metrics (`metrics/multilabel.py`, new, pure numpy)

Input: `(n, k)` distributions; `t` = threshold.

| function | definition | edge cases |
|---|---|---|
| `relative_labels(m, t)` | `m >= t × max(m)` row-wise (bool) | an all-zero row → no labels; boundary ties are applied |
| `exact_match_rows(a, b)` | 1.0 per row with identical label sets | |
| `jaccard_rows(a, b)` | `|a∧b| / |a∨b|` per row | both empty → 1.0 |
| `micro_f1(a, b)` | `2TP / (2TP + FP + FN)` over all cells | no positives → `None` |
| `macro_kappa(a, b)` / `macro_kappa_batch` | mean binary Cohen's κ per label over the labels where it is defined | none defined → `None` / NaN |
| `macro_fleiss(labels)` | `(n, raters, k)` bool → mean binary Fleiss' κ per label | none defined → `None` |

`metrics/agreement.py` gains `brier_to_target(p, target)` (the multi-class Brier against any target
distribution). `brier_score` is expressed through it (one-hot target), so there is still one formula.

## 7. Comparison (`compare/`)

- **Raters:** reference and human hard answers go through `hard_distribution` (§3.1).
- **Pairs.** A multi question gets its own function, and the single-label path is untouched:

  | field | multi value |
  |---|---|
  | `agreement` (+ CI) | exact-set match |
  | `kappa` (+ CI) | macro κ over labels |
  | `jsd` | as for choice (distributions) |
  | `brier` | `brier_to_target` against the hard rater's distribution |
  | `pearson` | `None` |
  | new `jaccard` (+ `jaccard_ci`) | |
  | new `f1` | |

  The new fields are `None` for other types.
- **Report:**
  - `QuestionReport.type` adds `"multi"`, plus `threshold` (the one used).
  - `RaterStats` gains, for multi only, `label_counts` (emails where each label is applied) and
    `mean_labels`. For score questions it gains `mean_score` (0–100).
  - Entropy, confidence and argmax counts for multi are computed as for choice.
  - Fleiss uses `macro_fleiss` over the applied labels.
- **Rows:**
  - `EmailRow.top[run][q]` for multi is the list of applied labels, sorted by p descending (ties in option
    order);
  - there are new `scores[run][q]` and `reference_scores[q]` (0–100, score questions);
  - `reference` / `human` may hold lists;
  - the disagreement index is unchanged (JSD over distributions).
- **Threshold override:** `compare(..., threshold=None)` / `email_rows(..., threshold=None)` override every
  multi question's threshold.
- **Base question set** of `GET /api/compare`: the snapshot of the **most recent** selected run (by
  `created_at`, ties by id). Then URL order never decides how a question is read. With compatible v1/v2
  runs, it decides whether `category` is read as single-label or multi-label.

## 8. API

- `GET /api/compare?runs=…&threshold=` and `GET /api/emails?…&threshold=`:
  - an optional float with 0 < t ≤ 1, finite (422 otherwise);
  - passed through to §7.
- `GET /api/emails/{id}` keeps returning raw predictions, with `human: dict[str, str | list[str]]`. The UI marks
  labels with `p >= threshold × max(p)` for each run.
- `PUT /api/labels/{id}`: see §5.

## 9. Minimum UI (full polish is sub-project B)

- **Threshold slider:**
  - it lives in the page chrome (`#threshold-control`) on the Benchmark and Explorer pages;
  - range 50–100%, step 5, default = the question's threshold;
  - 300 ms debounce, then re-fetch with `?threshold=`;
  - remembered via `storage.js`;
  - rendered once, so dragging never loses focus.
- **`report.js`:**
  - a multi card has: a chart of `label_counts`; a rater table (n · labels/email · entropy · confidence); a
    pair table (exact match · Jaccard · F1 · κ (macro) · JSD · Brier);
  - score cards show "Mean score (0–100)".
- **Explorer table:**
  - a multi cell lists the applied labels; its class is `cell-match` / `cell-partial` (overlap, amber) /
    `cell-mismatch`, from a pure `setMatch` helper under `node --test`;
  - score cells show `level · score`;
  - the reference filter matches list membership.
- **Explorer detail:**
  - a multi card has: a `≥ N% of top` badge; per run, a tick on each bar at `threshold × max(p)`; bold applied
    labels; ✓ on reference labels; human labels as checkboxes;
  - score cards have a `score (0–100)` footer row.
- **Explorer default runs:** without `?runs=`, pre-select the latest completed run per column on exactly the
  selected generations.
- All DOM is built with `h()`. There is no `innerHTML`.

## 10. Testing and verification

- TDD, one test file per module, `pytest.param` tables, OpenRouter always mocked.
- New `tests/test_metrics_multilabel.py`:
  - tables for the relative rule: exactly at the boundary, flat rows, an all-zero row;
  - set metrics and κ / Fleiss on worked examples;
  - Hypothesis properties: Jaccard symmetric and bounded; exact ≤ Jaccard; the top option always applied;
    monotone in t.
- New rows in existing tables:
  - Jev maps multi ↔ choice;
  - the LLM and embeddings treat multi as a distribution;
  - generator list validation and retry;
  - `count_mismatches` membership;
  - labels PUT;
  - `?threshold=` bounds;
  - v1 choice ↔ v2 multi compatibility;
  - base = newest run.
- A browser smoke against a scratch server with a fake OpenRouter.
- **Real smoke (paid, only after explicit approval):**
  - `jev-bench generate --count 10` on v2, then one run per column;
  - check that labels are read from every column, the label counts per email, and Jev's output tokens;
  - expected cost ≈ $0.15–0.25.

## 11. Follow-up sub-projects (captured so the request is not lost; each gets its own spec)

**B: UI and Benchmark upgrades.**
- Tooltips and Help:
  - a (?) tooltip next to every metric, written for non-technical managers (what it is, how to read it);
  - a Help page with all descriptions and the methodology;
  - one glossary module feeds both.
- Look and feel:
  - more icons on the email form (From/To/Cc/Sent/Generator/Traits);
  - more color while staying executive-grade;
  - a near-black dark theme.
- Explorer detail panel: narrower, with little empty space, correct values highlighted and sorted to the top.
- Benchmark page:
  - speed and cost relative to Jev (Jev 3 s, Claude 12 s → Jev ×1, Claude ×4);
  - selectable runs;
  - a preliminary cost estimate before each run.
- API key:
  - a browser-entered key overrides the server key, reversing the current "server key wins" invariant. The
    CLAUDE.md key-safety section is updated;
  - instructions on how to get an OpenRouter key.

**C: Analyze tab.**
- Pick an analyst LLM, a generation and runs; run an analysis job with a robust, editable prompt; show and
  persist the result.
- Input: the aggregate comparison report + a compact per-email table (top answer and confidence per run per
  question) + the full text of the N most-disputed emails.
