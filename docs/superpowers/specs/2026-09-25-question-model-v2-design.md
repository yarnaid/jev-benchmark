# Question model v2: multi-label category, 0–100 scores, new questions

Sub-project **A** of the 2026-09-25 improvement request (A → B → C, see §11). Builds on
`2026-09-24-jev-benchmark-design.md`; everything not mentioned here is unchanged.

## 1. Intent

- An email can belong to **several categories at once** (for example marketing *and* phishing). Every column
  returns an **independent probability per category**; a label is *applied* when its probability reaches an
  **editable threshold** (default 80%).
- Every ordered question (urgency, importance, sentiment, confidentiality) also gets a **0–100 score** that a
  manager can read at a glance.
- The question set gains seven questions that matter for automated email processing and risk.

**Success criteria**

1. `config/questions.toml` is `email-triage-v2` (§2) and loads.
2. All four columns (Jev, Anthropic, OpenAI, embeddings) produce multi-label answers, and the generator
   produces a multi-label reference.
3. The comparison report and the Explorer show multi-label metrics at a threshold that can be changed in the
   UI without a re-run.
4. Runs made with v1 remain comparable on the unchanged questions.
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
| 4 | `sentiment` | **score** ×3 | changed (was choice), options reordered lowest first |
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
- `multi_hot(question, option_ids) -> Distribution`. Rules:
  - an id that is not an option raises `ValueError`;
  - duplicates are ignored.
- `render_questions` gets a kind hint for `multi`: "multi-label: judge each option independently; several can
  apply".
- `compatible()` is unchanged: it already compares type and option ids. The threshold is a display/decision
  parameter, not part of compatibility.

### 3.2 Amended invariant

Old: "every answer is a distribution summing to 1". New:

- for `choice` / `score` / `noul`, an answer is a distribution summing to 1 (unchanged);
- for `multi`, an answer is a vector of **independent** probabilities in [0, 1], one per option, never
  renormalized.

The `Distribution` type alias stays `dict[str, float]`; the semantics are decided by the question type. The
same amendment goes into CLAUDE.md.

### 3.3 Score (0–100)

`metrics/distributions.py::score_0_100(matrix) = expected_level(matrix) / (k − 1) × 100`. Hard labels
(reference, human) are one-hot rows, so they go through the same function. This is the only implementation;
views never recompute the score themselves.

## 4. Columns

### 4.1 Jev (`classifiers/jev.py`)

Jev's documented primitives are `choice` / `score` / `noul`, and there is no multi-label type. A multi question
is therefore **unfolded** into one `noul` per option, all in the same Decisions call:

- key `f"{question.id}__{option_id}"`. The mapping key → (question, option) is built once at payload
  construction; nothing parses ids back;
- instructions `f"Does this label apply to the email: {description}? Several labels can apply to one email."`
  (self-contained, because Jev never sees ids);
- criteria:
  - `true`: the option description;
  - `false`: "This label does not apply to the email".

**Folding back:** the `noul` values of the sub-answers become `{option: p}`.

- A missing or mistyped sub-answer sets that option to 0.0 and adds a note
  `category: <n> label answers missing, set to 0`.
- If **all** of a question's sub-answers are missing, that question is a parse error, which is the same handling
  as a missing question today.

`jev_output_reserve` (1000) stays as it is. The real smoke (§10) checks real output usage, since Jev now
answers ~37 sub-questions instead of 11. If usage exceeds ~80% of the reserve, the reserve is raised in
`config/benchmark.toml`.

### 4.2 Chat LLMs (`classifiers/llm_schema.py`, `llm_parse.py`, `config/benchmark.toml`)

- **Schema:** a strict object with one `number` per option, described as "Independent probability from 0 to 1
  that each label applies; several labels can be likely at once; the values need not sum to 1."
- **Parse:** each value goes through `unit_probability` and is **not** renormalized.
  - A non-object value is a per-question parse note, the same as today.
  - Missing or non-numeric options become 0.0 with a note.
  - If every option is missing, it is a parse note (the question is unusable).
  - An all-zero vector is a **valid** answer: "no label applies".
- **System prompts** (`system_prompt` and `system_prompt_all_in_one`) gain one bullet: "for a multi-label
  question, give each label's probability independently; several labels can be likely at once and they need
  not sum to 1;".
- Output-token estimate: the number of values per email grows only from ~40 to ~49 (category was already 20
  numbers), so `est_output_tokens_per_email = 400` stays.

### 4.3 Embeddings (`classifiers/embeddings.py`)

- For `multi`: `p = softmax(cos / τ) / max(softmax(cos / τ))`, which equals `exp((cos − cos_max) / τ)`. So:
  - the best-matching label is always 1.0;
  - another label is applied at threshold `t` iff `cos_best − cos_label ≤ τ · ln(1 / t)` (≈ 0.011 at τ = 0.05,
    t = 0.8).
- This is a documented heuristic. Embeddings cannot produce calibrated independent probabilities without tuning,
  and tuning on references is forbidden.
- Similarities stay stored as today.

## 5. Generation (`generation/prompt.py`, `emails.py`, `config/generation.toml`)

- **Answers type:** `GeneratorOutput.answers` and `Email.reference_answers` become `dict[str, str | list[str]]`.
  Old `str` values still load.
- **Schema:** for multi, `{"type": "array", "items": {"type": "string", "enum": [...]}}`. There is no
  `minItems`, because enforcement by every generator provider under strict mode is not verified.
- **Parse:** a multi answer must be a **non-empty list of unique valid ids**. Otherwise the output is unusable
  and goes through the existing same-model retry (`max_attempts = 3`). The stored order is the generator's
  order.
- **Generator system prompt:** "choosing exactly one option id per question" → "choosing exactly one option id
  per question, or every option id that applies (at least one) for a multi-label question".
- **Category trait:** keeps stratifying over the 20 options. The trait is the email's *primary* category, and
  the generator may add more. `count_mismatches` counts the trait as contradicted iff the requested value is
  **not in** the list (for `str` answers the check is unchanged).
- **Label store and route:** `store/labels.py` and the `PUT /api/labels/{id}` route accept `str | list[str]`.
  - A multi value must be a list of unique valid ids.
  - `null` or `[]` clears the label.
  - A `str` for a multi question, or a list for a non-multi question, is a 400.

## 6. Metrics (`metrics/multilabel.py`, new, pure numpy)

Input: `(n, k)` matrices of independent probabilities; `t` = threshold.

| function | definition | edge cases |
|---|---|---|
| `binarize(m, t)` | `m >= t` (bool) | p exactly at t is applied |
| `exact_match(a, b)` | mean over rows of `all(a == b)` | n = 0 → `ValueError`, as `percent_agreement` |
| `jaccard_rows(a, b)` | per row `|a∧b| / |a∨b|` | both empty → 1.0 |
| `micro_f1(a, b)` | `2TP / (2TP + FP + FN)` over all cells | no positives on either side → `None` |
| `macro_kappa(a, b)` | mean of binary Cohen's κ per label (via `cohen_kappa`, k = 2) over labels where it is defined | none defined → `None` |
| `binary_entropy(m)` | per row, mean over labels of H(p) in bits | p ∈ {0, 1} → 0 |
| `binary_jsd(p, q)` | per row, mean over labels of JSD([p, 1−p], [q, 1−q]) via `js_divergence` | within [0, 1] |
| `label_brier(p, y)` | mean over rows and labels of (p − y)² | range [0, 1] |
| `macro_fleiss(labels)` | `(n, raters, k)` bool → mean of `fleiss_kappa(k=2)` per label over defined labels | none defined → `None` |

Bootstrap: CIs for exact match and Jaccard come from per-row values indexed by the shared `resample_index`, and
the macro-κ CI from `confusion_batch` per label with `nanmean` across labels. These reuse
`metrics/bootstrap.py`.

## 7. Comparison (`compare/`)

- **`raters.py`:** the reference and human raters build multi-hot rows from lists (`multi_hot`); `str` answers
  are unchanged.
- **`pairs.py`:** a `multi` branch, kept in its own function so the single-label path is untouched:

  | field | multi value |
  |---|---|
  | `agreement` / `agreement_ci` | exact-set match |
  | `kappa` / `kappa_ci` | macro κ |
  | `jsd` | mean binary JSD |
  | `pearson` | `None` |
  | `brier` | `label_brier` vs a hard rater (else `None`) |
  | new `jaccard`, `jaccard_ci` | mean Jaccard |
  | new `f1` | micro-F1 |

  The new fields are optional and `None` for non-multi questions.
- **`report.py`:**
  - `QuestionReport.type` adds `"multi"`, and there is a new `threshold: float | None`, which is the one
    actually used.
  - `RaterStats` gains:
    - `label_counts` (multi: emails where each label is applied);
    - `coverage` (multi: share of emails with ≥ 1 applied label);
    - `mean_labels` (multi);
    - `mean_score` (score questions, 0–100).
  - For multi:
    - `argmax_counts` is `{}`;
    - `mean_entropy` = mean `binary_entropy`;
    - `mean_confidence` = mean over labels of `max(p, 1 − p)`;
    - Fleiss uses `macro_fleiss`.
- **`rows.py`:**
  - for multi, `EmailRow.top[run][q]` is the list of applied labels sorted by p descending (possibly empty);
  - new `EmailRow.scores[run][q]` and `EmailRow.reference_scores[q]` (0–100, score questions only; the
    reference goes through the same `score_0_100` on its one-hot row);
  - the disagreement index uses `binary_jsd` for multi;
  - `reference` / `human` become `dict[str, str | list[str]]`.
- **Threshold override:** `compare(..., threshold: float | None = None)` and
  `email_rows(..., threshold: float | None = None)`. `None` means each multi question's own default. The single
  override is enough while only one multi question exists; a per-question map waits for a second one.
- **Base question set** of `GET /api/compare`: the snapshot of the **most recent** selected run (by
  `created_at`, ties by id) instead of `metas[0]`. Then URL order cannot decide whether v1 or v2 semantics are
  compared.

## 8. API

- `GET /api/compare?runs=…&threshold=` and `GET /api/emails?generations=…&runs=…&threshold=`:
  - `threshold` is an optional float with 0 < t ≤ 1, enforced as a FastAPI `Query` constraint (422 otherwise);
  - it is passed through to §7.
- `GET /api/emails/{id}` keeps returning raw predictions, and `EmailDetail.human` becomes
  `dict[str, str | list[str]]`. The UI marks labels ≥ the threshold, using the question's default
  (`questions` payload) or the slider value.
- `PUT /api/labels/{id}`: see §5.

## 9. Minimum UI (full polish is sub-project B)

- **Threshold slider** (`widgets.js`):
  - where: the comparison header on the Benchmark page and the Explorer filter bar;
  - range 50–100%, step 5, default = the question's threshold;
  - 300 ms debounce, then re-fetch with `?threshold=`;
  - remembered with `storage.js` (`threshold`).
- **`report.js`:**
  - a multi card has: a chart of `label_counts` per rater; a rater table (n · coverage · labels/email ·
    uncertainty · confidence); a pair table (exact match · Jaccard · F1 · κ (macro) · JSD · Brier).
  - Score cards show "Mean score (0–100)" instead of "Mean level".
- **Explorer table:**
  - a multi cell lists the applied labels. Its class is `cell-match` for the same set as the reference,
    `cell-partial` (new, amber) for overlap and `cell-mismatch` for disjoint sets. The classification is a pure
    helper with `node --test` coverage.
  - Score cells (reference and runs) show the level and the score, for example `today · 72`.
  - The reference filter matches list membership for multi.
- **Explorer detail:**
  - a multi section: per-label probability bars with a threshold tick; ✓ on reference labels; human labels as a
    checkbox group;
  - score sections: the 0–100 score per run in the header row.
- **Explorer default runs:** without `?runs=`, pre-select the latest completed run per column on exactly the
  selected generations (the same rule as the Benchmark page). This fixes the "Ref only" detail panel.
- All DOM is built with `h()`. There is no `innerHTML`, as before (`tests/test_web_static.py`).

## 10. Testing and verification

- TDD, one test file per module, `pytest.param(..., id=...)` tables, OpenRouter always mocked.
- New `tests/test_metrics_multilabel.py`:
  - table cases: empty vs empty, disjoint, identical, single label, all labels, p exactly at t, n = 0;
  - Hypothesis properties: Jaccard ∈ [0, 1] and symmetric; exact match ≤ mean Jaccard; `binarize` is monotone
    in t; identical inputs give κ = 1 or `None`.
- New rows in existing tables:
  - Jev unfold/fold including missing sub-answers;
  - LLM multi parse (all-zero valid, missing → 0 + note, all missing → note);
  - embeddings ratio-to-max;
  - generator list validation and retry;
  - `count_mismatches` membership;
  - labels PUT list validation;
  - `?threshold=` bounds (0, 1, 1.01, −1, non-number);
  - v1-snapshot skip warnings;
  - base = newest run.
- `tests/factories.py`: the mini question set gains one multi question, so every column path exercises it.
- A browser smoke against a scratch server with a fake OpenRouter (`MockTransport`), covering the slider,
  multi cells and human checkboxes.
- **Real smoke (paid, only after explicit approval):**
  - `jev-bench generate --count 10` on v2, then one run per column on that generation;
  - check that each column returns multi-label answers, the Jev output tokens vs `jev_output_reserve`, and that
    the report renders;
  - expected cost ≈ $0.15–0.25, based on the v1 runs: Jev $0.008, GPT $0.49, Sonnet $0.64, embeddings $0.004
    per 97 emails, plus the generation.

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
