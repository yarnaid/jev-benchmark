# Question Model v2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or
> superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for
> tracking. Each task lives in its own file under `2026-09-25-question-model-v2/`; an implementer needs only
> this index (Global Constraints, Review Focus, Planning revisions) and their task file.

**Goal:**
- Make `category` multi-label: every column keeps returning a distribution summing to 1, and a label is
  applied when `p >= threshold × max(p)` (default 80%, editable in the UI).
- Add a 0–100 score for every ordered question.
- Ship question set `email-triage-v2` with seven new questions.
- Keep v1 runs comparable.

**Architecture:**
- A new question type `multi` in `questions.py`, answered exactly like a `choice`; Jev gets it as a native
  Decisions `choice`.
- The multi reading rule lives once in `metrics.multilabel.relative_labels`, with the UI mirror
  `answers.appliedLimit`.
- Label-set metrics (exact match, Jaccard, micro-F1, macro κ, macro Fleiss' κ) live in `metrics/multilabel.py`.
  The compare-level glue lives in `compare/multi.py`: NamedTuples, so there is no import cycle with
  `pairs.py` / `report.py`.
- The threshold is a property of the question. An override is applied once at the entry points (`compare()`,
  `email_rows()`) with `with_threshold()`.
- `compatible()` becomes shape-based: choice, score and multi with the same option ids are comparable.

**Tech Stack:** Python ≥ 3.14, uv, FastAPI 0.141, pydantic v2, numpy, httpx2; pytest + hypothesis +
factory-boy; ruff, pyright (standard); vanilla ES modules + Bootstrap 5.3.8 + Chart.js 4.5.1; `node --test`.

**Spec:** `docs/superpowers/specs/2026-09-25-question-model-v2-design.md` (revision 2; §11 is out of scope).

**Branch:** `feat/question-model-v2`, stacked on `feat/benchmark-foundation`. Baseline: 607 tests pass in
~2.1 s.

## Global Constraints

- **Language:** all repository content in English (code, docs, UI copy, prompts, commits).
- **Python and dependencies:** `requires-python = ">=3.14"`; dependencies only via `uv add`. This plan adds
  **no** dependencies.
- **HTTP client:** `httpx2`, never `httpx`. No OpenAI SDK.
- **Every Python file:**
  - starts with a module docstring (purpose, plus a list of classes and functions), updated whenever a public
    name is added;
  - has no inline comments;
  - has full type annotations;
  - declares `__all__`.
- **Types:** pydantic models or `NamedTuple` for known shapes; `pathlib` only.
- **Functions:** short (≤ ~10 lines where practical), one abstraction level each.
- **Tests:**
  - one file per module;
  - `pytest.param(..., id="...")` tables;
  - OpenRouter always mocked (`httpx2.MockTransport`, `tests.factories.FakeOpenRouter`);
  - each test < 50 ms; the default suite < 5 s.
- **After every edit:** `uv run ruff check --fix && uv run ruff format && uv run pyright` (0 errors). For JS:
  `node --check src/jev_bench/web/static/js/*.js` and `node --test tests/js/`.
- **Commits:** one commit per task, `feat(scope): …` / `docs: …`, ending with
  `Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>`
- **Answer invariant (unchanged):**
  - every answer from every column is a distribution summing to 1, with `noul` as `{yes, no}`;
  - multi answers are distributions too;
  - a multi label is applied iff `p >= threshold × max(p)`, so the top option always applies, and boundary
    ties are applied;
  - the threshold is in (0, 1], default `0.8`.
- **Hard answers** (generator reference, human labels):
  - an option id (`str`); for multi it is a one-label set;
  - or, for multi, a non-empty list of unique option ids, stored as given and read as a uniform distribution
    over those labels;
  - type alias: `HardAnswer = str | list[str]`.
- **Jev:** a multi question is sent and parsed as a native Decisions `choice`. The LLM schema, the LLM prompts
  and the embeddings classifier are **unchanged**.
- **Score:** `score_0_100 = expected_level / (k − 1) × 100`, with a single implementation in
  `metrics/distributions.py`.
- **What models see:** exactly `Email.to_state()`. **Generator answers never feed the benchmark.**
- **Frontend:**
  - no `innerHTML` / `outerHTML` / `insertAdjacentHTML` / `document.write`; all DOM through `h()`;
  - no inline scripts;
  - `localStorage` only via `js/storage.js`.
- **Paid calls:** none during execution. The real smoke (10 emails, about $0.15–0.25) runs **only** after the
  user's explicit approval (Task 10).

## Review Focus

These are the five inputs the spec implies but its own examples never exercise, most likely first. Each one is
pinned by a test in the named task.

1. **Mixed v1/v2 runs in one comparison.**
   - v1 `choice` runs are now *compatible* with v2 `multi`, so they are read with the multi rule.
   - The base must be the **newest** run's snapshot whatever the URL order; otherwise `category` could be read
     as single-label.
   - Runs with different option ids are still skipped with a warning.
   - Pinned in **Task 7** (both URL orders, reused vs skipped).
2. **An old string reference for a now-multi question** becomes a one-label set:
   - on the backend, under a compatible snapshot (**Task 4**);
   - in the Explorer, which matches `"spam"` against `["spam"]` (**Task 9**, `setMatch`).
3. **A probability exactly at `threshold × max(p)`**, including a slider value sent as a query string, **is**
   applied.
   - Pinned in **Task 2** (`0.5 / 0.375 / 0.125` at 0.75) and **Task 7**: the fake chat answer sits on that
     exact boundary with the mini threshold 0.75.
4. **Flat or near-tied distributions apply many labels.**
   - Every metric must stay finite or `None` (κ/Fleiss are undefined for constant label columns), and the
     report must be strict JSON.
   - Pinned in **Task 2**, **Task 5** and **Task 6** (the flat-row tests).
5. **Human multi-label edits.**
   - `[]` clears a label; a single id is a valid one-label answer.
   - Duplicate or unknown labels, or a list for a single-choice question → 400.
   - Pinned in **Task 7** (labels route).

## Planning revisions (deviations from the spec, decided while planning)

1. **Spec §6, per-row set metrics:** `exact_match_rows` / `jaccard_rows` are per row, because bootstrap CIs
   need per-row values. n = 0 never reaches them: `pair_stats` returns `None` without shared emails.
2. **Spec §9, slider placement:** the threshold slider lives in page chrome (`#threshold-control`), not
   inside the re-rendered report or filter DOM, so dragging never loses focus.
3. **Spec §9, Explorer run selection:** changing the selected generations re-defaults the run selection to the
   latest completed run per column.
4. **Spec §9, score display:** Explorer detail scores are a footer row, "score (0–100)".
5. **`js/api.js` `query()`** also skips `null`, so an unset threshold is never sent as `threshold=null`
   (that would be a 422).
6. **`rows.py` reuses `relative_labels`** instead of re-implementing the rule. The detail panel's JS mirror
   is the pure `answers.appliedLimit`, under `node --test`.

## How this plan was validated

- The revised design was implemented task by task in a throwaway worktree (ruff, pyright and the full suite
  after each task), with a Playwright smoke over the fake OpenRouter.
- Every task's code blocks were then **generated from that worktree**:
  - unified diffs for modified files;
  - full contents for new files;
  - for files touched by several tasks, per-task intermediate versions.
- Finally the 10 task diffs were **replayed in order on a clean checkout**:
  - every diff applied;
  - lint, pyright and the suite were green after every task: 607 → 640 → 663 → 670 → 685 → 693 → 706 → 723;
  - coverage 99%, 72 node tests;
  - the result is byte-identical to the validated worktree.
- The validation caught these, now handled:
  - a pyright break (Task 1);
  - a runtime `TypeError` when references became lists (Task 4);
  - three wrong hand-computed expected values;
  - Hypothesis tests over budget;
  - a formatting difference in a reconstructed intermediate file.
- Reviewers: "validated while planning" and "verbatim from the plan" are **not** defenses. Review each diff on
  its merits, and mutation-probe the invariants:
  - `>=` versus `>` at the boundary;
  - the `threshold × max` rule;
  - the uniform hard distribution;
  - the newest-run base;
  - `[]` clears.

## File map (new ★, modified ✎)

```
src/jev_bench/questions.py                 ✎ MultiQuestion, HardAnswer, hard_distribution, with_threshold, shape-based compatible
src/jev_bench/metrics/distributions.py     ✎ score_0_100
src/jev_bench/metrics/multilabel.py        ★ relative_labels, exact/jaccard rows, micro-F1, macro κ, macro Fleiss
src/jev_bench/metrics/agreement.py         ✎ brier_to_target
src/jev_bench/classifiers/jev.py           ✎ multi → native choice
src/jev_bench/emails.py                    ✎ reference_answers: dict[str, HardAnswer]
src/jev_bench/generation/prompt.py         ✎ array schema, hard-answer validation, membership mismatch
src/jev_bench/compare/multi.py             ★ MultiPairValues, MultiRaterValues, multi_fleiss
src/jev_bench/compare/{raters,pairs,report,rows}.py   ✎ hard answers, multi branches, threshold, scores
src/jev_bench/store/labels.py              ✎ list labels
src/jev_bench/web/deps.py                  ✎ ThresholdQuery
src/jev_bench/web/routes/{compare,emails,labels}.py   ✎ threshold, newest base, list labels
src/jev_bench/web/static/js/{answers,selection,email-detail}.js   ★
src/jev_bench/web/static/js/{api,widgets,report,charts,benchmark,explorer,distribution}.js  ✎
src/jev_bench/web/static/{index,explorer}.html, css/app.css   ✎
config/questions.toml, config/generation.toml   ✎ (benchmark.toml unchanged)
tests/conftest.py, tests/factories.py, tests/compare_data.py  ✎
tests/test_metrics_multilabel.py, tests/test_compare_multi.py, tests/js/{answers,selection}.test.mjs   ★
CLAUDE.md                                  ✎
```

## Tasks (execute in order)

| # | File | Deliverable | Tests after |
|---|---|---|---|
| 1 | `task-01-question-model.md` | `MultiQuestion`, hard answers, `with_threshold`, shape-based `compatible`, `score_0_100`, `multi_questions` fixture | 640 |
| 2 | `task-02-multilabel-metrics.md` | `relative_labels` and the set metrics, `brier_to_target` | 663 |
| 3 | `task-03-columns-multi-as-choice.md` | Jev sends multi as `choice`; characterization tests for LLM and embeddings | 670 |
| 4 | `task-04-label-list-references.md` | label-list references: email model, generator, raters, row types | 685 |
| 5 | `task-05-compare-pairs.md` | `compare/multi.py` pair values, the `pairs.py` multi branch | 693 |
| 6 | `task-06-compare-report-rows.md` | label counts, macro Fleiss, threshold override, 0–100 scores, rows | 706 |
| 7 | `task-07-api-wiring.md` | mini config wiring, `?threshold=`, newest-run base, list labels | 723 |
| 8 | `task-08-ui-benchmark.md` | threshold slider, multi/score report cards | 723 |
| 9 | `task-09-ui-explorer.md` | Explorer cells, detail panel, checkboxes, default runs | 723 |
| 10 | `task-10-config-docs-verify.md` | `questions.toml` v2, CLAUDE.md, final gates, browser smoke, paid-smoke gate | 723 |
