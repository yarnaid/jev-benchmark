# Question Model v2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or
> superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for
> tracking. Each task lives in its own file under `2026-09-25-question-model-v2/`; an implementer needs only
> this index (Global Constraints, Review Focus, Planning revisions) and their task file.

**Goal:**
- Replace the single-label `category` with a multi-label question, where every column returns an independent
  probability per label and a label is applied at an editable threshold (default 80%).
- Add a 0–100 score for every ordered question.
- Ship question set `email-triage-v2` with seven new questions.

**Architecture:**
- A new first-class question type `multi` in `questions.py`. Each consumer gets one explicit branch:
  - Jev unfolds a multi question into one `noul` per option and folds the answers back;
  - LLMs return independent numbers;
  - embeddings use softmax ÷ max;
  - the generator returns a label list.
- Pure numpy metrics for multi-label answers live in `metrics/multilabel.py`. The compare-level glue lives in
  `compare/multi.py`: NamedTuples, so there is no import cycle with `pairs.py` / `report.py`.
- The threshold is a property of the question. An override is applied once at the entry points (`compare()`,
  `email_rows()`) with `with_threshold()`, so no downstream signature changes.

**Tech Stack:** Python ≥ 3.14, uv, FastAPI 0.141, pydantic v2, numpy, httpx2; pytest + hypothesis +
factory-boy; ruff, pyright (standard); vanilla ES modules + Bootstrap 5.3.8 + Chart.js 4.5.1; `node --test`.

**Spec:** `docs/superpowers/specs/2026-09-25-question-model-v2-design.md` (read it first; §11 is out of scope).

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
- **Types:** pydantic models or `NamedTuple` for known shapes (no `dataclasses`, no raw dicts for known
  shapes); `pathlib` only.
- **Functions:** short (≤ ~10 lines where practical), one abstraction level each.
- **Tests:**
  - one file per module;
  - `pytest.param(..., id="...")` tables (a new case is a new row);
  - OpenRouter always mocked (`httpx2.MockTransport`, `tests.factories.FakeOpenRouter`);
  - each test < 50 ms; the default suite < 5 s.
- **After every edit:** `uv run ruff check --fix && uv run ruff format && uv run pyright` (0 errors). For JS:
  `node --check src/jev_bench/web/static/js/*.js` and `node --test tests/js/`.
- **Commits:** one commit per task, `feat(scope): …` / `fix(scope): …` / `docs: …`, ending with
  `Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>`
- **Answer semantics:**
  - for `choice` / `score` / `noul`, an answer is a distribution summing to 1;
  - for `multi`, an answer is a vector of **independent** probabilities in [0, 1], one per option, **never
    renormalized**; an all-zero vector is a valid "no label applies";
  - a label is applied iff `p >= threshold`;
  - the threshold is in (0, 1] and defaults to `0.8`.
- **Hard answers** (generator reference, human labels):
  - an option id (`str`) for `choice` / `score` / `noul`;
  - a non-empty list of unique option ids for `multi`.
  - Type alias: `HardAnswer = str | list[str]`.
- **Jev:** only `choice` / `score` / `noul` exist. A multi question is sent as one `noul` per option, keyed
  `f"{question_id}__{option_id}"`, in the same Decisions call.
- **Score:** `score_0_100 = expected_level / (k − 1) × 100`, with a single implementation in
  `metrics/distributions.py`.
- **What models see:** exactly `Email.to_state()`. **Generator answers never feed the benchmark**: nothing is
  tuned on references.
- **Frontend:**
  - no `innerHTML` / `outerHTML` / `insertAdjacentHTML` / `document.write`; all DOM through `h()` from
    `js/dom.js`;
  - no inline scripts;
  - every `localStorage` access goes through `js/storage.js`.
- **Paid calls:** none during execution. The real smoke (10 emails, about $0.15–0.25) runs **only** after the
  user's explicit approval (Task 12).

## Review Focus

These are the five inputs the spec implies but its own examples never exercise, most likely first. Each one is
pinned by a test in the named task.

1. **Mixed v1/v2 runs in one comparison.**
   - A v1 run (where `category` is a choice) compared together with v2 runs must be *skipped* on the changed
     questions, with a warning. A choice-shaped distribution must never reach the multi metrics.
   - The report must be the same whatever the URL order of `runs=`.
   - Pinned in **Task 9**: base = the newest run, checked in both orders.
2. **Old emails whose reference is a string for a now-multi question.**
   - The reference rater skips them (**Task 6**).
   - The Explorer treats reference `"spam"` vs answer `["spam"]` as a match and never throws (**Task 11**,
     `setMatch`).
3. **A probability exactly at the threshold**, including a slider value sent as the query string `0.7`: the
   label **is** applied.
   - Pinned in **Task 2** (a `binarize` row) and **Task 9**: the fake chat answer has `meeting = 0.7` and the
     mini threshold is `0.7`.
4. **An email where no label reaches the threshold.**
   - `top` is `[]`, coverage is < 1, and every metric is finite or `None` (strict JSON).
   - The UI shows "none".
   - Pinned in **Task 8** (degenerate multi report, empty top) and **Task 11** (`answerText([])`).
5. **Human multi-label edits.**
   - `[]` clears a label.
   - Duplicate or unknown labels, a bare string for a multi question, or a list for a single-choice question →
     400.
   - Pinned in **Task 9** (labels route table).

## Planning revisions (deviations from the spec, decided while planning)

1. **§6, `exact_match`:** it becomes the per-row `exact_match_rows`, because bootstrap CIs need per-row values.
   n = 0 never reaches it: `pair_stats` already returns `None` without shared emails.
2. **§4.1 / §4.2, note texts:** Jev and the LLM parser share `independent_probabilities()` (distributions)
   and `missing_labels_note()` (classifiers/base). The single note text is
   `"<n> of <k> labels missing or unusable, set to 0"`. The all-missing notes stay each parser's existing text.
3. **§4.1, key collisions:** `questions_payload` raises `ValueError` if an unfolded key collides with another
   question id.
4. **§9, slider placement:** the threshold slider lives in page chrome (`#threshold-control`), not inside the
   re-rendered report or filter DOM, so dragging never loses focus.
5. **§9, Explorer run selection:** changing the selected generations re-defaults the run selection to the
   latest completed run per column. Runs are generation-specific.
6. **§9, score display:** scores in the Explorer detail are a footer row, "score (0–100)", in each score
   question's table.
7. **`js/api.js` `query()`** also skips `null`, so an unset threshold is never sent as `threshold=null`
   (that would be a 422).

## How this plan was validated

- Every Python task's code was applied, task by task, in a throwaway detached worktree. After each task,
  `ruff`, `pyright` and the full default suite ran there: 607 → 647 → … → 738 tests, coverage 99%.
- The UI tasks were checked with `node --check`, `node --test` (68 tests) and a Playwright smoke against the
  real app over the fake OpenRouter, with no console errors.
- This validation caught these problems, which the tasks now handle:
  - pyright needed `QuestionReport.type` widened in Task 1;
  - `raters._reference_hard` raised `TypeError: unhashable type: 'list'` once references could be lists
    (Task 6 owns it);
  - two hand-computed expected values were wrong (Task 7);
  - Hypothesis tests ran over budget at the default 100 examples (Task 2).
- Reviewers: "it was validated while planning" and "verbatim from the plan" are **not** defenses. Review the
  diff on its merits, and mutation-probe the invariants (threshold `>=`, no renormalization, newest-run base,
  `[]` clears).

## File map (new ★, modified ✎)

```
src/jev_bench/questions.py                 ✎ MultiQuestion, HardAnswer, multi_hot, hard_distribution, with_threshold
src/jev_bench/metrics/distributions.py     ✎ score_0_100, independent_probabilities
src/jev_bench/metrics/multilabel.py        ★ binarize, exact/jaccard rows, micro-F1, macro κ, binary entropy/JSD, Brier, macro Fleiss
src/jev_bench/classifiers/base.py          ✎ missing_labels_note
src/jev_bench/classifiers/jev.py           ✎ unfold/fold multi as nouls
src/jev_bench/classifiers/llm_schema.py    ✎ multi description
src/jev_bench/classifiers/llm_parse.py     ✎ independent parse for multi
src/jev_bench/classifiers/embeddings.py    ✎ softmax ÷ max for multi
src/jev_bench/emails.py                    ✎ reference_answers: dict[str, HardAnswer]
src/jev_bench/generation/prompt.py         ✎ array schema, list validation, membership mismatch
src/jev_bench/compare/multi.py             ★ MultiPairValues, MultiRaterValues, multi_fleiss
src/jev_bench/compare/{raters,pairs,report,rows}.py   ✎ multi branches, threshold, scores
src/jev_bench/store/labels.py              ✎ list labels
src/jev_bench/web/deps.py                  ✎ ThresholdQuery
src/jev_bench/web/routes/{compare,emails,labels}.py   ✎ threshold, newest base, list labels
src/jev_bench/web/static/js/{answers,selection}.js    ★ pure helpers
src/jev_bench/web/static/js/{api,widgets,report,charts,benchmark,explorer,distribution}.js  ✎
src/jev_bench/web/static/{index,explorer}.html, css/app.css   ✎
config/{questions,benchmark,generation}.toml   ✎ v2 question set, prompt bullets
tests/conftest.py, tests/factories.py, tests/compare_data.py  ✎ multi fixtures
tests/test_metrics_multilabel.py, tests/js/{answers,selection}.test.mjs   ★
CLAUDE.md                                  ✎
```

## Tasks (execute in order)

| # | File | Deliverable |
|---|---|---|
| 1 | `task-01-question-model.md` | `MultiQuestion`, hard answers, `with_threshold`, `score_0_100`, `independent_probabilities`, `multi_questions` fixture |
| 2 | `task-02-multilabel-metrics.md` | `metrics/multilabel.py` |
| 3 | `task-03-jev-multi.md` | Jev unfold/fold, `missing_labels_note` |
| 4 | `task-04-llm-multi.md` | LLM schema + parse for multi, prompt bullets in `benchmark.toml` |
| 5 | `task-05-embeddings-multi.md` | embeddings softmax ÷ max |
| 6 | `task-06-generation-multi.md` | label-list references: email model, generator schema/parse/mismatch/prompt, multi-hot raters |
| 7 | `task-07-compare-pairs.md` | `compare/multi.py` pair values, `pairs.py` multi branch |
| 8 | `task-08-compare-report-rows.md` | multi rater stats, macro Fleiss, threshold override, 0–100 scores, rows |
| 9 | `task-09-api-wiring.md` | mini config multi wiring, `?threshold=`, newest base, list labels |
| 10 | `task-10-ui-benchmark.md` | threshold slider, multi/score report cards |
| 11 | `task-11-ui-explorer.md` | Explorer multi cells, detail, human checkboxes, default runs |
| 12 | `task-12-config-docs-verify.md` | `questions.toml` v2, CLAUDE.md, final gates, browser smoke, paid-smoke gate |
