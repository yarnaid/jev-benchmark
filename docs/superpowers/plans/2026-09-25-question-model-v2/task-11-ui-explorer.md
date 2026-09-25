### Task 11: Explorer UI: multi-label cells, detail panel, human checkboxes, default runs

**Files:**
- Create: `src/jev_bench/web/static/js/answers.js`: pure answer comparison and formatting.
- Create: `src/jev_bench/web/static/js/selection.js`: pure run selection.
- Create: `src/jev_bench/web/static/js/email-detail.js`: the detail panel, moved out of `explorer.js`.
- Modify: `src/jev_bench/web/static/js/explorer.js`
- Modify: `src/jev_bench/web/static/js/distribution.js` (threshold tick)
- Modify: `src/jev_bench/web/static/explorer.html` (`#threshold-control`)
- Modify: `src/jev_bench/web/static/css/app.css` (`.prob-track` positioning, `.prob-tick`, `.cell-partial`)
- Modify: `tests/test_web_static.py` (`PAGE_MODULES`)
- Test: `tests/js/answers.test.mjs`, `tests/js/selection.test.mjs` (new, `node --test`)

**Why the split:** the detail panel changes would take `explorer.js` from 282 to about 340 lines, past the
300-line limit (`rules/modularity.md`). Moving the panel (header, sections, run cells, human controls) into
`email-detail.js` leaves `explorer.js` at 257 lines, with `email-detail.js` at 102. Both sizes were measured
while planning.

**Interfaces:**
- Consumes:
  - Task 9: `EmailRow.top` (a label list for multi), `.scores` and `.reference_scores`;
    `EmailDetail.human` lists; `PUT /api/labels` with lists.
  - Task 10: `api.emails(..., threshold)`, `thresholdSlider` and the `threshold` pref.
- Produces:
  - `answers.js`:
    - `asLabels(answer) -> string[]`: an array passes through; a scalar becomes `[x]`; `null`/`undefined`
      become `[]`.
    - `setMatch(reference, answer) -> "match" | "partial" | "mismatch" | ""`: `""` when either side is
      missing; a string against an array is compared as a one-label set.
    - `answerText(answer) -> string`: `"—"` if missing, `"none"` for `[]`, `", "`-joined for arrays.
    - `withScore(text, score) -> string`: `` `${text} · ${Math.round(score)}` `` for a finite number, else
      `text`.
  - `selection.js`: `sameSet(values, set)` and `latestCompletedPerColumn(runs, generationIds) -> runId[]`,
    the newest completed run per column whose `generation_ids` equal the given set. It sorts by id descending
    itself.
  - `email-detail.js`: `emailDetail(detail, { row = null, threshold = null, onSave }) -> Node[]`.
    `threshold` is the slider override; `null` means each question's own default.

**Behavior** (spec §9; planning revisions 5 and 6):
- Table cells:
  - multi cells list the applied labels (`none` when empty);
  - the cell class comes from `setMatch` → `cell-match` / `cell-partial` / `cell-mismatch`;
  - score cells show `level · score`.
- The reference filter matches list membership.
- Without `?runs=` the latest completed run per column is pre-selected, and changing the generations
  re-defaults it.
- The threshold slider (rendered once in `#threshold-control`) calls `refreshRows()`. That re-fetches only
  the rows, so the displayed question and the filters are kept.
- Detail panel:
  - a multi card shows a `≥ N%` badge, a tick at the threshold on every bar, **bold** labels at or above it,
    ✓ on the reference labels, and human checkboxes;
  - a score card has a `score (0–100)` footer row (reference, then each run), taken from the list row.
- The planning smoke confirmed all of this, and a saved checkbox pair was persisted as
  `"topics": ["meeting", "travel"]`.

- [ ] **Step 1: Write the failing JS tests**

`tests/js/answers.test.mjs`:

```js
// Run with `node --test tests/js/`. Exercises src/jev_bench/web/static/js/answers.js (pure, no DOM).
import assert from "node:assert/strict";
import test from "node:test";

const { asLabels, setMatch, answerText, withScore } = await import("../../src/jev_bench/web/static/js/answers.js");

const MATCHES = [
  ["same option", "spam", "spam", "match"],
  ["different option", "spam", "work", "mismatch"],
  ["same label set in any order", ["a", "b"], ["b", "a"], "match"],
  ["overlapping label sets", ["a", "b"], ["b", "c"], "partial"],
  ["answer is a subset", ["a", "b"], ["a"], "partial"],
  ["disjoint label sets", ["a"], ["b"], "mismatch"],
  ["both empty", [], [], "match"],
  ["old string reference vs label list", "spam", ["spam"], "match"],
  ["old string reference vs wider list", "spam", ["spam", "scam"], "partial"],
  ["no answer", "spam", undefined, ""],
  ["no reference", undefined, ["spam"], ""],
  ["null reference", null, "spam", ""],
];

for (const [name, reference, answer, expected] of MATCHES) {
  test(`setMatch: ${name}`, () => assert.equal(setMatch(reference, answer), expected));
}

const TEXTS = [
  ["missing", undefined, "—"],
  ["null", null, "—"],
  ["empty label list", [], "none"],
  ["label list", ["billing", "meeting"], "billing, meeting"],
  ["option id", "today", "today"],
];

for (const [name, answer, expected] of TEXTS) {
  test(`answerText: ${name}`, () => assert.equal(answerText(answer), expected));
}

test("asLabels wraps a single answer and passes arrays through", () => {
  assert.deepEqual(asLabels("a"), ["a"]);
  assert.deepEqual(asLabels(["a", "b"]), ["a", "b"]);
  assert.deepEqual(asLabels(undefined), []);
});

const SCORES = [
  ["rounded score", 72.4, "today · 72"],
  ["zero", 0, "today · 0"],
  ["missing", undefined, "today"],
  ["not finite", NaN, "today"],
];

for (const [name, score, expected] of SCORES) {
  test(`withScore: ${name}`, () => assert.equal(withScore("today", score), expected));
}
```

`tests/js/selection.test.mjs`:

```js
// Run with `node --test tests/js/`. Exercises src/jev_bench/web/static/js/selection.js (pure, no DOM).
import assert from "node:assert/strict";
import test from "node:test";

const { sameSet, latestCompletedPerColumn } = await import("../../src/jev_bench/web/static/js/selection.js");

const run = (id, column, status = "completed", generation_ids = ["g1"]) => ({ id, column, status, generation_ids });

test("sameSet compares membership and size", () => {
  assert.equal(sameSet(["a", "b"], new Set(["b", "a"])), true);
  assert.equal(sameSet(["a"], new Set(["a", "b"])), false);
});

test("latestCompletedPerColumn picks the newest completed run of each column on exactly those generations", () => {
  const runs = [
    run("20260925-080000-jev-a", "jev"),
    run("20260925-090000-jev-b", "jev"),
    run("20260925-100000-jev-c", "jev", "running"),
    run("20260925-090000-openai-a", "openai", "completed", ["g1", "g2"]),
    run("20260925-080000-openai-b", "openai"),
    run("20260925-110000-anthropic-a", "anthropic", "failed"),
  ];
  assert.deepEqual(latestCompletedPerColumn(runs, ["g1"]), ["20260925-090000-jev-b", "20260925-080000-openai-b"]);
  assert.deepEqual(latestCompletedPerColumn(runs, ["g1", "g2"]), ["20260925-090000-openai-a"]);
  assert.deepEqual(latestCompletedPerColumn([], ["g1"]), []);
});
```

In `tests/test_web_static.py`, add `"answers.js"`, `"selection.js"` and `"email-detail.js"` to the end of
`PAGE_MODULES`.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `node --test tests/js/ ; uv run pytest tests/test_web_static.py -q`
Expected: the node tests fail with `ERR_MODULE_NOT_FOUND`, and `test_page_modules_exist` fails.

- [ ] **Step 3: Create `js/answers.js`**

```js
/**
 * Pure helpers for comparing and displaying answers. A multi-label answer is an array of labels
 * (possibly empty); any other answer is one option id. A string compared with an array is treated as
 * a one-label array, so an old single-label reference still matches a multi-label answer.
 * Exports: asLabels, setMatch, answerText, withScore.
 */

export const asLabels = (answer) => (Array.isArray(answer) ? answer : answer === undefined || answer === null ? [] : [answer]);

export function setMatch(reference, answer) {
  if (reference === undefined || reference === null || answer === undefined || answer === null) return "";
  const expected = new Set(asLabels(reference));
  const actual = asLabels(answer);
  const shared = actual.filter((label) => expected.has(label)).length;
  if (shared === expected.size && shared === actual.length) return "match";
  return shared > 0 ? "partial" : "mismatch";
}

export function answerText(answer) {
  if (answer === undefined || answer === null) return "—";
  if (Array.isArray(answer)) return answer.length ? answer.join(", ") : "none";
  return String(answer);
}

export function withScore(text, score) {
  return typeof score === "number" && Number.isFinite(score) ? `${text} · ${Math.round(score)}` : text;
}
```

- [ ] **Step 4: Create `js/selection.js`**

```js
/**
 * Pure run-selection helpers: set equality of generation ids and the default run per column.
 * Exports: sameSet, latestCompletedPerColumn.
 */

export const sameSet = (values, set) => values.length === set.size && values.every((value) => set.has(value));

export function latestCompletedPerColumn(runs, generationIds) {
  const wanted = new Set(generationIds);
  const chosen = new Map();
  const newestFirst = [...runs].sort((a, b) => b.id.localeCompare(a.id));
  for (const run of newestFirst) {
    if (run.status === "completed" && sameSet(run.generation_ids, wanted) && !chosen.has(run.column)) chosen.set(run.column, run.id);
  }
  return [...chosen.values()];
}
```

Run: `node --test tests/js/`. Expected: all pass (68 tests while planning).

- [ ] **Step 5: Create `js/email-detail.js`**

`emailHeader` and `humanSelect` move here **unchanged** from `explorer.js`. The rest is new or adapted.

```js
/**
 * Explorer detail panel content: the email header, the (hostile-by-design) body as plain text, and one
 * card per question with each run's probability per option, the generator reference, a 0-100 score
 * row for score questions, and the human-label control (a select, or checkboxes for a multi-label
 * question, whose card also shows the label threshold and marks labels at or above it).
 * Exports: emailDetail.
 */
import { asLabels } from "./answers.js";
import { probabilityCell } from "./distribution.js";
import { h, icon } from "./dom.js";
import { fixed, when } from "./format.js";

export function emailDetail(detail, { row = null, threshold = null, onSave }) {
  return [emailHeader(detail.email), h("div", { class: "email-body mb-3" }, detail.email.body), labellingForm(detail, row, threshold, onSave)];
}

function emailHeader(email) {
  const party = (value) => (value.name ? `${value.name} <${value.address}>` : value.address);
  const traits = Object.entries(email.traits).map(([name, value]) => `${name}=${value}`).join(", ");
  const rows = [
    ["From", party(email.sender)],
    ["To", email.to.map(party).join(", ")],
    ["Cc", email.cc.map(party).join(", ") || "—"],
    ["Sent", when(email.sent_at)],
    ["Generator", email.generator_model],
    ["Traits", traits || "—"],
    ["Id", email.id],
  ];
  return h("dl", { class: "row small mb-2" }, rows.flatMap(([label, value]) => [h("dt", { class: "col-3" }, label), h("dd", { class: "col-9 text-break mb-1" }, value)]));
}

function labellingForm(detail, row, threshold, onSave) {
  const choices = { ...detail.human };
  const context = { detail, row, threshold, choices, runIds: Object.keys(detail.predictions) };
  const sections = detail.questions.questions.map((question) => questionSection(question, context));
  const save = h("button", { class: "btn btn-primary", type: "button", onclick: () => onSave(detail.email.id, choices, detail.human) }, icon("save"), " Save labels");
  return h("div", {}, sections, h("div", { class: "d-flex justify-content-end mt-2" }, save));
}

function questionSection(question, { detail, row, threshold, choices, runIds }) {
  const cut = question.type === "multi" ? (threshold ?? question.threshold) : null;
  const reference = asLabels(detail.email.reference_answers[question.id]);
  const head = ["Option", "Ref", ...runIds.map((id) => detail.run_labels[id] ?? id)];
  const rows = Object.entries(question.options).map(([option, text]) =>
    h("tr", {}, h("td", { class: "small", title: text }, option), h("td", { class: "text-center" }, reference.includes(option) ? icon("check-lg") : ""), runIds.map((id) => runCell(detail.predictions[id], question.id, option, cut))),
  );
  const foot = question.type === "score" ? scoreRow(question.id, row, runIds) : null;
  const table = h("table", { class: "table table-sm mb-2" }, h("thead", {}, h("tr", {}, head.map((cell) => h("th", { class: "small" }, cell)))), h("tbody", {}, rows), foot);
  const badge = cut === null ? null : h("span", { class: "badge text-bg-info", title: "A label counts as applied at this probability or higher" }, `≥ ${Math.round(cut * 100)}%`);
  const human = h("div", { class: "d-flex align-items-center gap-2" }, h("span", { class: "small text-nowrap" }, icon("person"), " Human"), humanControl(question, detail.human[question.id], choices));
  return h(
    "div",
    { class: "card mb-2" },
    h("div", { class: "card-header py-1 d-flex align-items-center gap-2" }, h("code", {}, question.id), badge, h("span", { class: "small text-body-secondary text-truncate" }, question.instructions)),
    h("div", { class: "card-body py-2" }, h("div", { class: "table-responsive" }, table), human),
  );
}

function scoreRow(questionId, row, runIds) {
  if (!row) return null;
  const cell = (score) => h("td", { class: "small font-monospace" }, fixed(score, 0));
  return h("tfoot", {}, h("tr", {}, h("td", { class: "small fw-semibold" }, "score (0–100)"), cell(row.reference_scores?.[questionId]), runIds.map((id) => cell(row.scores?.[id]?.[questionId]))));
}

const topOption = (distribution) => Object.entries(distribution).reduce((best, entry) => (entry[1] > best[1] ? entry : best))[0];

function runCell(prediction, questionId, option, cut) {
  if (!prediction) return h("td", { class: "small text-body-secondary" }, "—");
  if (prediction.error) return h("td", { class: "small text-danger", title: prediction.error }, "error");
  const distribution = prediction.answers?.[questionId];
  if (!distribution) return h("td", { class: "small text-body-secondary" }, "—");
  const similarity = prediction.similarities?.[questionId]?.[option] ?? null;
  const highlight = cut === null ? topOption(distribution) === option : (distribution[option] ?? 0) >= cut;
  return probabilityCell(distribution[option], { similarity, highlight, threshold: cut });
}

function humanControl(question, current, choices) {
  return question.type === "multi" ? humanLabels(question, current, choices) : humanSelect(question, current, choices);
}

function humanSelect(question, current, choices) {
  const onchange = (event) => {
    choices[question.id] = event.target.value || null;
  };
  const options = Object.entries(question.options).map(([option, text]) => h("option", { value: option, selected: current === option, title: text }, option));
  return h("select", { class: "form-select form-select-sm", "aria-label": `Human label for ${question.id}`, onchange }, h("option", { value: "" }, "— not labelled —"), options);
}

function humanLabels(question, current, choices) {
  const chosen = new Set(asLabels(current));
  const toggle = (option, checked) => {
    if (checked) chosen.add(option);
    else chosen.delete(option);
    choices[question.id] = [...chosen];
  };
  const boxes = Object.entries(question.options).map(([option, text], index) => {
    const id = `human-${question.id}-${index}`;
    const input = h("input", { class: "form-check-input", type: "checkbox", id, checked: chosen.has(option), onchange: (event) => toggle(option, event.target.checked) });
    return h("div", { class: "form-check form-check-inline mb-0" }, input, h("label", { class: "form-check-label small", for: id, title: text }, option));
  });
  return h("div", { class: "d-flex flex-wrap", role: "group", "aria-label": `Human labels for ${question.id}` }, boxes);
}
```

Unchecking every box stores `[]`. The server converts it to "cleared" (Task 9), so `saveLabels` in
`explorer.js` needs no change.

- [ ] **Step 6: Modify `js/explorer.js`**

Header comment:

```js
 * Explorer page: browse the emails of selected generations, compare each run's top answer (the applied
 * labels of a multi-label question; level and 0-100 score of a score question) with the generator
 * reference (match / partial / mismatch), sort by disagreement, and open an email to see every
 * distribution and edit human labels. Without ?runs= the latest completed run per column on the
 * selected generations is pre-selected; the label-threshold slider re-fetches the rows.
```

The imports become:

```js
import { answerText, asLabels, setMatch, withScore } from "./answers.js";
import { api } from "./api.js";
import { clear, h, icon } from "./dom.js";
import { emailDetail } from "./email-detail.js";
import { fixed, shortModel, when } from "./format.js";
import { initLayout, toastError, toastSuccess } from "./layout.js";
import { latestCompletedPerColumn } from "./selection.js";
import { readPref, writePref } from "./storage.js";
import { checklist, emptyState, thresholdSlider } from "./widgets.js";
```

After `EMPTY_FILTERS`, add:

```js
const MATCH_CLASS = { match: "cell-match", partial: "cell-partial", mismatch: "cell-mismatch", "": "" };
```

- Append `threshold: readPref("threshold", null),` and `sliderShown: false,` to `state`.
- The body of `matchesPair` becomes

  ```js
    const [key, value] = pair.split("=");
    return asLabels(values[key]).includes(value);
  ```

- In `main`, replace `state.selectedRuns = listParam(params, "runs");` with:

  ```js
    const requestedRuns = listParam(params, "runs");
    state.selectedRuns = requestedRuns.length ? requestedRuns : latestCompletedPerColumn(state.runs, state.selectedGenerations);
  ```

- In `onGenerations`, add `state.selectedRuns = latestCompletedPerColumn(state.runs, values);` right after
  `state.selectedGenerations = values;`.
- In `loadRows`:
  - pass the threshold: `const list = await api.emails(state.selectedGenerations, activeRuns(), state.threshold);`;
  - call `renderThreshold();` between `renderFilters();` and `renderTable();`.
- Add after `loadRows`:

  ```js
  async function refreshRows() {
    state.rows = (await api.emails(state.selectedGenerations, activeRuns(), state.threshold)).rows;
    renderTable();
  }

  function renderThreshold() {
    const target = document.getElementById("threshold-control");
    const multi = state.questions.find((question) => question.type === "multi");
    if (!multi) {
      clear(target);
      state.sliderShown = false;
      return;
    }
    if (state.sliderShown) return;
    clear(target, thresholdSlider({ value: state.threshold ?? multi.threshold, onChange: onThreshold }));
    state.sliderShown = true;
  }

  function onThreshold(value) {
    state.threshold = value;
    writePref("threshold", value);
    refreshRows().catch(toastError);
  }
  ```

- In `emailRow`, the run cells and the reference cell become:

  ```js
    const cells = runs.map((id) => {
      const answer = row.top[id]?.[state.question];
      const text = withScore(answerText(answer), row.scores[id]?.[state.question]);
      return h("td", { class: `small ${MATCH_CLASS[setMatch(reference, answer)]}` }, text);
    });
  ```

  ```js
      h("td", { class: "small fw-semibold" }, withScore(answerText(reference), row.reference_scores[state.question]), labelled),
  ```

- In `openEmail`, the `try` body becomes:

  ```js
      const detail = await api.email(emailId, runColumns());
      const row = state.rows.find((item) => item.id === emailId) ?? null;
      clear(document.getElementById("email-panel-title"), detail.email.subject);
      clear(body, emailDetail(detail, { row, threshold: state.threshold, onSave: saveLabels }));
  ```

- **Delete** `emailHeader`, `labellingForm`, `humanSelect`, `questionSection` and `runCell` from
  `explorer.js`; they now live in `email-detail.js`. Remove the now-unused `probabilityCell` import.
  `saveLabels` stays unchanged.

- [ ] **Step 7: `js/distribution.js`**

The header comment becomes:

```js
 * Probability visualisation for the Explorer detail panel: a table cell with a bar and the value;
 * the tooltip adds the cosine similarity for embedding runs, and a multi-label threshold is drawn as a
 * tick on the bar.
```

The signature becomes
`export function probabilityCell(probability, { similarity = null, highlight = false, threshold = null } = {}) {`,
and the `bar` line is replaced by:

```js
  const tick = threshold === null ? null : h("div", { class: "prob-tick", style: `left: ${Math.round(threshold * 100)}%`, "aria-hidden": "true" });
  const bar = h("div", { class: "prob-track flex-grow-1" }, h("div", { class: "prob-bar", style: `width: ${Math.round(value * 100)}%` }), tick);
```

- [ ] **Step 8: `explorer.html` and `css/app.css`**

In `explorer.html`, insert `<div id="threshold-control"></div>` between the `<h1>` and
`<div class="d-flex flex-wrap gap-2" id="pickers"></div>`. In `app.css`:
- add `position: relative;` as the first declaration of `.prob-track`;
- add before `.cell-mismatch`:

```css
.prob-tick {
  position: absolute;
  top: -0.2rem;
  bottom: -0.2rem;
  width: 2px;
  background: var(--bs-emphasis-color);
  opacity: 0.6;
}

.cell-partial {
  background: rgba(var(--bs-warning-rgb), 0.18) !important;
}
```

- [ ] **Step 9: Verify**

Run:
```bash
for f in src/jev_bench/web/static/js/*.js; do node --check "$f" || echo "FAIL $f"; done
node --test tests/js/
uv run pytest tests/test_web_static.py -q
wc -l src/jev_bench/web/static/js/explorer.js src/jev_bench/web/static/js/email-detail.js
```
Expected:
- no `FAIL`;
- all node tests pass;
- the static contract passes;
- `explorer.js` is under 300 lines (≈ 257) and `email-detail.js` ≈ 102.

The Task 12 browser smoke covers the behavior.

- [ ] **Step 10: Commit**

```bash
git add src/jev_bench/web/static/js/answers.js src/jev_bench/web/static/js/selection.js \
  src/jev_bench/web/static/js/email-detail.js src/jev_bench/web/static/js/explorer.js \
  src/jev_bench/web/static/js/distribution.js src/jev_bench/web/static/explorer.html \
  src/jev_bench/web/static/css/app.css tests/js/answers.test.mjs tests/js/selection.test.mjs \
  tests/test_web_static.py
git commit -m "feat(ui): multi-label Explorer cells, detail panel, human label checkboxes, default runs

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```
