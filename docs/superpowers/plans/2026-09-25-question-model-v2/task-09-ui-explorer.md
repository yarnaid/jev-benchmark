### Task 9: Explorer UI: multi-label cells, detail panel, human checkboxes, default runs

**Files:**
- Create: `src/jev_bench/web/static/js/answers.js`: pure answer comparison, formatting and the applied limit.
- Create: `src/jev_bench/web/static/js/selection.js`: pure run selection.
- Create: `src/jev_bench/web/static/js/email-detail.js`: the detail panel, moved out of `explorer.js`.
- Modify: `src/jev_bench/web/static/js/explorer.js`
- Modify: `src/jev_bench/web/static/js/distribution.js` (the optional `tick`)
- Modify: `src/jev_bench/web/static/explorer.html` (`#threshold-control`)
- Modify: `src/jev_bench/web/static/css/app.css` (`.prob-track` positioning, `.prob-tick`, `.cell-partial`)
- Modify: `tests/test_web_static.py` (`PAGE_MODULES`)
- Test: `tests/js/answers.test.mjs`, `tests/js/selection.test.mjs` (new, `node --test`)

**Why the split:** the detail-panel changes would take `explorer.js` from 282 to about 340 lines, past the
300-line limit (`rules/modularity.md`). Moving the panel into `email-detail.js` leaves `explorer.js` at 257
lines and `email-detail.js` at 104. Both sizes were measured while planning.

**Interfaces:**
- Consumes:
  - Task 7: `EmailRow.top` (a label list for multi), `.scores`, `.reference_scores`; `EmailDetail.human`
    lists; `PUT /api/labels` with lists.
  - Task 8: `api.emails(..., threshold)`, `thresholdSlider` and the `threshold` pref.
- Produces:
  - `answers.js`:
    - `asLabels(answer)`;
    - `setMatch(reference, answer) -> "match" | "partial" | "mismatch" | ""`, which compares a string with
      an array as a one-label set (Review Focus #2);
    - `answerText(answer)`: `"—"` if missing, `"none"` for `[]`;
    - `withScore(text, score)`;
    - `appliedLimit(distribution, threshold) -> number`: `threshold × max(p)`, or `Infinity` when the max is
      0. This is the JS mirror of `relative_labels`, with the same formula.
  - `selection.js`: `sameSet`, and `latestCompletedPerColumn(runs, generationIds)` (sorts newest first
    itself).
  - `email-detail.js`: `emailDetail(detail, { row = null, threshold = null, onSave }) -> Node[]`.

**Behavior** (spec §9):
- Table cells:
  - multi cells list the applied labels;
  - `cell-match` / `cell-partial` / `cell-mismatch` come from `setMatch`;
  - score cells show `level · score`.
- The reference filter matches list membership.
- Without `?runs=` the latest completed run per column is pre-selected, and changing the generations
  re-defaults it.
- The slider (rendered once) re-fetches only the rows, so the displayed question and the filters are kept.
- Detail panel:
  - the multi card shows `≥ N% of top`;
  - per run, a tick marks `threshold × max(p)` on every bar, and labels at or above it are bold;
  - ✓ marks reference labels, and human labels are checkboxes;
  - score cards have a `score (0–100)` footer.
- The smoke run while planning checked this:
  - ticks at 52% (Jev, 0.75 × 0.7), 38% (Claude, 0.75 × 0.5) and 30% (embeddings);
  - `meeting` bold exactly at Claude's boundary;
  - a saved checkbox persisted `["travel"]`.

- [ ] **Step 1: Write the failing JS tests**

Create `tests/js/answers.test.mjs`:

```js
// Run with `node --test tests/js/`. Exercises src/jev_bench/web/static/js/answers.js (pure, no DOM).
import assert from "node:assert/strict";
import test from "node:test";

const { asLabels, setMatch, answerText, withScore, appliedLimit } = await import("../../src/jev_bench/web/static/js/answers.js");

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

const LIMITS = [
  ["three quarters of the top", { billing: 0.5, meeting: 0.375, travel: 0.125 }, 0.75, 0.375],
  ["threshold one keeps only the top", { a: 0.6, b: 0.4 }, 1, 0.6],
  ["empty distribution applies nothing", {}, 0.8, Infinity],
  ["all-zero distribution applies nothing", { a: 0, b: 0 }, 0.8, Infinity],
];

for (const [name, distribution, threshold, expected] of LIMITS) {
  test(`appliedLimit: ${name}`, () => assert.equal(appliedLimit(distribution, threshold), expected));
}
```

Create `tests/js/selection.test.mjs`:

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

Apply to `tests/test_web_static.py`:

```diff
--- a/tests/test_web_static.py
+++ b/tests/test_web_static.py
@@ -21,6 +21,9 @@
     "explorer.js",
     "runs.js",
     "distribution.js",
+    "answers.js",
+    "selection.js",
+    "email-detail.js",
 )
 
 
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `node --test tests/js/ ; uv run pytest tests/test_web_static.py -q`
Expected: the node tests fail with `ERR_MODULE_NOT_FOUND`, and `test_page_modules_exist` fails.

- [ ] **Step 3: Implement the pure modules**

Create `src/jev_bench/web/static/js/answers.js`:

```js
/**
 * Pure helpers for comparing and displaying answers. A multi-label answer is an array of labels; any
 * other answer is one option id. A string compared with an array is treated as a one-label array, so
 * an old single-label reference still matches a multi-label answer. A multi-label question applies a
 * label when its probability is at least `appliedLimit` (threshold x the top probability), the same
 * rule as the server's metrics.multilabel.relative_labels.
 * Exports: asLabels, setMatch, answerText, withScore, appliedLimit.
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

export function appliedLimit(distribution, threshold) {
  const top = Math.max(0, ...Object.values(distribution ?? {}));
  return top > 0 ? threshold * top : Infinity;
}
```

Create `src/jev_bench/web/static/js/selection.js`:

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

Run: `node --test tests/js/`. Expected: all pass (72 while planning).

- [ ] **Step 4: Move the detail panel into `email-detail.js`, and adapt `explorer.js` and `distribution.js`**

`emailHeader` and `humanSelect` move **unchanged** from `explorer.js`. `labellingForm`, `questionSection` and
`runCell` are adapted, and `humanControl`, `humanLabels` and `scoreRow` are new. Unchecking every box stores
`[]`, which the server treats as "cleared" (Task 7), so `saveLabels` needs no change.

Create `src/jev_bench/web/static/js/email-detail.js`:

```js
/**
 * Explorer detail panel content: the email header, the (hostile-by-design) body as plain text, and one
 * card per question with each run's probability per option, the generator reference, a 0-100 score
 * row for score questions, and the human-label control (a select, or checkboxes for a multi-label
 * question, whose card also shows the threshold; per run, a tick marks threshold x the top probability
 * and labels at or above it are bold).
 * Exports: emailDetail.
 */
import { appliedLimit, asLabels } from "./answers.js";
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
  const badge = cut === null ? null : h("span", { class: "badge text-bg-info", title: "A label counts as applied when its probability is at least this share of the most probable label" }, `≥ ${Math.round(cut * 100)}% of top`);
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
  if (cut === null) return probabilityCell(distribution[option], { similarity, highlight: topOption(distribution) === option });
  const limit = appliedLimit(distribution, cut);
  return probabilityCell(distribution[option], { similarity, highlight: (distribution[option] ?? 0) >= limit, tick: limit });
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

Apply to `src/jev_bench/web/static/js/explorer.js`:

```diff
--- a/src/jev_bench/web/static/js/explorer.js
+++ b/src/jev_bench/web/static/js/explorer.js
@@ -1,17 +1,23 @@
 /**
- * Explorer page: browse the emails of selected generations, compare each run's top answer with the
- * generator reference, sort by disagreement, and open an email to see every distribution and edit
- * human labels.
+ * Explorer page: browse the emails of selected generations, compare each run's top answer (the applied
+ * labels of a multi-label question; level and 0-100 score of a score question) with the generator
+ * reference (match / partial / mismatch), sort by disagreement, and open an email to see every
+ * distribution and edit human labels. Without ?runs= the latest completed run per column on the
+ * selected generations is pre-selected; the label-threshold slider re-fetches the rows.
  * Exports: none (page entry point).
  */
+import { answerText, asLabels, setMatch, withScore } from "./answers.js";
 import { api } from "./api.js";
-import { probabilityCell } from "./distribution.js";
 import { clear, h, icon } from "./dom.js";
+import { emailDetail } from "./email-detail.js";
 import { fixed, shortModel, when } from "./format.js";
 import { initLayout, toastError, toastSuccess } from "./layout.js";
-import { checklist, emptyState } from "./widgets.js";
+import { latestCompletedPerColumn } from "./selection.js";
+import { readPref, writePref } from "./storage.js";
+import { checklist, emptyState, thresholdSlider } from "./widgets.js";
 
 const EMPTY_FILTERS = { text: "", generator: "", reference: "", trait: "" };
+const MATCH_CLASS = { match: "cell-match", partial: "cell-partial", mismatch: "cell-mismatch", "": "" };
 const state = {
   generations: [],
   runs: [],
@@ -22,12 +28,14 @@
   rows: [],
   filters: { ...EMPTY_FILTERS },
   byDisagreement: true,
+  threshold: readPref("threshold", null),
+  sliderShown: false,
 };
 
 const listParam = (params, name) => (params.get(name) ?? "").split(",").filter(Boolean);
 const matchesPair = (values, pair) => {
   const [key, value] = pair.split("=");
-  return values[key] === value;
+  return asLabels(values[key]).includes(value);
 };
 const runLabel = (run) => `${run.column} · ${shortModel(run.model)}${run.mode === "all_in_one" ? " · all-in-one" : ""}`;
 
@@ -39,7 +47,8 @@
   state.runs = runs.map((view) => view.meta).filter((run) => run.status !== "running");
   const requested = listParam(params, "generations");
   state.selectedGenerations = requested.length ? requested : state.generations.slice(0, 1).map((generation) => generation.id);
-  state.selectedRuns = listParam(params, "runs");
+  const requestedRuns = listParam(params, "runs");
+  state.selectedRuns = requestedRuns.length ? requestedRuns : latestCompletedPerColumn(state.runs, state.selectedGenerations);
   renderPickers();
   await loadRows();
 }
@@ -57,6 +66,7 @@
 
 function onGenerations(values) {
   state.selectedGenerations = values;
+  state.selectedRuns = latestCompletedPerColumn(state.runs, values);
   syncUrl();
   renderPickers();
   loadRows().catch(toastError);
@@ -88,13 +98,38 @@
     return;
   }
   clear(target, h("p", { class: "small text-body-secondary" }, "Loading…"));
-  const list = await api.emails(state.selectedGenerations, activeRuns());
+  const list = await api.emails(state.selectedGenerations, activeRuns(), state.threshold);
   state.questions = list.questions.questions;
   state.question = state.questions[0]?.id ?? null;
   state.rows = list.rows;
   state.filters = { ...EMPTY_FILTERS };
   renderFilters();
+  renderThreshold();
   renderTable();
+}
+
+async function refreshRows() {
+  state.rows = (await api.emails(state.selectedGenerations, activeRuns(), state.threshold)).rows;
+  renderTable();
+}
+
+function renderThreshold() {
+  const target = document.getElementById("threshold-control");
+  const multi = state.questions.find((question) => question.type === "multi");
+  if (!multi) {
+    clear(target);
+    state.sliderShown = false;
+    return;
+  }
+  if (state.sliderShown) return;
+  clear(target, thresholdSlider({ value: state.threshold ?? multi.threshold, onChange: onThreshold }));
+  state.sliderShown = true;
+}
+
+function onThreshold(value) {
+  state.threshold = value;
+  writePref("threshold", value);
+  refreshRows().catch(toastError);
 }
 
 function filterSelect(label, key, options) {
@@ -173,8 +208,8 @@
   const reference = row.reference[state.question];
   const cells = runs.map((id) => {
     const answer = row.top[id]?.[state.question];
-    const style = answer === undefined ? "" : answer === reference ? "cell-match" : "cell-mismatch";
-    return h("td", { class: `small ${style}` }, answer ?? "—");
+    const text = withScore(answerText(answer), row.scores[id]?.[state.question]);
+    return h("td", { class: `small ${MATCH_CLASS[setMatch(reference, answer)]}` }, text);
   });
   const labelled = Object.keys(row.human).length ? h("span", { class: "badge text-bg-info ms-1", title: "Has human labels" }, icon("person-check")) : null;
   const open = () => openEmail(row.id);
@@ -185,7 +220,7 @@
     h("td", { class: "small text-truncate", style: "max-width: 14rem", title: row.sender }, row.sender),
     h("td", { class: "text-truncate", style: "max-width: 22rem", title: row.subject }, row.subject),
     h("td", { class: "small" }, shortModel(row.generator_model)),
-    h("td", { class: "small fw-semibold" }, reference ?? "—", labelled),
+    h("td", { class: "small fw-semibold" }, withScore(answerText(reference), row.reference_scores[state.question]), labelled),
     cells,
     h("td", { class: "font-monospace small" }, fixed(row.disagreement, 3)),
   );
@@ -197,72 +232,12 @@
   bootstrap.Offcanvas.getOrCreateInstance(document.getElementById("email-panel")).show();
   try {
     const detail = await api.email(emailId, runColumns());
+    const row = state.rows.find((item) => item.id === emailId) ?? null;
     clear(document.getElementById("email-panel-title"), detail.email.subject);
-    clear(body, emailHeader(detail.email), h("div", { class: "email-body mb-3" }, detail.email.body), labellingForm(detail));
+    clear(body, emailDetail(detail, { row, threshold: state.threshold, onSave: saveLabels }));
   } catch (error) {
     clear(body, h("div", { class: "alert alert-danger" }, error instanceof Error ? error.message : String(error)));
   }
-}
-
-function emailHeader(email) {
-  const party = (value) => (value.name ? `${value.name} <${value.address}>` : value.address);
-  const traits = Object.entries(email.traits).map(([name, value]) => `${name}=${value}`).join(", ");
-  const rows = [
-    ["From", party(email.sender)],
-    ["To", email.to.map(party).join(", ")],
-    ["Cc", email.cc.map(party).join(", ") || "—"],
-    ["Sent", when(email.sent_at)],
-    ["Generator", email.generator_model],
-    ["Traits", traits || "—"],
-    ["Id", email.id],
-  ];
-  return h("dl", { class: "row small mb-2" }, rows.flatMap(([label, value]) => [h("dt", { class: "col-3" }, label), h("dd", { class: "col-9 text-break mb-1" }, value)]));
-}
-
-function labellingForm(detail) {
-  const choices = { ...detail.human };
-  const runIds = Object.keys(detail.predictions);
-  const sections = detail.questions.questions.map((question) => questionSection(question, detail, runIds, choices));
-  const save = h("button", { class: "btn btn-primary", type: "button", onclick: () => saveLabels(detail.email.id, choices, detail.human) }, icon("save"), " Save labels");
-  return h("div", {}, sections, h("div", { class: "d-flex justify-content-end mt-2" }, save));
-}
-
-function humanSelect(question, current, choices) {
-  const onchange = (event) => {
-    choices[question.id] = event.target.value || null;
-  };
-  const options = Object.entries(question.options).map(([option, text]) => h("option", { value: option, selected: current === option, title: text }, option));
-  return h("select", { class: "form-select form-select-sm", "aria-label": `Human label for ${question.id}`, onchange }, h("option", { value: "" }, "— not labelled —"), options);
-}
-
-function questionSection(question, detail, runIds, choices) {
-  const reference = detail.email.reference_answers[question.id];
-  const head = ["Option", "Ref", ...runIds.map((id) => detail.run_labels[id] ?? id)];
-  const rows = Object.entries(question.options).map(([option, text]) =>
-    h("tr", {}, h("td", { class: "small", title: text }, option), h("td", { class: "text-center" }, reference === option ? icon("check-lg") : ""), runIds.map((id) => runCell(detail.predictions[id], question.id, option))),
-  );
-  const table = h("table", { class: "table table-sm mb-2" }, h("thead", {}, h("tr", {}, head.map((cell) => h("th", { class: "small" }, cell)))), h("tbody", {}, rows));
-  return h(
-    "div",
-    { class: "card mb-2" },
-    h("div", { class: "card-header py-1 d-flex align-items-center gap-2" }, h("code", {}, question.id), h("span", { class: "small text-body-secondary text-truncate" }, question.instructions)),
-    h(
-      "div",
-      { class: "card-body py-2" },
-      h("div", { class: "table-responsive" }, table),
-      h("div", { class: "d-flex align-items-center gap-2" }, h("span", { class: "small text-nowrap" }, icon("person"), " Human"), humanSelect(question, detail.human[question.id], choices)),
-    ),
-  );
-}
-
-function runCell(prediction, questionId, option) {
-  if (!prediction) return h("td", { class: "small text-body-secondary" }, "—");
-  if (prediction.error) return h("td", { class: "small text-danger", title: prediction.error }, "error");
-  const distribution = prediction.answers?.[questionId];
-  if (!distribution) return h("td", { class: "small text-body-secondary" }, "—");
-  const top = Object.entries(distribution).reduce((best, entry) => (entry[1] > best[1] ? entry : best))[0];
-  const similarity = prediction.similarities?.[questionId]?.[option] ?? null;
-  return probabilityCell(distribution[option], { similarity, highlight: top === option });
 }
 
 async function saveLabels(emailId, choices, original) {
```

Apply to `src/jev_bench/web/static/js/distribution.js`:

```diff
--- a/src/jev_bench/web/static/js/distribution.js
+++ b/src/jev_bench/web/static/js/distribution.js
@@ -1,14 +1,16 @@
 /**
  * Probability visualisation for the Explorer detail panel: a table cell with a bar and the value;
- * the tooltip adds the cosine similarity for embedding runs.
+ * the tooltip adds the cosine similarity for embedding runs, and an optional `tick` (a probability,
+ * e.g. a multi-label question's applied limit) is drawn as a mark on the bar.
  * Exports: probabilityCell.
  */
 import { h } from "./dom.js";
 import { fixed } from "./format.js";
 
-export function probabilityCell(probability, { similarity = null, highlight = false } = {}) {
+export function probabilityCell(probability, { similarity = null, highlight = false, tick = null } = {}) {
   const value = probability ?? 0;
   const title = similarity === null ? `p = ${fixed(value, 3)}` : `p = ${fixed(value, 3)} · cos = ${fixed(similarity, 3)}`;
-  const bar = h("div", { class: "prob-track flex-grow-1" }, h("div", { class: "prob-bar", style: `width: ${Math.round(value * 100)}%` }));
+  const mark = tick === null || !Number.isFinite(tick) ? null : h("div", { class: "prob-tick", style: `left: ${Math.round(tick * 100)}%`, "aria-hidden": "true" });
+  const bar = h("div", { class: "prob-track flex-grow-1" }, h("div", { class: "prob-bar", style: `width: ${Math.round(value * 100)}%` }), mark);
   return h("td", { class: highlight ? "fw-semibold" : "", title }, h("div", { class: "d-flex align-items-center gap-2" }, bar, h("span", { class: "small font-monospace" }, fixed(value, 2))));
 }
```

Apply to `src/jev_bench/web/static/explorer.html`:

```diff
--- a/src/jev_bench/web/static/explorer.html
+++ b/src/jev_bench/web/static/explorer.html
@@ -17,6 +17,7 @@
   <main class="container-fluid pb-5">
     <div class="d-flex flex-wrap align-items-center gap-2 mb-3">
       <h1 class="h4 mb-0 me-auto">Explorer</h1>
+      <div id="threshold-control"></div>
       <div class="d-flex flex-wrap gap-2" id="pickers"></div>
     </div>
     <div class="row g-2 mb-3" id="filters"></div>
```

Apply to `src/jev_bench/web/static/css/app.css`:

```diff
--- a/src/jev_bench/web/static/css/app.css
+++ b/src/jev_bench/web/static/css/app.css
@@ -47,6 +47,7 @@
 }
 
 .prob-track {
+  position: relative;
   background: var(--bs-secondary-bg);
   border-radius: 0.25rem;
   height: 0.5rem;
@@ -61,6 +62,19 @@
 
 .cell-match {
   background: rgba(var(--bs-success-rgb), 0.12) !important;
+}
+
+.prob-tick {
+  position: absolute;
+  top: -0.2rem;
+  bottom: -0.2rem;
+  width: 2px;
+  background: var(--bs-emphasis-color);
+  opacity: 0.6;
+}
+
+.cell-partial {
+  background: rgba(var(--bs-warning-rgb), 0.18) !important;
 }
 
 .cell-mismatch {
```

- [ ] **Step 5: Verify**

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
- `explorer.js` is under 300 lines.

Task 10 smoke-tests the behavior.

- [ ] **Step 6: Commit**

```bash
git add src/jev_bench/web/static/js/answers.js src/jev_bench/web/static/js/selection.js \
  src/jev_bench/web/static/js/email-detail.js src/jev_bench/web/static/js/explorer.js \
  src/jev_bench/web/static/js/distribution.js src/jev_bench/web/static/explorer.html \
  src/jev_bench/web/static/css/app.css tests/js/answers.test.mjs tests/js/selection.test.mjs \
  tests/test_web_static.py
git commit -m "feat(ui): multi-label Explorer cells, detail panel, human label checkboxes, default runs

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```
