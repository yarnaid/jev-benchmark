### Task 8: Benchmark UI: threshold slider, multi-label and 0–100 score report cards

**Files:**
- Modify: `src/jev_bench/web/static/js/api.js`:
  - `query()` skips `null`;
  - `threshold` is passed for `compare` and `emails`.
- Modify: `src/jev_bench/web/static/js/widgets.js` (`thresholdSlider`)
- Modify: `src/jev_bench/web/static/js/charts.js` (applied-label counts)
- Modify: `src/jev_bench/web/static/js/report.js` (the multi card, the mean score)
- Modify: `src/jev_bench/web/static/js/benchmark.js` (the slider in page chrome, `?threshold=`)
- Modify: `src/jev_bench/web/static/index.html` (`#threshold-control`)
- Modify: `src/jev_bench/web/static/css/app.css` (`.threshold-range`)
- Verify: `node --check`, `node --test tests/js/`, `tests/test_web_static.py`, and the browser smoke (Task 10)

**Interfaces:**
- Consumes (the Task 7 API):
  - `GET /api/compare?threshold=`;
  - `QuestionReport.type === "multi"` and `.threshold`;
  - `RaterStats.label_counts` / `.mean_labels` / `.mean_score`;
  - `PairStats.jaccard` / `.jaccard_ci` / `.f1`.
- Produces (Task 9 uses these):
  - `api.compare(runIds, threshold)` and `api.emails(generationIds, runIds = [], threshold)`. An undefined or
    `null` threshold is omitted from the query string.
  - `widgets.thresholdSlider({ value, onChange, delayMs = 300 })`: a range from 50 to 100, step 5, with a live
    `%` output; it calls `onChange(fraction)` `delayMs` after the last `input`.
  - The `localStorage` pref `threshold`, shared by the Benchmark and Explorer pages.

**Behavior** (spec §9): the slider lives in `#threshold-control` in the page header. It is rendered **once**,
when the first report containing a multi question arrives, so a re-rendered report never destroys the slider
being dragged. The multi card shows a `≥ N% of top` badge. The smoke run while planning checked the
behavior:
- the chat run showed 2.00 labels per email at 75%, because `meeting` sits exactly on the boundary;
- at 100% it showed 1.00 (the top label only);
- the slider kept its focus across the re-fetch;
- there were no console errors.

- [ ] **Step 1: Implement**

Apply to `src/jev_bench/web/static/js/api.js`:

```diff
--- a/src/jev_bench/web/static/js/api.js
+++ b/src/jev_bench/web/static/js/api.js
@@ -1,5 +1,6 @@
 /**
  * JSON client for the /api endpoints; the browser-stored key is attached only to job-starting calls.
+ * Query parameters that are undefined, null or "" are omitted (an unset label threshold is never sent).
  * Exports: api, ApiError, needsKey.
  */
 import { getKey } from "./key.js";
@@ -24,7 +25,7 @@
 }
 
 function query(params) {
-  const entries = Object.entries(params).filter(([, value]) => value !== undefined && value !== "");
+  const entries = Object.entries(params).filter(([, value]) => value !== undefined && value !== null && value !== "");
   const text = new URLSearchParams(entries).toString();
   return text ? `?${text}` : "";
 }
@@ -42,9 +43,9 @@
   run: (id) => request("GET", `/runs/${segment(id)}`),
   createRun: (body) => request("POST", "/runs", { body, withKey: true }),
   cancelRun: (id) => request("POST", `/runs/${segment(id)}/cancel`),
-  compare: (runIds) => request("GET", `/compare${query({ runs: runIds.join(",") })}`),
-  emails: (generationIds, runIds = []) =>
-    request("GET", `/emails${query({ generations: generationIds.join(","), runs: runIds.join(",") })}`),
+  compare: (runIds, threshold) => request("GET", `/compare${query({ runs: runIds.join(","), threshold })}`),
+  emails: (generationIds, runIds = [], threshold) =>
+    request("GET", `/emails${query({ generations: generationIds.join(","), runs: runIds.join(","), threshold })}`),
   email: (id, runIds = []) => request("GET", `/emails/${segment(id)}${query({ runs: runIds.join(",") })}`),
   putLabel: (id, answers) => request("PUT", `/labels/${segment(id)}`, { body: { answers } }),
 };
```

Apply to `src/jev_bench/web/static/js/widgets.js`:

```diff
--- a/src/jev_bench/web/static/js/widgets.js
+++ b/src/jev_bench/web/static/js/widgets.js
@@ -1,7 +1,9 @@
 /**
- * Reusable widgets built with h(): status badges, progress bars, empty states and a checkbox dropdown
- * ("checklist") taking items [{ value, text, hint }] and calling onChange(selectedValues).
- * Exports: statusBadge, progressBar, emptyState, checklist.
+ * Reusable widgets built with h(): status badges, progress bars, empty states, a checkbox dropdown
+ * ("checklist") taking items [{ value, text, hint }] and calling onChange(selectedValues), and a
+ * multi-label threshold slider (50-100 % of the top probability, step 5) calling onChange(fraction)
+ * after a debounce.
+ * Exports: statusBadge, progressBar, emptyState, checklist, thresholdSlider.
  */
 import { h, icon } from "./dom.js";
 
@@ -59,3 +61,19 @@
     h("ul", { class: "dropdown-menu shadow-sm checklist-menu" }, options.length ? options : empty),
   );
 }
+
+export function thresholdSlider({ value, onChange, delayMs = 300 }) {
+  const percent = (fraction) => `${Math.round(fraction * 100)}%`;
+  const output = h("output", { class: "small font-monospace text-nowrap" }, percent(value));
+  let timer = null;
+  const oninput = (event) => {
+    const next = Number(event.target.value) / 100;
+    output.textContent = percent(next);
+    clearTimeout(timer);
+    timer = setTimeout(() => onChange(next), delayMs);
+  };
+  const input = h("input", { type: "range", class: "form-range threshold-range", min: 50, max: 100, step: 5, value: Math.round(value * 100), "aria-label": "Label threshold", oninput });
+  const label = h("span", { class: "small text-nowrap" }, icon("sliders"), " Label threshold");
+  const hint = "A label counts as applied when its probability is at least this share of the most probable label";
+  return h("div", { class: "d-flex align-items-center gap-2", title: hint }, label, input, output);
+}
```

Apply to `src/jev_bench/web/static/js/charts.js`:

```diff
--- a/src/jev_bench/web/static/js/charts.js
+++ b/src/jev_bench/web/static/js/charts.js
@@ -1,5 +1,6 @@
 /**
- * Chart.js helpers: grouped bar chart of argmax counts per option (one dataset per rater).
+ * Chart.js helpers: grouped bar chart of answer counts per option (one dataset per rater): argmax
+ * counts, or applied-label counts for a multi-label question.
  * Exports: countsChart, destroyCharts.
  */
 
@@ -9,7 +10,7 @@
 export function countsChart(canvas, question, labels) {
   const datasets = question.raters.map((stats, index) => ({
     label: labels[stats.rater] ?? stats.rater,
-    data: question.options.map((option) => stats.argmax_counts[option] ?? 0),
+    data: question.options.map((option) => (stats.label_counts ?? stats.argmax_counts)[option] ?? 0),
     backgroundColor: PALETTE[index % PALETTE.length],
     borderRadius: 3,
   }));
```

Apply to `src/jev_bench/web/static/js/report.js`:

```diff
--- a/src/jev_bench/web/static/js/report.js
+++ b/src/jev_bench/web/static/js/report.js
@@ -1,6 +1,8 @@
 /**
  * Renders a ComparisonReport: a raters summary table, warnings, then one card per question with an
- * argmax-count chart, per-rater statistics and pairwise metrics (95% bootstrap CIs) plus Fleiss' kappa.
+ * answer-count chart, per-rater statistics and pairwise metrics (95% bootstrap CIs) plus Fleiss' kappa.
+ * Score questions show a 0-100 mean score; multi-label questions show labels per email, exact-set
+ * match, Jaccard, F1 and the macro kappa at the threshold the report was computed with.
  * Exports: renderReport.
  */
 import { countsChart, destroyCharts } from "./charts.js";
@@ -58,6 +60,9 @@
 
 function questionCard(question, labels) {
   const canvas = h("canvas", { role: "img", "aria-label": `${question.id} answer counts` });
+  const multi = question.type === "multi";
+  const fleiss = `Fleiss κ${multi ? " (mean over labels)" : ""} ${fixed(question.fleiss_kappa, 3)}`;
+  const tables = multi ? [multiRaterTable(question, labels), multiPairTable(question, labels)] : [raterTable(question, labels), pairTable(question, labels)];
   const element = h(
     "div",
     { class: "col-12 col-xl-6" },
@@ -69,19 +74,20 @@
         { class: "card-header d-flex align-items-center gap-2" },
         h("code", { class: "fw-semibold" }, question.id),
         h("span", { class: "badge text-bg-light border" }, question.type),
-        h("span", { class: "ms-auto small text-body-secondary" }, `Fleiss κ ${fixed(question.fleiss_kappa, 3)}`),
+        multi ? thresholdBadge(question.threshold) : null,
+        h("span", { class: "ms-auto small text-body-secondary" }, fleiss),
       ),
-      h("div", { class: "card-body" }, h("div", { class: "chart-box mb-3" }, canvas), raterTable(question, labels), pairTable(question, labels)),
+      h("div", { class: "card-body" }, h("div", { class: "chart-box mb-3" }, canvas), tables),
     ),
   );
   return { element, draw: () => question.raters.length && countsChart(canvas, question, labels) };
 }
 
 function raterTable(question, labels) {
-  const extra = question.type === "score" ? "Mean level" : question.type === "noul" ? "Mean P(yes)" : null;
+  const extra = question.type === "score" ? "Mean score (0–100)" : question.type === "noul" ? "Mean P(yes)" : null;
   const head = ["Rater", "n", "Entropy", "Confidence", ...(extra ? [extra] : [])];
   const rows = question.raters.map((stats) => {
-    const value = question.type === "score" ? fixed(stats.mean_level, 2) : fixed(stats.mean.yes, 3);
+    const value = question.type === "score" ? fixed(stats.mean_score, 0) : fixed(stats.mean.yes, 3);
     return h("tr", {}, h("td", { class: "small" }, labels[stats.rater] ?? stats.rater), h("td", {}, num(stats.n)), h("td", {}, fixed(stats.mean_entropy, 3)), h("td", {}, pct(stats.mean_confidence)), extra ? h("td", {}, value) : null);
   });
   return h("div", { class: "table-responsive" }, table(head, rows));
@@ -91,14 +97,64 @@
   return bounds ? h("small", { class: "text-body-secondary" }, ` [${format(bounds[0])}, ${format(bounds[1])}]`) : null;
 }
 
+function thresholdBadge(threshold) {
+  const title = "A label counts as applied when its probability is at least this share of the most probable label";
+  return h("span", { class: "badge text-bg-info", title }, `≥ ${Math.round(threshold * 100)}% of top`);
+}
+
+function multiRaterTable(question, labels) {
+  const head = ["Rater", "n", "Labels / email", "Entropy", "Confidence"];
+  const rows = question.raters.map((stats) =>
+    h(
+      "tr",
+      {},
+      h("td", { class: "small" }, labels[stats.rater] ?? stats.rater),
+      h("td", {}, num(stats.n)),
+      h("td", {}, fixed(stats.mean_labels, 2)),
+      h("td", {}, fixed(stats.mean_entropy, 3)),
+      h("td", {}, pct(stats.mean_confidence)),
+    ),
+  );
+  return h("div", { class: "table-responsive" }, table(head, rows));
+}
+
+function noPairs() {
+  return h("p", { class: "small text-body-secondary mb-0" }, "No overlapping raters for this question.");
+}
+
+function pairName(pair, labels) {
+  return h("td", { class: "small" }, `${labels[pair.a] ?? pair.a} ↔ ${labels[pair.b] ?? pair.b}`);
+}
+
+function multiPairTable(question, labels) {
+  if (!question.pairs.length) return noPairs();
+  const head = ["Pair", "n", "Exact match", "Jaccard", "F1", "κ (macro)", "JSD", "Brier"];
+  const two = (value) => fixed(value, 2);
+  const rows = question.pairs.map((pair) =>
+    h(
+      "tr",
+      {},
+      pairName(pair, labels),
+      h("td", {}, num(pair.n)),
+      h("td", {}, pct(pair.agreement), interval(pair.agreement_ci, pct)),
+      h("td", {}, fixed(pair.jaccard, 3), interval(pair.jaccard_ci, two)),
+      h("td", {}, fixed(pair.f1, 3)),
+      h("td", {}, fixed(pair.kappa, 3), interval(pair.kappa_ci, two)),
+      h("td", {}, fixed(pair.jsd, 3)),
+      h("td", {}, fixed(pair.brier, 3)),
+    ),
+  );
+  return h("div", { class: "table-responsive" }, table(head, rows));
+}
+
 function pairTable(question, labels) {
-  if (!question.pairs.length) return h("p", { class: "small text-body-secondary mb-0" }, "No overlapping raters for this question.");
+  if (!question.pairs.length) return noPairs();
   const head = ["Pair", "n", "Agreement", "κ", "JSD", "r", "Brier"];
   const rows = question.pairs.map((pair) =>
     h(
       "tr",
       {},
-      h("td", { class: "small" }, `${labels[pair.a] ?? pair.a} ↔ ${labels[pair.b] ?? pair.b}`),
+      pairName(pair, labels),
       h("td", {}, num(pair.n)),
       h("td", {}, pct(pair.agreement), interval(pair.agreement_ci, pct)),
       h("td", {}, fixed(pair.kappa, 3), interval(pair.kappa_ci, (value) => fixed(value, 2))),
```

Apply to `src/jev_bench/web/static/js/benchmark.js`:

```diff
--- a/src/jev_bench/web/static/js/benchmark.js
+++ b/src/jev_bench/web/static/js/benchmark.js
@@ -1,6 +1,8 @@
 /**
  * Benchmark page: generation picker, one card per column (model, mode, run/cancel, live stats) and the
  * comparison of the latest completed run per column on exactly the selected generations (or ?runs=…).
+ * A label-threshold slider appears in the page header when the report has a multi-label question; it is
+ * rendered once (so dragging never loses focus) and re-fetches the comparison with ?threshold=.
  * Exports: none (page entry point).
  */
 import { api } from "./api.js";
@@ -9,10 +11,10 @@
 import { initLayout, startJob, toastError } from "./layout.js";
 import { renderReport } from "./report.js";
 import { readPref, writePref } from "./storage.js";
-import { checklist, emptyState, progressBar, statusBadge } from "./widgets.js";
+import { checklist, emptyState, progressBar, statusBadge, thresholdSlider } from "./widgets.js";
 
 const POLL_MS = 1000;
-const state = { catalog: [], generations: [], runs: [], progress: new Map(), selected: [], pinned: [], timers: new Map() };
+const state = { catalog: [], generations: [], runs: [], progress: new Map(), selected: [], pinned: [], timers: new Map(), threshold: readPref("threshold", null), sliderShown: false };
 
 const sameSet = (values, set) => values.length === set.size && values.every((value) => set.has(value));
 const selectedSet = () => new Set(state.selected);
@@ -237,7 +239,28 @@
     return;
   }
   clear(container, h("p", { class: "small text-body-secondary" }, "Loading comparison…"));
-  renderReport(container, await api.compare(runIds), { pinned: state.pinned.length > 0 });
+  const report = await api.compare(runIds, state.threshold);
+  renderThreshold(report);
+  renderReport(container, report, { pinned: state.pinned.length > 0 });
+}
+
+function renderThreshold(report) {
+  const target = document.getElementById("threshold-control");
+  const multi = report.questions.find((question) => question.type === "multi");
+  if (!multi) {
+    clear(target);
+    state.sliderShown = false;
+    return;
+  }
+  if (state.sliderShown) return;
+  clear(target, thresholdSlider({ value: multi.threshold, onChange: onThreshold }));
+  state.sliderShown = true;
+}
+
+function onThreshold(value) {
+  state.threshold = value;
+  writePref("threshold", value);
+  refreshComparison().catch(toastError);
 }
 
 main().catch(toastError);
```

Apply to `src/jev_bench/web/static/index.html`:

```diff
--- a/src/jev_bench/web/static/index.html
+++ b/src/jev_bench/web/static/index.html
@@ -17,6 +17,7 @@
   <main class="container-fluid pb-5">
     <div class="d-flex flex-wrap align-items-center gap-2 mb-3">
       <h1 class="h4 mb-0 me-auto">Benchmark</h1>
+      <div id="threshold-control"></div>
       <div id="generation-picker"></div>
     </div>
     <div class="row g-3 mb-4" id="columns"></div>
```

Apply to `src/jev_bench/web/static/css/app.css`:

```diff
--- a/src/jev_bench/web/static/css/app.css
+++ b/src/jev_bench/web/static/css/app.css
@@ -80,6 +80,10 @@
   min-width: 18rem;
 }
 
+.threshold-range {
+  width: 8rem;
+}
+
 .clickable {
   cursor: pointer;
 }
```

`Number("85") / 100` is exactly the IEEE double that Python parses from `"0.85"`, so a slider value compares on
the server exactly as the same literal would.

- [ ] **Step 2: Verify**

Run:
```bash
for f in src/jev_bench/web/static/js/*.js; do node --check "$f" || echo "FAIL $f"; done
node --test tests/js/
uv run pytest tests/test_web_static.py -q
```
Expected: no `FAIL`; the node tests pass; the static contract passes (no HTML sinks, imports resolve). Task 10
smoke-tests the behavior.

- [ ] **Step 3: Commit**

```bash
git add src/jev_bench/web/static/js/api.js src/jev_bench/web/static/js/widgets.js \
  src/jev_bench/web/static/js/charts.js src/jev_bench/web/static/js/report.js \
  src/jev_bench/web/static/js/benchmark.js src/jev_bench/web/static/index.html \
  src/jev_bench/web/static/css/app.css
git commit -m "feat(ui): label threshold slider and multi-label / 0–100 score report cards

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```
