### Task 10: Benchmark UI: threshold slider, multi-label and 0–100 score report cards

**Files:**
- Modify: `src/jev_bench/web/static/js/api.js` (`query()` skips `null`; `threshold` for `compare`/`emails`)
- Modify: `src/jev_bench/web/static/js/widgets.js` (`thresholdSlider`)
- Modify: `src/jev_bench/web/static/js/charts.js` (applied-label counts)
- Modify: `src/jev_bench/web/static/js/report.js` (multi card, mean score)
- Modify: `src/jev_bench/web/static/js/benchmark.js` (slider in page chrome, `?threshold=`)
- Modify: `src/jev_bench/web/static/index.html` (`#threshold-control`)
- Modify: `src/jev_bench/web/static/css/app.css` (`.threshold-range`)
- Verify: `node --check`, `node --test tests/js/`, `tests/test_web_static.py`, and the browser smoke (Task 12)

**Interfaces:**
- Consumes (Task 9 API):
  - `GET /api/compare?runs=…&threshold=`;
  - `QuestionReport.type === "multi"` and `.threshold`;
  - `RaterStats.label_counts` / `.coverage` / `.mean_labels` / `.mean_score`;
  - `PairStats.jaccard` / `.jaccard_ci` / `.f1`.
- Produces (Task 11 uses these):
  - `api.compare(runIds, threshold)` and `api.emails(generationIds, runIds = [], threshold)`. An undefined or
    `null` threshold is omitted from the query string.
  - `widgets.thresholdSlider({ value, onChange, delayMs = 300 })`: a range from 50 to 100 step 5 with a
    live `%` output. It calls `onChange(fraction)` once, `delayMs` after the last `input` event.
  - The `localStorage` pref `threshold` (via `storage.js`), shared by the Benchmark and Explorer pages.

**Behavior** (spec §9; planning revision 4):
- The slider lives in `#threshold-control` in the page header. It is rendered **once**, when the first report
  containing a multi question arrives, so a re-rendered report never destroys the slider being dragged.
- The report re-fetches with `?threshold=`.

The smoke run during planning checked the behavior:
- Moving the slider to 95% re-fetched the report.
- The slider element and its focus survived.
- The pref `jev-bench.threshold` became `0.95`.
- Coverage for Jev and Claude dropped to 0%, and stayed at 100% for embeddings: their best label is always
  1.0, by design (Task 5).

- [ ] **Step 1: `js/api.js`**

Add this line to the header comment:
` * Query parameters that are undefined, null or "" are omitted (an unset label threshold is never sent).`

The `query()` filter becomes:

```js
  const entries = Object.entries(params).filter(([, value]) => value !== undefined && value !== null && value !== "");
```

Replace the `compare` and `emails` entries:

```js
  compare: (runIds, threshold) => request("GET", `/compare${query({ runs: runIds.join(","), threshold })}`),
  emails: (generationIds, runIds = [], threshold) =>
    request("GET", `/emails${query({ generations: generationIds.join(","), runs: runIds.join(","), threshold })}`),
```

- [ ] **Step 2: `js/widgets.js`**

Replace the header comment's description and exports with:

```js
 * Reusable widgets built with h(): status badges, progress bars, empty states, a checkbox dropdown
 * ("checklist") taking items [{ value, text, hint }] and calling onChange(selectedValues), and a
 * multi-label threshold slider (50-100 %, step 5) calling onChange(fraction) after a debounce.
 * Exports: statusBadge, progressBar, emptyState, checklist, thresholdSlider.
```

Append:

```js
export function thresholdSlider({ value, onChange, delayMs = 300 }) {
  const percent = (fraction) => `${Math.round(fraction * 100)}%`;
  const output = h("output", { class: "small font-monospace text-nowrap" }, percent(value));
  let timer = null;
  const oninput = (event) => {
    const next = Number(event.target.value) / 100;
    output.textContent = percent(next);
    clearTimeout(timer);
    timer = setTimeout(() => onChange(next), delayMs);
  };
  const input = h("input", { type: "range", class: "form-range threshold-range", min: 50, max: 100, step: 5, value: Math.round(value * 100), "aria-label": "Label threshold", oninput });
  const label = h("span", { class: "small text-nowrap" }, icon("sliders"), " Label threshold");
  return h("div", { class: "d-flex align-items-center gap-2", title: "A label counts as applied when its probability is at least this value" }, label, input, output);
}
```

`Number("85") / 100` is exactly the IEEE double that Python parses from `"0.85"`, so a slider value and a
probability stored as `0.85` compare equal on the server (Review Focus #3).

- [ ] **Step 3: `js/charts.js`**

The header comment becomes:
` * Chart.js helpers: grouped bar chart of answer counts per option (one dataset per rater): argmax`
` * counts, or applied-label counts for a multi-label question.`
The dataset `data` line becomes:

```js
    data: question.options.map((option) => (stats.label_counts ?? stats.argmax_counts)[option] ?? 0),
```

- [ ] **Step 4: `js/report.js`**

The header comment becomes:

```js
 * Renders a ComparisonReport: a raters summary table, warnings, then one card per question with an
 * answer-count chart, per-rater statistics and pairwise metrics (95% bootstrap CIs) plus Fleiss' kappa.
 * Score questions show a 0-100 mean score; multi-label questions show coverage, labels per email,
 * exact-set match, Jaccard, F1 and the macro kappa at the threshold the report was computed with.
```

In `questionCard`, add these three constants after `canvas`:

```js
  const multi = question.type === "multi";
  const fleiss = `Fleiss κ${multi ? " (mean over labels)" : ""} ${fixed(question.fleiss_kappa, 3)}`;
  const tables = multi ? [multiRaterTable(question, labels), multiPairTable(question, labels)] : [raterTable(question, labels), pairTable(question, labels)];
```

In the header, add the threshold badge after the type badge and use `fleiss`. In the body, use `tables`:

```js
        h("span", { class: "badge text-bg-light border" }, question.type),
        multi ? h("span", { class: "badge text-bg-info", title: "A label counts as applied at this probability or higher" }, `≥ ${Math.round(question.threshold * 100)}%`) : null,
        h("span", { class: "ms-auto small text-body-secondary" }, fleiss),
      ),
      h("div", { class: "card-body" }, h("div", { class: "chart-box mb-3" }, canvas), tables),
```

In `raterTable`, the score column becomes a 0–100 score:

```js
  const extra = question.type === "score" ? "Mean score (0–100)" : question.type === "noul" ? "Mean P(yes)" : null;
```

```js
    const value = question.type === "score" ? fixed(stats.mean_score, 0) : fixed(stats.mean.yes, 3);
```

Add these functions before `pairTable`:

```js
function multiRaterTable(question, labels) {
  const head = ["Rater", "n", "Coverage", "Labels / email", "Uncertainty", "Confidence"];
  const rows = question.raters.map((stats) =>
    h(
      "tr",
      {},
      h("td", { class: "small" }, labels[stats.rater] ?? stats.rater),
      h("td", {}, num(stats.n)),
      h("td", {}, pct(stats.coverage)),
      h("td", {}, fixed(stats.mean_labels, 2)),
      h("td", {}, fixed(stats.mean_entropy, 3)),
      h("td", {}, pct(stats.mean_confidence)),
    ),
  );
  return h("div", { class: "table-responsive" }, table(head, rows));
}

function noPairs() {
  return h("p", { class: "small text-body-secondary mb-0" }, "No overlapping raters for this question.");
}

function pairName(pair, labels) {
  return h("td", { class: "small" }, `${labels[pair.a] ?? pair.a} ↔ ${labels[pair.b] ?? pair.b}`);
}

function multiPairTable(question, labels) {
  if (!question.pairs.length) return noPairs();
  const head = ["Pair", "n", "Exact match", "Jaccard", "F1", "κ (macro)", "JSD", "Brier"];
  const two = (value) => fixed(value, 2);
  const rows = question.pairs.map((pair) =>
    h(
      "tr",
      {},
      pairName(pair, labels),
      h("td", {}, num(pair.n)),
      h("td", {}, pct(pair.agreement), interval(pair.agreement_ci, pct)),
      h("td", {}, fixed(pair.jaccard, 3), interval(pair.jaccard_ci, two)),
      h("td", {}, fixed(pair.f1, 3)),
      h("td", {}, fixed(pair.kappa, 3), interval(pair.kappa_ci, two)),
      h("td", {}, fixed(pair.jsd, 3)),
      h("td", {}, fixed(pair.brier, 3)),
    ),
  );
  return h("div", { class: "table-responsive" }, table(head, rows));
}
```

In the existing `pairTable`:
- the empty case becomes `if (!question.pairs.length) return noPairs();`;
- its first cell becomes `pairName(pair, labels),`.

- [ ] **Step 5: `js/benchmark.js`**

Append to the header comment:

```js
 * A label-threshold slider appears in the page header when the report has a multi-label question; it is
 * rendered once (so dragging never loses focus) and re-fetches the comparison with ?threshold=.
```

- Add `thresholdSlider` to the `./widgets.js` import.
- Extend `state` with `threshold: readPref("threshold", null), sliderShown: false`.
- Replace the end of `refreshComparison` and add two functions:

```js
  clear(container, h("p", { class: "small text-body-secondary" }, "Loading comparison…"));
  const report = await api.compare(runIds, state.threshold);
  renderThreshold(report);
  renderReport(container, report, { pinned: state.pinned.length > 0 });
}

function renderThreshold(report) {
  const target = document.getElementById("threshold-control");
  const multi = report.questions.find((question) => question.type === "multi");
  if (!multi) {
    clear(target);
    state.sliderShown = false;
    return;
  }
  if (state.sliderShown) return;
  clear(target, thresholdSlider({ value: multi.threshold, onChange: onThreshold }));
  state.sliderShown = true;
}

function onThreshold(value) {
  state.threshold = value;
  writePref("threshold", value);
  refreshComparison().catch(toastError);
}
```

(`multi.threshold` is the threshold the server actually used: the stored pref when one is set, otherwise the
question's default.)

- [ ] **Step 6: `index.html` and `css/app.css`**

In `index.html`, insert `<div id="threshold-control"></div>` between the `<h1>` and
`<div id="generation-picker"></div>`. In `app.css`, add before `.clickable`:

```css
.threshold-range {
  width: 8rem;
}
```

- [ ] **Step 7: Verify**

Run:
```bash
for f in src/jev_bench/web/static/js/*.js; do node --check "$f" || echo "FAIL $f"; done
node --test tests/js/
uv run pytest tests/test_web_static.py -q
```
Expected: no `FAIL`; the node tests pass; the static contract passes (no HTML sinks, imports resolve). The
behavior is verified by the browser smoke in Task 12. It already passed once while planning: slider re-fetch,
focus kept, multi and score columns rendered, no console errors.

- [ ] **Step 8: Commit**

```bash
git add src/jev_bench/web/static/js/api.js src/jev_bench/web/static/js/widgets.js \
  src/jev_bench/web/static/js/charts.js src/jev_bench/web/static/js/report.js \
  src/jev_bench/web/static/js/benchmark.js src/jev_bench/web/static/index.html \
  src/jev_bench/web/static/css/app.css
git commit -m "feat(ui): label threshold slider and multi-label / 0–100 score report cards

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```
