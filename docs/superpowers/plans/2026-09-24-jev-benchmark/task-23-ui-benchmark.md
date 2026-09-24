### Task 23: Benchmark page

**Files:**
- Replace: `src/jev_bench/web/static/index.html`
- Create: `src/jev_bench/web/static/js/benchmark.js`, `src/jev_bench/web/static/js/report.js`,
  `src/jev_bench/web/static/js/charts.js`
- Modify: `tests/test_web_static.py` (add the page-module existence row)

**Interfaces:**
- Consumes:
  - Task 22: `api`, `h`, `clear`, `icon`, formatters, `readPref` / `writePref`, `initLayout`,
    `startJob`, `toastError`, `statusBadge`, `progressBar`, `emptyState`, `checklist`;
  - `GET /api/catalog`, `/api/generations`, `/api/runs`, `/api/runs/{id}`, `POST /api/runs`,
    `/api/runs/{id}/cancel`, `GET /api/compare`.
- Produces:
  - `report.js` → `renderReport(container, report, { pinned })`;
  - `charts.js` → `countsChart(canvas, question, labels)` and `destroyCharts()`.

Behaviour (spec §10):
- **Generation picker.** A checklist of generations with at least one email. The selection is remembered
  in `benchmark.generations`; the default is the newest generation.
- **Column cards.** One per catalog column. Each card has:
  - a model `<select>` showing "name — $in / $out per 1M" (remembered per column);
  - a mode toggle (`per_email` / `all_in_one`) on chat cards, remembered per column;
  - on the embeddings card, τ and the batch size;
  - Run and Cancel buttons;
  - a stats block: status, progress bar, elapsed, cost, errors, requests (+ splits), tokens,
    p50/p95; embeddings also show cache hits and cold cost.

  The stats describe the running run of that column on exactly the selected generation set, otherwise
  the latest one. Running runs are polled every second until they finish.
- **Comparison.** Shows `?runs=a,b` when pinned. Otherwise it shows the latest *completed* run per column
  on exactly the selected set. It refreshes when a run finishes or the selection changes.

- [ ] **Step 1: Extend the static contract test**

In `tests/test_web_static.py`, add:
```python
PAGE_MODULES = ("benchmark.js", "report.js", "charts.js")


def test_page_modules_exist() -> None:
    assert [name for name in PAGE_MODULES if not (STATIC_DIR / "js" / name).exists()] == []
```
Later tasks append to `PAGE_MODULES`.

Run: `uv run pytest tests/test_web_static.py -v`
Expected: `test_page_modules_exist` FAILS.

- [ ] **Step 2: Write `index.html`**

```html
<!doctype html>
<html lang="en" data-bs-theme="light">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="color-scheme" content="light dark">
  <title>Benchmark · jev-bench</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap">
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.8/dist/css/bootstrap.min.css" integrity="sha384-sRIl4kxILFvY47J16cr9ZwB07vP4J8+LH7qKQnuqkuIAvNWLzeN8tE5YBujZqJLB" crossorigin="anonymous">
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.13.1/font/bootstrap-icons.min.css" integrity="sha384-CK2SzKma4jA5H/MXDUU7i1TqZlCFaD4T01vtyDFvPlD97JQyS+IsSh1nI2EFbpyk" crossorigin="anonymous">
  <link rel="stylesheet" href="/css/app.css">
</head>
<body>
  <main class="container-fluid pb-5">
    <div class="d-flex flex-wrap align-items-center gap-2 mb-3">
      <h1 class="h4 mb-0 me-auto">Benchmark</h1>
      <div id="generation-picker"></div>
    </div>
    <div class="row g-3 mb-4" id="columns"></div>
    <section id="comparison" aria-live="polite"></section>
  </main>
  <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.8/dist/js/bootstrap.bundle.min.js" integrity="sha384-FKyoEForCGlyvwx9Hj09JcYn3nv7wiPVlz7YYwJrWVcXK/BmnVDxM+D2scQbITxI" crossorigin="anonymous"></script>
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4.5.1/dist/chart.umd.min.js" integrity="sha384-jb8JQMbMoBUzgWatfe6COACi2ljcDdZQ2OxczGA3bGNeWe+6DChMTBJemed7ZnvJ" crossorigin="anonymous"></script>
  <script type="module" src="/js/benchmark.js"></script>
</body>
</html>
```

- [ ] **Step 3: Write `charts.js`**

```js
/**
 * Chart.js helpers: grouped bar chart of argmax counts per option (one dataset per rater).
 * Exports: countsChart, destroyCharts.
 */

const PALETTE = ["#6f42c1", "#d63384", "#0d6efd", "#20c997", "#fd7e14", "#6c757d", "#198754", "#dc3545"];
const charts = new Set();

export function countsChart(canvas, question, labels) {
  const datasets = question.raters.map((stats, index) => ({
    label: labels[stats.rater] ?? stats.rater,
    data: question.options.map((option) => stats.argmax_counts[option] ?? 0),
    backgroundColor: PALETTE[index % PALETTE.length],
    borderRadius: 3,
  }));
  const chart = new Chart(canvas, {
    type: "bar",
    data: { labels: question.options, datasets },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { position: "bottom", labels: { boxWidth: 12 } } },
      scales: { x: { ticks: { autoSkip: false, maxRotation: 60 } }, y: { beginAtZero: true, ticks: { precision: 0 } } },
    },
  });
  charts.add(chart);
  return chart;
}

export function destroyCharts() {
  for (const chart of charts) chart.destroy();
  charts.clear();
}
```

- [ ] **Step 4: Write `report.js`**

```js
/**
 * Renders a ComparisonReport: a raters summary table, warnings, then one card per question with an
 * argmax-count chart, per-rater statistics and pairwise metrics (95% bootstrap CIs) plus Fleiss' kappa.
 * Exports: renderReport.
 */
import { countsChart, destroyCharts } from "./charts.js";
import { clear, h, icon } from "./dom.js";
import { duration, fixed, money, num, pct } from "./format.js";

export function renderReport(container, report, { pinned = false } = {}) {
  destroyCharts();
  const labels = Object.fromEntries(report.raters.map((rater) => [rater.id, raterLabel(rater)]));
  const cards = report.questions.map((question) => questionCard(question, labels));
  clear(
    container,
    header(report, pinned),
    summaryTable(report.raters, labels),
    report.warnings.map((warning) => h("div", { class: "alert alert-warning py-1 small" }, icon("exclamation-triangle"), ` ${warning}`)),
    h("div", { class: "row g-3" }, cards.map((card) => card.element)),
  );
  for (const card of cards) card.draw();
}

function raterLabel(rater) {
  if (!rater.run) return rater.label;
  const model = rater.run.model.split("/").pop();
  return `${rater.run.column} · ${model}${rater.run.mode === "all_in_one" ? " · all-in-one" : ""}`;
}

function header(report, pinned) {
  const runs = report.raters.filter((rater) => rater.run).map((rater) => rater.run);
  const generations = [...new Set(runs.flatMap((run) => run.generation_ids))];
  const explore = `/explorer.html?${new URLSearchParams({ generations: generations.join(","), runs: runs.map((run) => run.id).join(",") })}`;
  return h(
    "div",
    { class: "d-flex flex-wrap align-items-center gap-2 mb-2" },
    h("h2", { class: "h5 mb-0 me-auto" }, icon("graph-up"), " Comparison"),
    h("a", { class: "btn btn-sm btn-outline-primary", href: explore }, icon("search"), " Explore emails"),
    pinned ? h("a", { class: "btn btn-sm btn-outline-secondary", href: "/" }, icon("x-lg"), " Clear pinned runs") : null,
  );
}

function summaryTable(raters, labels) {
  const head = ["Rater", "Kind", "Emails", "Errors", "Duration", "Cost", "Cold cost", "Requests", "Latency p50 / p95"];
  const rows = raters.map((rater) => {
    const run = rater.run;
    const cells = run
      ? [num(run.n_done), num(run.n_errors), duration(run.duration_s), money(run.total_cost), run.kind === "embeddings" ? money(run.cold_cost) : "—", `${num(run.n_requests)}${run.n_splits ? ` (+${run.n_splits})` : ""}`, `${fixed(run.latency_p50_ms, 0)} / ${fixed(run.latency_p95_ms, 0)} ms`]
      : [num(rater.n_items), "—", "—", "—", "—", "—", "—"];
    return h("tr", {}, h("td", {}, labels[rater.id]), h("td", {}, rater.kind), cells.map((cell) => h("td", { class: "font-monospace small" }, cell)));
  });
  return h("div", { class: "table-responsive mb-3" }, table(head, rows));
}

function table(head, rows) {
  return h("table", { class: "table table-sm align-middle mb-2" }, h("thead", {}, h("tr", {}, head.map((cell) => h("th", { class: "small" }, cell)))), h("tbody", {}, rows));
}

function questionCard(question, labels) {
  const canvas = h("canvas", { role: "img", "aria-label": `${question.id} answer counts` });
  const element = h(
    "div",
    { class: "col-12 col-xl-6" },
    h(
      "div",
      { class: "card h-100 shadow-sm" },
      h(
        "div",
        { class: "card-header d-flex align-items-center gap-2" },
        h("code", { class: "fw-semibold" }, question.id),
        h("span", { class: "badge text-bg-light border" }, question.type),
        h("span", { class: "ms-auto small text-body-secondary" }, `Fleiss κ ${fixed(question.fleiss_kappa, 3)}`),
      ),
      h("div", { class: "card-body" }, h("div", { class: "chart-box mb-3" }, canvas), raterTable(question, labels), pairTable(question, labels)),
    ),
  );
  return { element, draw: () => question.raters.length && countsChart(canvas, question, labels) };
}

function raterTable(question, labels) {
  const extra = question.type === "score" ? "Mean level" : question.type === "noul" ? "Mean P(yes)" : null;
  const head = ["Rater", "n", "Entropy", "Confidence", ...(extra ? [extra] : [])];
  const rows = question.raters.map((stats) => {
    const value = question.type === "score" ? fixed(stats.mean_level, 2) : fixed(stats.mean.yes, 3);
    return h("tr", {}, h("td", { class: "small" }, labels[stats.rater] ?? stats.rater), h("td", {}, num(stats.n)), h("td", {}, fixed(stats.mean_entropy, 3)), h("td", {}, pct(stats.mean_confidence)), extra ? h("td", {}, value) : null);
  });
  return h("div", { class: "table-responsive" }, table(head, rows));
}

function interval(bounds, format) {
  return bounds ? h("small", { class: "text-body-secondary" }, ` [${format(bounds[0])}, ${format(bounds[1])}]`) : null;
}

function pairTable(question, labels) {
  if (!question.pairs.length) return h("p", { class: "small text-body-secondary mb-0" }, "No overlapping raters for this question.");
  const head = ["Pair", "n", "Agreement", "κ", "JSD", "r", "Brier"];
  const rows = question.pairs.map((pair) =>
    h(
      "tr",
      {},
      h("td", { class: "small" }, `${labels[pair.a] ?? pair.a} ↔ ${labels[pair.b] ?? pair.b}`),
      h("td", {}, num(pair.n)),
      h("td", {}, pct(pair.agreement), interval(pair.agreement_ci, pct)),
      h("td", {}, fixed(pair.kappa, 3), interval(pair.kappa_ci, (value) => fixed(value, 2))),
      h("td", {}, fixed(pair.jsd, 3)),
      h("td", {}, fixed(pair.pearson, 3)),
      h("td", {}, fixed(pair.brier, 3)),
    ),
  );
  return h("div", { class: "table-responsive" }, table(head, rows));
}
```

- [ ] **Step 5: Write `benchmark.js`**

```js
/**
 * Benchmark page: generation picker, one card per column (model, mode, run/cancel, live stats) and the
 * comparison of the latest completed run per column on exactly the selected generations (or ?runs=…).
 * Exports: none (page entry point).
 */
import { api } from "./api.js";
import { clear, h, icon } from "./dom.js";
import { duration, fixed, money, num, perMillion } from "./format.js";
import { initLayout, startJob, toastError } from "./layout.js";
import { renderReport } from "./report.js";
import { readPref, writePref } from "./storage.js";
import { checklist, emptyState, progressBar, statusBadge } from "./widgets.js";

const POLL_MS = 1000;
const state = { catalog: [], generations: [], runs: [], progress: new Map(), selected: [], pinned: [], timers: new Map() };

const sameSet = (values, set) => values.length === set.size && values.every((value) => set.has(value));
const selectedSet = () => new Set(state.selected);
const selectedMode = (columnId) => readPref(`benchmark.mode.${columnId}`, "per_email");

async function main() {
  await initLayout();
  state.pinned = (new URLSearchParams(location.search).get("runs") ?? "").split(",").filter(Boolean);
  const [catalog, generations, runs] = await Promise.all([api.catalog(), api.generations(), api.runs()]);
  state.catalog = catalog;
  state.generations = generations.map((view) => view.meta).filter((meta) => meta.done > 0);
  state.runs = runs.map((view) => view.meta);
  state.selected = initialSelection();
  renderPicker();
  renderColumns();
  for (const run of state.runs) if (run.status === "running") track(run.id);
  await refreshComparison();
}

function initialSelection() {
  const known = new Set(state.generations.map((generation) => generation.id));
  const stored = (readPref("benchmark.generations", []) ?? []).filter((id) => known.has(id));
  return stored.length ? stored : state.generations.slice(0, 1).map((generation) => generation.id);
}

function renderPicker() {
  const items = state.generations.map((generation) => ({ value: generation.id, text: `${generation.name} · ${generation.done} emails`, hint: `${generation.id} · ${generation.status}` }));
  clear(document.getElementById("generation-picker"), checklist({ label: "Generations", items, selected: state.selected, onChange: onSelection }));
}

function onSelection(selected) {
  state.selected = selected;
  state.pinned = [];
  writePref("benchmark.generations", selected);
  for (const column of state.catalog) refreshStats(column.id);
  refreshComparison().catch(toastError);
}

function renderColumns() {
  const container = clear(document.getElementById("columns"));
  if (!state.catalog.length) {
    container.append(emptyState("No columns are configured in config/benchmark.toml."));
    return;
  }
  for (const column of state.catalog) container.append(h("div", { class: "col-12 col-md-6 col-xxl-3" }, columnCard(column)));
}

function modelLabel(model) {
  return `${model.name} — ${perMillion(model.prompt_price_per_m)} / ${perMillion(model.completion_price_per_m)} per 1M`;
}

function columnCard(column) {
  const chosen = readPref(`benchmark.model.${column.id}`, column.default_model);
  const select = h(
    "select",
    { class: "form-select form-select-sm", "aria-label": `${column.title} model`, onchange: (event) => writePref(`benchmark.model.${column.id}`, event.target.value) },
    column.models.map((model) => h("option", { value: model.id, selected: model.id === chosen }, modelLabel(model))),
  );
  const running = currentRun(column.id)?.status === "running";
  const run = h("button", { class: "btn btn-primary btn-sm", type: "button", onclick: () => launch(column, select.value) }, icon("play-fill"), " Run");
  const cancel = h("button", { class: "btn btn-outline-danger btn-sm", type: "button", id: `cancel-${column.id}`, disabled: !running, onclick: () => cancelColumn(column.id) }, icon("stop-fill"), " Cancel");
  return h(
    "div",
    { class: "card h-100 column-card shadow-sm" },
    h("div", { class: "card-header d-flex align-items-center" }, h("span", { class: "fw-semibold me-auto" }, column.title), h("span", { class: "badge text-bg-light border" }, column.kind)),
    h(
      "div",
      { class: "card-body d-flex flex-column gap-2" },
      column.error ? h("div", { class: "alert alert-warning py-1 px-2 small mb-0" }, icon("exclamation-triangle"), ` Catalog unavailable: ${column.error}`) : null,
      h("div", {}, h("label", { class: "form-label small mb-1" }, "Model"), select, columnExtras(column)),
      h("div", { class: "d-flex gap-2" }, run, cancel),
      h("div", { id: `stats-${column.id}` }, statsFor(column.id)),
    ),
  );
}

function columnExtras(column) {
  if (column.kind === "chat") return modeToggle(column);
  if (column.kind === "embeddings") {
    return h("div", { class: "small text-body-secondary mt-2" }, icon("thermometer-half"), ` τ = ${column.embedding_temperature} · ${column.emails_per_request} emails per request`);
  }
  return null;
}

function modeToggle(column) {
  const current = selectedMode(column.id);
  const option = (value, text) => {
    const id = `mode-${column.id}-${value}`;
    return [
      h("input", { type: "radio", class: "btn-check", name: `mode-${column.id}`, id, value, checked: current === value, onchange: () => writePref(`benchmark.mode.${column.id}`, value) }),
      h("label", { class: "btn btn-outline-secondary btn-sm", for: id }, text),
    ];
  };
  return h("div", { class: "btn-group mt-2 w-100", role: "group", "aria-label": "Request mode" }, option("per_email", "Per email"), option("all_in_one", "All in one"));
}

function currentRun(columnId) {
  const selected = selectedSet();
  const matching = state.runs.filter((run) => run.column === columnId && sameSet(run.generation_ids, selected));
  return matching.find((run) => run.status === "running") ?? matching[0];
}

function statRows(run, live) {
  const rows = [
    ["Status", statusBadge(run.status)],
    ["Elapsed", duration(live ? live.elapsed_s : run.duration_s)],
    ["Cost", money(live ? live.cost : run.total_cost)],
    ["Errors", num(live ? live.errors : run.n_errors)],
    ["Requests", `${num(run.n_requests)}${run.n_splits ? ` (+${run.n_splits} split)` : ""}`],
    ["Tokens in / out", `${num(run.input_tokens)} / ${num(run.output_tokens)}`],
    ["Latency p50 / p95", `${fixed(run.latency_p50_ms, 0)} / ${fixed(run.latency_p95_ms, 0)} ms`],
  ];
  if (run.kind === "embeddings") {
    rows.push(["Cache hits", `${num(run.cache_hits)} / ${num(run.cache_hits + run.cache_misses)}`], ["Cold cost", money(run.cold_cost)]);
  }
  return rows;
}

function statsFor(columnId) {
  const run = currentRun(columnId);
  if (!run) return h("p", { class: "small text-body-secondary mb-0" }, "No run on the selected generations yet.");
  const live = run.status === "running" ? state.progress.get(run.id) : null;
  const done = live ? live.done : run.n_done;
  const items = statRows(run, live).flatMap(([label, value]) => [h("dt", { class: "col-6 stat-label" }, label), h("dd", { class: "col-6 stat-value mb-1" }, value)]);
  return h(
    "div",
    {},
    progressBar(done, run.n_emails, { animated: run.status === "running" }),
    h("dl", { class: "row small mb-0 mt-2" }, items),
    h("div", { class: "small text-body-secondary text-truncate", title: run.id }, `${run.model} · ${run.mode} · ${run.id}`),
    run.error ? h("div", { class: "alert alert-danger py-1 px-2 small mt-2 mb-0" }, run.error) : null,
  );
}

function refreshStats(columnId) {
  const target = document.getElementById(`stats-${columnId}`);
  if (target) clear(target, statsFor(columnId));
  const cancel = document.getElementById(`cancel-${columnId}`);
  if (cancel) cancel.disabled = currentRun(columnId)?.status !== "running";
}

async function launch(column, model) {
  if (!state.selected.length) {
    toastError("Select at least one generation first.");
    return;
  }
  const body = { column: column.id, model, generation_ids: state.selected };
  if (column.kind === "chat") body.mode = selectedMode(column.id);
  const view = await startJob(() => api.createRun(body));
  if (!view) return;
  upsertRun(view.meta, view.progress);
  track(view.meta.id);
}

async function cancelColumn(columnId) {
  const run = currentRun(columnId);
  if (run?.status === "running") await api.cancelRun(run.id).catch(toastError);
}

function upsertRun(meta, progress) {
  state.runs = [meta, ...state.runs.filter((run) => run.id !== meta.id)].sort((a, b) => b.id.localeCompare(a.id));
  if (progress) state.progress.set(meta.id, progress);
  else state.progress.delete(meta.id);
  refreshStats(meta.column);
}

function track(runId) {
  if (state.timers.has(runId)) return;
  state.timers.set(runId, setInterval(() => poll(runId), POLL_MS));
}

function stopTracking(runId) {
  clearInterval(state.timers.get(runId));
  state.timers.delete(runId);
}

async function poll(runId) {
  try {
    const view = await api.run(runId);
    upsertRun(view.meta, view.progress);
    if (view.meta.status === "running") return;
    stopTracking(runId);
    await refreshComparison();
  } catch (error) {
    stopTracking(runId);
    toastError(error);
  }
}

function latestCompletedRuns() {
  const selected = selectedSet();
  return state.catalog
    .map((column) => state.runs.find((run) => run.column === column.id && run.status === "completed" && sameSet(run.generation_ids, selected)))
    .filter(Boolean)
    .map((run) => run.id);
}

async function refreshComparison() {
  const container = document.getElementById("comparison");
  const runIds = state.pinned.length ? state.pinned : latestCompletedRuns();
  if (!runIds.length) {
    clear(container, emptyState("Run at least one column on the selected generations to see the comparison.", "bar-chart"));
    return;
  }
  clear(container, h("p", { class: "small text-body-secondary" }, "Loading comparison…"));
  renderReport(container, await api.compare(runIds), { pinned: state.pinned.length > 0 });
}

main().catch(toastError);
```

- [ ] **Step 6: Run the static contract test**

Run: `uv run pytest tests/test_web_static.py -v`
Expected: all PASS (index.html now has pinned SRI assets and a module script; no forbidden sinks).

- [ ] **Step 7: Manual browser check**

Run: `uv run python -c "from jev_bench.web.app import create_default_app"` (the import must succeed).
The full browser smoke is in Task 25.

- [ ] **Step 8: Commit**

```bash
git add src/jev_bench/web/static tests/test_web_static.py
git commit -m "feat(ui): benchmark page with column cards, live stats and comparison report

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```
