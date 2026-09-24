### Task 24: Generations, Explorer and Runs pages

**Files:**
- Create in `src/jev_bench/web/static/`: `generations.html`, `explorer.html`, `runs.html`,
  `js/generations.js`, `js/explorer.js`, `js/runs.js`, `js/distribution.js`
- Modify: `tests/test_web_static.py` (extend `PAGE_MODULES`)

**Interfaces:**
- Consumes:
  - Task 22 modules;
  - `GET/POST /api/generations`, `POST /api/generations/{id}/cancel`, `GET /api/runs`,
    `GET /api/emails`, `GET /api/emails/{id}`, `PUT /api/labels/{id}`.
- Produces:
  - `distribution.js` → `probabilityCell(probability, { similarity, highlight }) -> <td>` (a bar plus
    the value; the tooltip also shows the cosine for embedding runs);
  - three page entry scripts.

Behaviour (spec §10):
- **Generations.**
  - A form with name, count (1–2000), optional seed, and optional models (one per line; empty means the
    config mix).
  - A table of every generation with status, progress, errors, cost, duration, trait mismatches, models,
    seed, "Explore" and "Cancel" (while running).
  - It polls every second while anything is running.
- **Explorer.**
  - Checklists for generations and completed runs (only runs overlapping the selected generations). The
    selection is synced to `?generations=…&runs=…`.
  - Filters: text in subject/sender, generator model, reference answer (`question=option`), trait
    (`name=value`), and the displayed question. A "most disagreement first" switch is on by default.
  - A table of sent_at, from, subject, generator, the reference answer, one column per run (green when
    it matches the reference, red otherwise), and the disagreement index.
  - Clicking a row (or pressing Enter on it) opens an offcanvas panel with:
    - the headers, and the body as text;
    - per question: the options against the reference ✓ and each run's probability bars;
    - a human-label select, with "Save labels" (`PUT`).
- **Runs.**
  - A table of every run with a checkbox.
  - "Compare selected" → `/?runs=…`, and "Explore selected" → `/explorer.html?runs=…&generations=…`.
  - It polls every 2 seconds while anything is running.

- [ ] **Step 1: Extend the static contract test**

In `tests/test_web_static.py`, extend the tuple:
```python
PAGE_MODULES = ("benchmark.js", "report.js", "charts.js", "generations.js", "explorer.js", "runs.js", "distribution.js")
```
Run: `uv run pytest tests/test_web_static.py -v`
Expected: `test_page_modules_exist` FAILS.

- [ ] **Step 2: Write the three HTML pages**

Every page uses the same `<head>` as `index.html` (Task 23) except the `<title>`, and loads only the
Bootstrap bundle script (no Chart.js), followed by its module.

`generations.html`:
```html
<!doctype html>
<html lang="en" data-bs-theme="light">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="color-scheme" content="light dark">
  <title>Generations · jev-bench</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap">
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.8/dist/css/bootstrap.min.css" integrity="sha384-sRIl4kxILFvY47J16cr9ZwB07vP4J8+LH7qKQnuqkuIAvNWLzeN8tE5YBujZqJLB" crossorigin="anonymous">
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.13.1/font/bootstrap-icons.min.css" integrity="sha384-CK2SzKma4jA5H/MXDUU7i1TqZlCFaD4T01vtyDFvPlD97JQyS+IsSh1nI2EFbpyk" crossorigin="anonymous">
  <link rel="stylesheet" href="/css/app.css">
</head>
<body>
  <main class="container-fluid pb-5">
    <h1 class="h4 mb-3">Generations</h1>
    <div class="card shadow-sm mb-4">
      <div class="card-header fw-semibold">New generation</div>
      <div class="card-body" id="new-generation"></div>
    </div>
    <div id="generation-list" aria-live="polite"></div>
  </main>
  <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.8/dist/js/bootstrap.bundle.min.js" integrity="sha384-FKyoEForCGlyvwx9Hj09JcYn3nv7wiPVlz7YYwJrWVcXK/BmnVDxM+D2scQbITxI" crossorigin="anonymous"></script>
  <script type="module" src="/js/generations.js"></script>
</body>
</html>
```

`explorer.html`:
```html
<!doctype html>
<html lang="en" data-bs-theme="light">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="color-scheme" content="light dark">
  <title>Explorer · jev-bench</title>
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
      <h1 class="h4 mb-0 me-auto">Explorer</h1>
      <div class="d-flex flex-wrap gap-2" id="pickers"></div>
    </div>
    <div class="row g-2 mb-3" id="filters"></div>
    <div id="email-table" aria-live="polite"></div>
  </main>
  <div class="offcanvas offcanvas-end" tabindex="-1" id="email-panel" aria-labelledby="email-panel-title" style="width: min(900px, 100vw)">
    <div class="offcanvas-header">
      <h5 class="offcanvas-title text-truncate" id="email-panel-title">Email</h5>
      <button type="button" class="btn-close" data-bs-dismiss="offcanvas" aria-label="Close"></button>
    </div>
    <div class="offcanvas-body" id="email-panel-body"></div>
  </div>
  <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.8/dist/js/bootstrap.bundle.min.js" integrity="sha384-FKyoEForCGlyvwx9Hj09JcYn3nv7wiPVlz7YYwJrWVcXK/BmnVDxM+D2scQbITxI" crossorigin="anonymous"></script>
  <script type="module" src="/js/explorer.js"></script>
</body>
</html>
```

`runs.html`:
```html
<!doctype html>
<html lang="en" data-bs-theme="light">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="color-scheme" content="light dark">
  <title>Runs · jev-bench</title>
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
      <h1 class="h4 mb-0 me-auto">Runs</h1>
      <button class="btn btn-sm btn-primary" id="compare" type="button" disabled><i class="bi bi-graph-up" aria-hidden="true"></i> Compare selected</button>
      <button class="btn btn-sm btn-outline-primary" id="explore" type="button" disabled><i class="bi bi-search" aria-hidden="true"></i> Explore selected</button>
    </div>
    <div id="run-list" aria-live="polite"></div>
  </main>
  <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.8/dist/js/bootstrap.bundle.min.js" integrity="sha384-FKyoEForCGlyvwx9Hj09JcYn3nv7wiPVlz7YYwJrWVcXK/BmnVDxM+D2scQbITxI" crossorigin="anonymous"></script>
  <script type="module" src="/js/runs.js"></script>
</body>
</html>
```

- [ ] **Step 3: Write `distribution.js`**

```js
/**
 * Probability visualisation for the Explorer detail panel: a table cell with a bar and the value;
 * the tooltip adds the cosine similarity for embedding runs.
 * Exports: probabilityCell.
 */
import { h } from "./dom.js";
import { fixed } from "./format.js";

export function probabilityCell(probability, { similarity = null, highlight = false } = {}) {
  const value = probability ?? 0;
  const title = similarity === null ? `p = ${fixed(value, 3)}` : `p = ${fixed(value, 3)} · cos = ${fixed(similarity, 3)}`;
  const bar = h("div", { class: "prob-track flex-grow-1" }, h("div", { class: "prob-bar", style: `width: ${Math.round(value * 100)}%` }));
  return h("td", { class: highlight ? "fw-semibold" : "", title }, h("div", { class: "d-flex align-items-center gap-2" }, bar, h("span", { class: "small font-monospace" }, fixed(value, 2))));
}
```

- [ ] **Step 4: Write `generations.js`**

```js
/**
 * Generations page: create a generation (name, count, seed, models) and watch every generation's progress.
 * Exports: none (page entry point).
 */
import { api } from "./api.js";
import { clear, h, icon } from "./dom.js";
import { duration, money, num, when } from "./format.js";
import { initLayout, startJob, toastError, toastSuccess } from "./layout.js";
import { emptyState, progressBar, statusBadge } from "./widgets.js";

const POLL_MS = 1000;
let timer = null;

async function main() {
  await initLayout();
  clear(document.getElementById("new-generation"), generationForm());
  await refresh();
}

function field(label, input, hint = null) {
  return h("div", { class: "col-12 col-md-6 col-xl-3" }, h("label", { class: "form-label small", for: input.id }, label), input, hint ? h("div", { class: "form-text" }, hint) : null);
}

function generationForm() {
  const name = h("input", { class: "form-control", id: "gen-name", value: "generation", maxlength: 80, required: true });
  const count = h("input", { class: "form-control", id: "gen-count", type: "number", min: 1, max: 2000, value: 50, required: true });
  const seed = h("input", { class: "form-control", id: "gen-seed", type: "number", min: 0, placeholder: "random" });
  const models = h("textarea", { class: "form-control font-monospace", id: "gen-models", rows: 1, placeholder: "one model id per line" });
  const submit = (event) => {
    event.preventDefault();
    create({ name, count, seed, models });
  };
  return h(
    "form",
    { class: "row g-3 align-items-end", onsubmit: submit },
    field("Name", name),
    field("Emails", count, "1–2000"),
    field("Seed", seed, "Empty = random; recorded in the generation"),
    field("Models", models, "Empty = the mix from config/generation.toml"),
    h("div", { class: "col-12" }, h("button", { class: "btn btn-primary", type: "submit" }, icon("magic"), " Generate")),
  );
}

async function create({ name, count, seed, models }) {
  const body = { name: name.value.trim() || "generation", count: Number(count.value) };
  if (seed.value !== "") body.seed = Number(seed.value);
  const list = models.value.split("\n").map((line) => line.trim()).filter(Boolean);
  if (list.length) body.models = list;
  const view = await startJob(() => api.createGeneration(body));
  if (!view) return;
  toastSuccess(`Generation ${view.meta.id} started`);
  await refresh();
}

async function refresh() {
  try {
    const views = await api.generations();
    render(views);
    schedule(views.some((view) => view.meta.status === "running"));
  } catch (error) {
    schedule(false);
    toastError(error);
  }
}

function schedule(active) {
  clearTimeout(timer);
  timer = active ? setTimeout(refresh, POLL_MS) : null;
}

function render(views) {
  const target = document.getElementById("generation-list");
  if (!views.length) {
    clear(target, emptyState("No generations yet. Create one above or run `jev-bench generate`.", "envelope-paper"));
    return;
  }
  const head = ["Name", "Created", "Status", "Progress", "Errors", "Cost", "Duration", "Mismatches", "Models", "Seed", ""];
  const table = h("table", { class: "table table-sm align-middle table-hover" }, h("thead", {}, h("tr", {}, head.map((cell) => h("th", { class: "small" }, cell)))), h("tbody", {}, views.map(row)));
  clear(target, h("div", { class: "table-responsive" }, table));
}

function row({ meta, progress }) {
  const live = meta.status === "running" ? progress : null;
  const actions = [h("a", { class: "btn btn-sm btn-outline-primary", href: `/explorer.html?generations=${encodeURIComponent(meta.id)}` }, icon("search"), " Explore")];
  if (meta.status === "running") actions.push(h("button", { class: "btn btn-sm btn-outline-danger", type: "button", onclick: () => cancel(meta.id) }, icon("stop-fill"), " Cancel"));
  return h(
    "tr",
    {},
    h("td", {}, h("div", { class: "fw-semibold" }, meta.name), h("div", { class: "small text-body-secondary font-monospace" }, meta.id)),
    h("td", { class: "small" }, when(meta.created_at)),
    h("td", {}, statusBadge(meta.status), meta.error ? h("div", { class: "small text-danger" }, meta.error) : null),
    h("td", { class: "w-25" }, progressBar(live ? live.done : meta.done, meta.requested, { animated: Boolean(live) })),
    h("td", {}, num(live ? live.errors : meta.errors)),
    h("td", { class: "font-monospace small" }, money(live ? live.cost : meta.total_cost)),
    h("td", { class: "small" }, duration(live ? live.elapsed_s : meta.duration_s)),
    h("td", {}, num(meta.trait_mismatches)),
    h("td", { class: "small" }, meta.models.join(", ")),
    h("td", { class: "font-monospace small" }, String(meta.seed)),
    h("td", { class: "text-end" }, h("div", { class: "d-flex gap-1 justify-content-end" }, actions)),
  );
}

async function cancel(id) {
  await api.cancelGeneration(id).catch(toastError);
  await refresh();
}

main().catch(toastError);
```

- [ ] **Step 5: Write `runs.js`**

```js
/**
 * Runs page: every persisted run with its totals; select runs to compare (Benchmark ?runs=) or explore.
 * Exports: none (page entry point).
 */
import { api } from "./api.js";
import { clear, h } from "./dom.js";
import { duration, money, num, shortModel, when } from "./format.js";
import { initLayout, toastError } from "./layout.js";
import { emptyState, statusBadge } from "./widgets.js";

const POLL_MS = 2000;
const selected = new Set();
let runs = [];
let timer = null;

async function main() {
  await initLayout();
  document.getElementById("compare").addEventListener("click", () => go("/"));
  document.getElementById("explore").addEventListener("click", () => go("/explorer.html"));
  await refresh();
}

function go(page) {
  const chosen = runs.filter((run) => selected.has(run.id));
  const params = new URLSearchParams({ runs: chosen.map((run) => run.id).join(",") });
  if (page !== "/") params.set("generations", [...new Set(chosen.flatMap((run) => run.generation_ids))].join(","));
  location.href = `${page}?${params}`;
}

async function refresh() {
  try {
    runs = (await api.runs()).map((view) => view.meta);
    render();
    clearTimeout(timer);
    timer = runs.some((run) => run.status === "running") ? setTimeout(refresh, POLL_MS) : null;
  } catch (error) {
    toastError(error);
  }
}

function updateButtons() {
  for (const id of ["compare", "explore"]) document.getElementById(id).disabled = selected.size === 0;
}

function toggle(runId, checked) {
  if (checked) selected.add(runId);
  else selected.delete(runId);
  updateButtons();
}

function render() {
  const target = document.getElementById("run-list");
  updateButtons();
  if (!runs.length) {
    clear(target, emptyState("No runs yet. Start one on the Benchmark page.", "clock-history"));
    return;
  }
  const head = ["", "Created", "Column", "Model", "Mode", "Generations", "Status", "Emails", "Errors", "Duration", "Cost", "Requests"];
  const table = h("table", { class: "table table-sm align-middle table-hover" }, h("thead", {}, h("tr", {}, head.map((cell) => h("th", { class: "small" }, cell)))), h("tbody", {}, runs.map(row)));
  clear(target, h("div", { class: "table-responsive" }, table));
}

function row(run) {
  const box = h("input", { class: "form-check-input", type: "checkbox", checked: selected.has(run.id), "aria-label": `select ${run.id}`, onchange: (event) => toggle(run.id, event.target.checked) });
  return h(
    "tr",
    {},
    h("td", {}, box),
    h("td", { class: "small" }, when(run.created_at)),
    h("td", {}, run.column),
    h("td", { class: "small", title: run.model }, shortModel(run.model)),
    h("td", { class: "small" }, run.mode),
    h("td", { class: "small font-monospace", title: run.generation_ids.join("\n") }, `${run.generation_ids.length} gen.`),
    h("td", {}, statusBadge(run.status), run.error ? h("div", { class: "small text-danger" }, run.error) : null),
    h("td", {}, `${num(run.n_done)}/${num(run.n_emails)}`),
    h("td", {}, num(run.n_errors)),
    h("td", { class: "small" }, duration(run.duration_s)),
    h("td", { class: "font-monospace small" }, money(run.total_cost)),
    h("td", {}, `${num(run.n_requests)}${run.n_splits ? ` (+${run.n_splits})` : ""}`),
  );
}

main().catch(toastError);
```

- [ ] **Step 6: Write `explorer.js`**

```js
/**
 * Explorer page: browse the emails of selected generations, compare each run's top answer with the
 * generator reference, sort by disagreement, and open an email to see every distribution and edit
 * human labels.
 * Exports: none (page entry point).
 */
import { api } from "./api.js";
import { probabilityCell } from "./distribution.js";
import { clear, h, icon } from "./dom.js";
import { fixed, shortModel, when } from "./format.js";
import { initLayout, toastError, toastSuccess } from "./layout.js";
import { checklist, emptyState } from "./widgets.js";

const EMPTY_FILTERS = { text: "", generator: "", reference: "", trait: "" };
const state = {
  generations: [],
  runs: [],
  selectedGenerations: [],
  selectedRuns: [],
  questions: null,
  question: null,
  rows: [],
  filters: { ...EMPTY_FILTERS },
  byDisagreement: true,
};

const listParam = (params, name) => (params.get(name) ?? "").split(",").filter(Boolean);
const matchesPair = (values, pair) => {
  const [key, value] = pair.split("=");
  return values[key] === value;
};
const runLabel = (run) => `${run.column} · ${shortModel(run.model)}${run.mode === "all_in_one" ? " · all-in-one" : ""}`;

async function main() {
  await initLayout();
  const params = new URLSearchParams(location.search);
  const [generations, runs] = await Promise.all([api.generations(), api.runs()]);
  state.generations = generations.map((view) => view.meta);
  state.runs = runs.map((view) => view.meta).filter((run) => run.status === "completed");
  const requested = listParam(params, "generations");
  state.selectedGenerations = requested.length ? requested : state.generations.slice(0, 1).map((generation) => generation.id);
  state.selectedRuns = listParam(params, "runs");
  renderPickers();
  await loadRows();
}

function renderPickers() {
  const chosen = new Set(state.selectedGenerations);
  const generationItems = state.generations.map((generation) => ({ value: generation.id, text: `${generation.name} · ${generation.done} emails`, hint: generation.id }));
  const runItems = state.runs.filter((run) => run.generation_ids.some((id) => chosen.has(id))).map((run) => ({ value: run.id, text: runLabel(run), hint: run.id }));
  clear(
    document.getElementById("pickers"),
    checklist({ label: "Generations", items: generationItems, selected: state.selectedGenerations, onChange: onGenerations }),
    checklist({ label: "Runs", items: runItems, selected: state.selectedRuns, onChange: onRuns }),
  );
}

function onGenerations(values) {
  state.selectedGenerations = values;
  syncUrl();
  renderPickers();
  loadRows().catch(toastError);
}

function onRuns(values) {
  state.selectedRuns = values;
  syncUrl();
  loadRows().catch(toastError);
}

function syncUrl() {
  const params = new URLSearchParams();
  if (state.selectedGenerations.length) params.set("generations", state.selectedGenerations.join(","));
  if (state.selectedRuns.length) params.set("runs", state.selectedRuns.join(","));
  history.replaceState(null, "", `${location.pathname}?${params}`);
}

function activeRuns() {
  const known = new Set(state.runs.map((run) => run.id));
  return state.selectedRuns.filter((id) => known.has(id));
}

async function loadRows() {
  const target = document.getElementById("email-table");
  if (!state.selectedGenerations.length) {
    clear(document.getElementById("filters"));
    clear(target, emptyState("Select at least one generation.", "collection"));
    return;
  }
  clear(target, h("p", { class: "small text-body-secondary" }, "Loading…"));
  const list = await api.emails(state.selectedGenerations, activeRuns());
  state.questions = list.questions.questions;
  state.question = state.questions[0]?.id ?? null;
  state.rows = list.rows;
  state.filters = { ...EMPTY_FILTERS };
  renderFilters();
  renderTable();
}

function filterSelect(label, key, options) {
  const onchange = (event) => {
    state.filters[key] = event.target.value;
    renderTable();
  };
  return h("select", { class: "form-select form-select-sm", "aria-label": label, onchange }, h("option", { value: "" }, `${label}: any`), options);
}

function renderFilters() {
  const generators = [...new Set(state.rows.map((row) => row.generator_model))].sort();
  const traits = [...new Set(state.rows.flatMap((row) => Object.entries(row.traits).map(([name, value]) => `${name}=${value}`)))].sort();
  const references = state.questions.map((question) => h("optgroup", { label: question.id }, Object.keys(question.options).map((option) => h("option", { value: `${question.id}=${option}` }, option))));
  const questionPicker = h(
    "select",
    { class: "form-select form-select-sm", "aria-label": "Displayed question", onchange: (event) => { state.question = event.target.value; renderTable(); } },
    state.questions.map((question) => h("option", { value: question.id, selected: question.id === state.question }, `Question: ${question.id}`)),
  );
  const search = h("input", { class: "form-control form-control-sm", type: "search", placeholder: "Search subject or sender…", "aria-label": "Search", oninput: (event) => { state.filters.text = event.target.value.toLowerCase(); renderTable(); } });
  const sort = h(
    "div",
    { class: "form-check form-switch mb-0" },
    h("input", { class: "form-check-input", type: "checkbox", id: "sort-disagreement", checked: state.byDisagreement, onchange: (event) => { state.byDisagreement = event.target.checked; renderTable(); } }),
    h("label", { class: "form-check-label small", for: "sort-disagreement" }, "Most disagreement first"),
  );
  clear(
    document.getElementById("filters"),
    h("div", { class: "col-12 col-lg-3" }, search),
    h("div", { class: "col-6 col-lg-2" }, questionPicker),
    h("div", { class: "col-6 col-lg-2" }, filterSelect("Generator", "generator", generators.map((model) => h("option", { value: model }, model)))),
    h("div", { class: "col-6 col-lg-2" }, filterSelect("Reference", "reference", references)),
    h("div", { class: "col-6 col-lg-2" }, filterSelect("Trait", "trait", traits.map((trait) => h("option", { value: trait }, trait)))),
    h("div", { class: "col-12 col-lg-1 d-flex align-items-center" }, sort),
  );
}

function visibleRows() {
  const { text, generator, reference, trait } = state.filters;
  const rows = state.rows.filter(
    (row) =>
      (!text || `${row.subject} ${row.sender}`.toLowerCase().includes(text)) &&
      (!generator || row.generator_model === generator) &&
      (!reference || matchesPair(row.reference, reference)) &&
      (!trait || matchesPair(row.traits, trait)),
  );
  if (state.byDisagreement) rows.sort((a, b) => (b.disagreement ?? -1) - (a.disagreement ?? -1));
  return rows;
}

function runColumns() {
  const present = new Set(state.rows.flatMap((row) => Object.keys(row.top)));
  return activeRuns().filter((id) => present.has(id));
}

function renderTable() {
  const target = document.getElementById("email-table");
  const rows = visibleRows();
  if (!rows.length) {
    clear(target, emptyState("No emails match the filters.", "search"));
    return;
  }
  const runs = runColumns();
  const labels = Object.fromEntries(state.runs.map((run) => [run.id, runLabel(run)]));
  const head = ["Sent", "From", "Subject", "Generator", `Reference: ${state.question}`, ...runs.map((id) => labels[id] ?? id), "Disagreement"];
  const table = h(
    "table",
    { class: "table table-sm table-hover align-middle table-sticky" },
    h("thead", {}, h("tr", {}, head.map((cell) => h("th", { class: "small" }, cell)))),
    h("tbody", {}, rows.map((row) => emailRow(row, runs))),
  );
  clear(target, h("p", { class: "small text-body-secondary" }, `${rows.length} of ${state.rows.length} emails`), h("div", { class: "table-responsive", style: "max-height: 70vh" }, table));
}

function emailRow(row, runs) {
  const reference = row.reference[state.question];
  const cells = runs.map((id) => {
    const answer = row.top[id]?.[state.question];
    const style = answer === undefined ? "" : answer === reference ? "cell-match" : "cell-mismatch";
    return h("td", { class: `small ${style}` }, answer ?? "—");
  });
  const labelled = Object.keys(row.human).length ? h("span", { class: "badge text-bg-info ms-1", title: "Has human labels" }, icon("person-check")) : null;
  const open = () => openEmail(row.id);
  return h(
    "tr",
    { class: "clickable", tabindex: 0, onclick: open, onkeydown: (event) => event.key === "Enter" && open() },
    h("td", { class: "small text-nowrap" }, when(row.sent_at)),
    h("td", { class: "small text-truncate", style: "max-width: 14rem", title: row.sender }, row.sender),
    h("td", { class: "text-truncate", style: "max-width: 22rem", title: row.subject }, row.subject),
    h("td", { class: "small" }, shortModel(row.generator_model)),
    h("td", { class: "small fw-semibold" }, reference ?? "—", labelled),
    cells,
    h("td", { class: "font-monospace small" }, fixed(row.disagreement, 3)),
  );
}

async function openEmail(emailId) {
  const body = document.getElementById("email-panel-body");
  clear(body, h("p", { class: "small text-body-secondary" }, "Loading…"));
  bootstrap.Offcanvas.getOrCreateInstance(document.getElementById("email-panel")).show();
  try {
    const detail = await api.email(emailId, runColumns());
    clear(document.getElementById("email-panel-title"), detail.email.subject);
    clear(body, emailHeader(detail.email), h("div", { class: "email-body mb-3" }, detail.email.body), labellingForm(detail));
  } catch (error) {
    clear(body, h("div", { class: "alert alert-danger" }, error instanceof Error ? error.message : String(error)));
  }
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

function labellingForm(detail) {
  const choices = { ...detail.human };
  const runIds = Object.keys(detail.predictions);
  const sections = detail.questions.questions.map((question) => questionSection(question, detail, runIds, choices));
  const save = h("button", { class: "btn btn-primary", type: "button", onclick: () => saveLabels(detail.email.id, choices, detail.human) }, icon("save"), " Save labels");
  return h("div", {}, sections, h("div", { class: "d-flex justify-content-end mt-2" }, save));
}

function humanSelect(question, current, choices) {
  const onchange = (event) => {
    choices[question.id] = event.target.value || null;
  };
  const options = Object.entries(question.options).map(([option, text]) => h("option", { value: option, selected: current === option, title: text }, option));
  return h("select", { class: "form-select form-select-sm", "aria-label": `Human label for ${question.id}`, onchange }, h("option", { value: "" }, "— not labelled —"), options);
}

function questionSection(question, detail, runIds, choices) {
  const reference = detail.email.reference_answers[question.id];
  const head = ["Option", "Ref", ...runIds.map((id) => detail.run_labels[id] ?? id)];
  const rows = Object.entries(question.options).map(([option, text]) =>
    h("tr", {}, h("td", { class: "small", title: text }, option), h("td", { class: "text-center" }, reference === option ? icon("check-lg") : ""), runIds.map((id) => runCell(detail.predictions[id], question.id, option))),
  );
  const table = h("table", { class: "table table-sm mb-2" }, h("thead", {}, h("tr", {}, head.map((cell) => h("th", { class: "small" }, cell)))), h("tbody", {}, rows));
  return h(
    "div",
    { class: "card mb-2" },
    h("div", { class: "card-header py-1 d-flex align-items-center gap-2" }, h("code", {}, question.id), h("span", { class: "small text-body-secondary text-truncate" }, question.instructions)),
    h(
      "div",
      { class: "card-body py-2" },
      h("div", { class: "table-responsive" }, table),
      h("div", { class: "d-flex align-items-center gap-2" }, h("span", { class: "small text-nowrap" }, icon("person"), " Human"), humanSelect(question, detail.human[question.id], choices)),
    ),
  );
}

function runCell(prediction, questionId, option) {
  if (!prediction) return h("td", { class: "small text-body-secondary" }, "—");
  if (prediction.error) return h("td", { class: "small text-danger", title: prediction.error }, "error");
  const distribution = prediction.answers?.[questionId];
  if (!distribution) return h("td", { class: "small text-body-secondary" }, "—");
  const top = Object.entries(distribution).reduce((best, entry) => (entry[1] > best[1] ? entry : best))[0];
  const similarity = prediction.similarities?.[questionId]?.[option] ?? null;
  return probabilityCell(distribution[option], { similarity, highlight: top === option });
}

async function saveLabels(emailId, choices, original) {
  const keys = new Set([...Object.keys(original), ...Object.keys(choices)]);
  const answers = Object.fromEntries([...keys].map((key) => [key, choices[key] ?? null]));
  try {
    const saved = await api.putLabel(emailId, answers);
    const row = state.rows.find((item) => item.id === emailId);
    if (row) row.human = saved;
    renderTable();
    toastSuccess("Labels saved");
  } catch (error) {
    toastError(error);
  }
}

main().catch(toastError);
```

- [ ] **Step 7: Run the static contract test**

Run: `uv run pytest tests/test_web_static.py -v`
Expected: all PASS (three new pages parsed; no sinks; all imports resolve).

- [ ] **Step 8: Commit**

```bash
git add src/jev_bench/web/static tests/test_web_static.py
git commit -m "feat(ui): generations, explorer (with labelling) and runs pages

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```
