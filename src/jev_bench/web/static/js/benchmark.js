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
