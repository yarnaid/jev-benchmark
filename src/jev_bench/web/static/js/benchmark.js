/**
 * Benchmark page: generation and run pickers, one card per column (column-card.js) with the quality of the
 * column's current run in the comparison (quality.js), the estimated cost of a run and the live stats of
 * the current one, and the comparison report of the chosen runs. By default
 * the latest completed run per column on exactly the selected generations is compared; picking runs (or
 * opening ?runs=…) pins a choice, and "Latest" returns to the default. A label-threshold slider appears in
 * the page header when the report has a multi-label question; it is rendered once and re-fetches the
 * comparison with ?threshold=. The report is redrawn with the other theme's colors when the theme changes.
 * Columns sharing a slot share one card with a toggle; the choice is stored per slot and "Latest" compares
 * the visible columns only (a hidden column's runs stay pickable).
 * Exports: none (page entry point).
 */
import { api, notPublished, publishedThreshold } from "./api.js";
import { columnCard, estimateBlock, statsBlock } from "./column-card.js";
import { STATIC } from "./deployment.js";
import { clear, h, icon } from "./dom.js";
import { initLayout, startJob, THEME_EVENT, toastError } from "./layout.js";
import { hideTooltips } from "./glossary.js";
import { cardQuality, qualityBlock } from "./quality.js";
import { renderReport } from "./report.js";
import { defaultRunIds, generationChoices, initialGenerations, orderByColumn, runChoices, sameSet } from "./selection.js";
import { hiddenColumnIds, slotPicks, slotPrefKey, slotView, visibleColumns } from "./slots.js";
import { readPref, writePref } from "./storage.js";
import { checklist, emptyState, thresholdSlider } from "./widgets.js";

const POLL_MS = 1000;
const ESTIMATE_DELAY_MS = 300;
const state = {
  catalog: [],
  generations: [],
  runs: [],
  progress: new Map(),
  timers: new Map(),
  estimates: new Map(),
  selected: [],
  chosen: [],
  explicit: false,
  threshold: readPref("threshold", null),
  sliderShown: false,
  report: null,
};

const selectedSet = () => new Set(state.selected);
const modelOf = (column) => readPref(`benchmark.model.${column.id}`, column.default_model);
const modeOf = (column) => readPref(`benchmark.mode.${column.id}`, "per_email");

async function main() {
  await initLayout();
  const requested = (new URLSearchParams(location.search).get("runs") ?? "").split(",").filter(Boolean);
  const [catalog, generations, runs] = await Promise.all([api.catalog(), api.generations(), api.runs()]);
  state.catalog = catalog;
  state.generations = generations.map((view) => view.meta).filter((meta) => meta.done > 0);
  state.runs = runs.map((view) => view.meta);
  state.selected = initialGenerations(state.generations, state.runs, requested, readPref("benchmark.generations", []));
  state.explicit = requested.length > 0;
  state.chosen = state.explicit ? requested : defaultRuns();
  renderPickers();
  renderColumns();
  trackRunning();
  window.addEventListener(THEME_EVENT, redrawReport);
  await refreshComparison();
}

const picks = () => slotPicks(state.catalog, (key) => readPref(key, null));
const shownColumns = () => visibleColumns(state.catalog, picks());
const defaultRuns = () => defaultRunIds(state.runs, state.selected, state.catalog, hiddenColumnIds(state.catalog, picks()));

function renderPickers() {
  const generations = checklist({ label: "Generations", items: generationChoices(state.generations), selected: state.selected, onChange: onGenerations });
  clear(document.getElementById("generation-picker"), h("div", { class: "d-flex flex-wrap gap-2" }, generations, h("div", { class: "d-flex flex-wrap gap-2", id: "run-picker" })));
  renderRunPicker();
}

function renderRunPicker() {
  const latest = h("button", { class: "btn btn-sm btn-outline-secondary", type: "button", id: "latest-runs", title: "Compare the latest completed run of every column", disabled: !state.explicit, onclick: resetRuns }, icon("arrow-counterclockwise"), " Latest");
  clear(document.getElementById("run-picker"), checklist({ label: "Runs", items: runChoices(state.runs, state.selected, state.catalog), selected: state.chosen, onChange: onRuns, disabled: STATIC }), latest);
}

function onGenerations(selected) {
  state.selected = selected;
  writePref("benchmark.generations", selected);
  resetRuns();
  for (const column of shownColumns()) {
    refreshStats(column.id);
    scheduleEstimate(column);
  }
}

function onRuns(chosen) {
  state.chosen = chosen;
  state.explicit = true;
  syncUrl();
  document.getElementById("latest-runs").disabled = false;
  refreshComparison().catch(toastError);
}

function resetRuns() {
  state.explicit = false;
  state.chosen = defaultRuns();
  syncUrl();
  renderRunPicker();
  refreshComparison().catch(toastError);
}

function syncUrl() {
  const query = state.explicit && state.chosen.length ? `?${new URLSearchParams({ runs: state.chosen.join(",") })}` : "";
  history.replaceState(null, "", `${location.pathname}${query}`);
}

function renderColumns() {
  const container = clear(document.getElementById("columns"));
  if (!state.catalog.length) {
    container.append(emptyState("No columns are configured in config/benchmark.toml."));
    return;
  }
  for (const { slot, columns, shown: column } of slotView(state.catalog, picks())) {
    const handlers = {
      model: modelOf(column),
      mode: modeOf(column),
      running: currentRun(column.id)?.status === "running",
      siblings: columns,
      onSwap: (id) => swapSlot(slot, id),
      onModel: (value) => savePref(column, "model", value),
      onMode: (value) => savePref(column, "mode", value),
      onRun: (model) => launch(column, model),
      onCancel: () => cancelColumn(column.id),
    };
    container.append(h("div", { class: "col-12 col-md-6 col-xxl-3" }, columnCard(column, handlers)));
    refreshStats(column.id);
    scheduleEstimate(column, 0);
  }
}

function swapSlot(slot, columnId) {
  writePref(slotPrefKey(slot), columnId);
  renderColumns();
  if (!state.explicit) resetRuns();
}

function savePref(column, name, value) {
  writePref(`benchmark.${name}.${column.id}`, value);
  scheduleEstimate(column);
}

function requestBody(column, model) {
  const body = { column: column.id, model, generation_ids: state.selected };
  if (column.kind === "chat") body.mode = modeOf(column);
  return body;
}

function scheduleEstimate(column, delay = ESTIMATE_DELAY_MS) {
  if (STATIC) return;
  clearTimeout(state.estimates.get(column.id)?.timer);
  const token = { timer: setTimeout(() => refreshEstimate(column, token), delay) };
  state.estimates.set(column.id, token);
}

async function refreshEstimate(column, token) {
  const show = (estimate) => {
    const target = document.getElementById(`estimate-${column.id}`);
    if (target && state.estimates.get(column.id) === token) replaceContent(target, estimateBlock(estimate));
  };
  if (!state.selected.length) return show(null);
  show(await api.estimateRun(requestBody(column, modelOf(column))).catch((error) => error));
}

function replaceContent(target, content) {
  hideTooltips(target);
  clear(target, content);
}

function currentRun(columnId) {
  const selected = selectedSet();
  const matching = state.runs.filter((run) => run.column === columnId && sameSet(run.generation_ids, selected));
  return matching.find((run) => run.status === "running") ?? matching[0];
}

function refreshStats(columnId) {
  const run = currentRun(columnId);
  const live = run?.status === "running" ? state.progress.get(run.id) : null;
  const target = document.getElementById(`stats-${columnId}`);
  if (target) replaceContent(target, statsBlock(run, live));
  const cancel = document.getElementById(`cancel-${columnId}`);
  if (cancel) cancel.disabled = run?.status !== "running";
  refreshQuality(columnId, run);
}

function refreshQuality(columnId, run = currentRun(columnId)) {
  const target = document.getElementById(`quality-${columnId}`);
  if (target) replaceContent(target, qualityBlock(cardQuality(run, state.report)));
}

function refreshAllQuality() {
  for (const column of state.catalog) refreshQuality(column.id);
}

async function launch(column, model) {
  if (!state.selected.length) {
    toastError("Select at least one generation first.");
    return;
  }
  const view = await startJob(() => api.createRun(requestBody(column, model)));
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
  if (!state.timers.has(runId)) state.timers.set(runId, setTimeout(() => poll(runId), POLL_MS));
}

function stopTracking(runId) {
  clearTimeout(state.timers.get(runId));
  state.timers.delete(runId);
}

async function poll(runId) {
  try {
    const view = await api.run(runId);
    if (!state.timers.has(runId)) return;
    state.timers.delete(runId);
    upsertRun(view.meta, view.progress);
    if (view.meta.status === "running") return track(runId);
    runFinished(view.meta);
  } catch (error) {
    stopTracking(runId);
    toastError(error);
  }
}

function runFinished(meta) {
  const column = shownColumns().find((item) => item.id === meta.column);
  if (column) scheduleEstimate(column);
  if (state.explicit) return renderRunPicker();
  resetRuns();
}

function trackRunning() {
  for (const run of state.runs) if (run.status === "running") track(run.id);
}

window.addEventListener("pagehide", () => {
  for (const runId of [...state.timers.keys()]) stopTracking(runId);
});
window.addEventListener("pageshow", (event) => {
  if (event.persisted) trackRunning();
});

async function refreshComparison() {
  const container = document.getElementById("comparison");
  const known = new Set(state.runs.map((run) => run.id));
  const runIds = orderByColumn(state.chosen.filter((id) => known.has(id)), state.runs, state.catalog);
  state.report = null;
  refreshAllQuality();
  if (!runIds.length) {
    clear(container, emptyState("Run at least one column on the selected generations, or pick runs, to see the comparison.", "bar-chart"));
    return;
  }
  clear(container, h("p", { class: "small text-body-secondary" }, "Loading comparison…"));
  try {
    const loaded = await publishedThreshold((threshold) => api.compare(runIds, threshold), state.threshold);
    state.threshold = loaded.threshold;
    state.report = loaded.value;
  } catch (error) {
    if (!notPublished(error)) throw error;
    return clear(container, emptyState(error.message, "camera"));
  }
  renderThreshold(state.report);
  renderReport(container, state.report);
  refreshAllQuality();
}

function redrawReport() {
  if (state.report) renderReport(document.getElementById("comparison"), state.report);
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

main().catch(toastError);
