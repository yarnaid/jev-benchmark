/**
 * Analyze page: pick the generations and runs, the analyst model, the number of disputed emails and the
 * instructions (edits are kept per browser), see the estimated cost, start an analysis and follow it live,
 * then read, copy or download the report. Past analyses are listed below. The runs are sent in catalog
 * column order, so the analyst's R1 is Jev. ?analysis=<id> opens an analysis; ?runs=… pins the runs.
 * Exports: none (page entry point).
 */
import { api } from "./api.js";
import { historyTable, resultCard } from "./analysis-result.js";
import { estimateView, setupCard } from "./analysis-setup.js";
import { clear, h, icon } from "./dom.js";
import { hideTooltips } from "./glossary.js";
import { initLayout, startJob, toastError, toastSuccess } from "./layout.js";
import { defaultRunIds, generationChoices, initialGenerations, orderByColumn, runChoices, runText } from "./selection.js";
import { readPref, removePref, writePref } from "./storage.js";
import { checklist, thresholdSlider } from "./widgets.js";

const POLL_MS = 1000;
const ESTIMATE_DELAY_MS = 500;
const state = {
  catalog: [],
  generations: [],
  runs: [],
  defaults: null,
  questions: [],
  history: [],
  selected: [],
  chosen: [],
  explicit: false,
  model: "",
  disputed: 0,
  prompts: { system: null, user: null },
  threshold: readPref("threshold", null),
  estimate: { token: null, timer: null },
  current: null,
  timer: null,
  promptsOpen: false,
  sentPrompts: new Map(),
};

async function main() {
  await initLayout();
  const params = new URLSearchParams(location.search);
  const requested = (params.get("runs") ?? "").split(",").filter(Boolean);
  const [catalog, generations, runs, defaults, questions, history] = await Promise.all([api.catalog(), api.generations(), api.runs(), api.analysisDefaults(), api.questions(), api.analyses()]);
  Object.assign(state, { catalog, defaults, history, questions: questions.questions, runs: runs.map((view) => view.meta) });
  state.generations = generations.map((view) => view.meta).filter((meta) => meta.done > 0);
  state.selected = initialGenerations(state.generations, state.runs, requested, readPref("analyze.generations", []));
  state.explicit = requested.length > 0;
  state.chosen = state.explicit ? requested : defaultRuns();
  state.model = readPref("analyze.model", defaults.default_model);
  state.disputed = readPref("analyze.disputed", defaults.max_disputed_emails);
  state.prompts = { system: readPref("analyze.prompt.system", null), user: readPref("analyze.prompt.user", null) };
  renderPickers();
  renderThreshold();
  renderSetup();
  scheduleEstimate(0);
  await show(params.get("analysis") ?? state.history[0]?.meta.id ?? null);
}

const defaultRuns = () => defaultRunIds(state.runs, state.selected, state.catalog);
const runLabel = (id) => {
  const run = state.runs.find((item) => item.id === id);
  return run ? runText(run, state.catalog) : id;
};

function renderPickers() {
  const generations = checklist({ label: "Generations", items: generationChoices(state.generations), selected: state.selected, onChange: onGenerations });
  clear(document.getElementById("pickers"), generations, h("div", { class: "d-flex flex-wrap gap-2", id: "run-picker" }));
  renderRunPicker();
}

function renderRunPicker() {
  const latest = h("button", { class: "btn btn-sm btn-outline-secondary", type: "button", id: "latest-runs", title: "Analyse the latest completed run of every column", disabled: !state.explicit, onclick: resetRuns }, icon("arrow-counterclockwise"), " Latest");
  clear(document.getElementById("run-picker"), checklist({ label: "Runs", items: runChoices(state.runs, state.selected, state.catalog), selected: state.chosen, onChange: onRuns }), latest);
}

function onGenerations(selected) {
  state.selected = selected;
  writePref("analyze.generations", selected);
  resetRuns();
}

function onRuns(chosen) {
  state.chosen = chosen;
  state.explicit = true;
  syncUrl();
  document.getElementById("latest-runs").disabled = false;
  scheduleEstimate();
}

function resetRuns() {
  state.explicit = false;
  state.chosen = defaultRuns();
  syncUrl();
  renderRunPicker();
  scheduleEstimate();
}

function syncUrl() {
  const params = new URLSearchParams();
  if (state.explicit && state.chosen.length) params.set("runs", state.chosen.join(","));
  if (state.current) params.set("analysis", state.current);
  const query = params.toString();
  history.replaceState(null, "", `${location.pathname}${query ? `?${query}` : ""}`);
}

function renderThreshold() {
  const multi = state.questions.find((question) => question.type === "multi");
  const target = document.getElementById("threshold-control");
  if (!multi) return clear(target);
  const onChange = (value) => {
    state.threshold = value;
    writePref("threshold", value);
    scheduleEstimate();
  };
  clear(target, thresholdSlider({ value: state.threshold ?? multi.threshold, onChange }));
}

function renderSetup() {
  const chat = state.catalog.filter((column) => column.kind === "chat").flatMap((column) => column.models.map((model) => model.id));
  const handlers = { onModel: setModel, onDisputed: setDisputed, onPrompt: setPrompt, onRun: launch };
  const view = { defaults: state.defaults, models: [...new Set([state.defaults.default_model, ...chat])], model: state.model, disputed: state.disputed, prompts: state.prompts };
  clear(document.getElementById("setup"), setupCard({ ...view, ...handlers }));
}

function setModel(value) {
  state.model = value.trim();
  if (state.model) writePref("analyze.model", state.model);
  else removePref("analyze.model");
  scheduleEstimate();
}

function setDisputed(input) {
  const value = Math.min(100, Math.max(0, Math.round(Number(input.value) || 0)));
  input.value = value;
  state.disputed = value;
  writePref("analyze.disputed", value);
  scheduleEstimate();
}

function setPrompt(kind, value) {
  state.prompts[kind] = value;
  if (value === null) removePref(`analyze.prompt.${kind}`);
  else writePref(`analyze.prompt.${kind}`, value);
  scheduleEstimate();
}

function requestBody() {
  const known = new Set(state.runs.map((run) => run.id));
  const runs = orderByColumn(state.chosen.filter((id) => known.has(id)), state.runs, state.catalog);
  return { runs, model: state.model || null, system_prompt: state.prompts.system, user_prompt: state.prompts.user, threshold: state.threshold, max_disputed_emails: state.disputed };
}

function scheduleEstimate(delay = ESTIMATE_DELAY_MS) {
  clearTimeout(state.estimate.timer);
  const token = {};
  state.estimate = { token, timer: setTimeout(() => refreshEstimate(token), delay) };
}

async function refreshEstimate(token) {
  const body = requestBody();
  const estimate = body.runs.length ? await api.estimateAnalysis(body).catch((error) => error) : null;
  if (state.estimate.token !== token) return;
  const target = document.getElementById("analysis-estimate");
  hideTooltips(target);
  clear(target, estimateView(estimate));
  document.getElementById("run-analysis").disabled = !(estimate && !(estimate instanceof Error) && estimate.fits);
}

async function launch() {
  const body = requestBody();
  if (!body.runs.length) return toastError("Select at least one run first.");
  const view = await startJob(() => api.createAnalysis(body));
  if (!view) return;
  state.history = [view, ...state.history.filter((item) => item.meta.id !== view.meta.id)];
  await show(view.meta.id);
}

async function show(analysisId) {
  clearTimeout(state.timer);
  state.current = analysisId;
  syncUrl();
  renderHistory();
  if (!analysisId) return renderResult(null);
  const view = await api.analysis(analysisId).catch((error) => (toastError(error), null));
  if (state.current === analysisId) follow(view);
}

function follow(view) {
  renderResult(view);
  if (!view) return;
  if (view.meta.status === "running") state.timer = setTimeout(() => poll(view.meta.id), POLL_MS);
  else if (state.history.find((item) => item.meta.id === view.meta.id)?.meta.status !== view.meta.status) refreshHistory();
}

async function poll(analysisId) {
  const view = await api.analysis(analysisId).catch((error) => (toastError(error), null));
  if (state.current === analysisId && view) follow(view);
}

async function refreshHistory() {
  state.history = await api.analyses().catch(() => state.history);
  renderHistory();
}

function renderHistory() {
  const target = document.getElementById("history");
  clear(target, historyTable(state.history, { selectedId: state.current, onSelect: (id) => show(id).catch(toastError) }));
}

function renderResult(view) {
  const target = document.getElementById("result");
  hideTooltips(target);
  const handlers = { runLabel, onCancel: cancel, onCopy: copy, onDownload: download, loadPrompts, promptsOpen: state.promptsOpen, onPromptsToggle: (open) => (state.promptsOpen = open) };
  clear(target, resultCard(view, handlers));
}

function loadPrompts(analysisId) {
  if (!state.sentPrompts.has(analysisId)) {
    const forget = (error) => {
      state.sentPrompts.delete(analysisId);
      throw error;
    };
    state.sentPrompts.set(analysisId, api.analysisPrompts(analysisId).catch(forget));
  }
  return state.sentPrompts.get(analysisId);
}

async function cancel(analysisId) {
  await api.cancelAnalysis(analysisId).catch(toastError);
}

function copy(meta) {
  navigator.clipboard.writeText(meta.result).then(() => toastSuccess("Report copied as Markdown."), toastError);
}

function download(meta) {
  const url = URL.createObjectURL(new Blob([meta.result], { type: "text/markdown" }));
  const link = document.createElement("a");
  link.href = url;
  link.download = `${meta.id}.md`;
  link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

window.addEventListener("pagehide", () => clearTimeout(state.timer));

main().catch(toastError);
