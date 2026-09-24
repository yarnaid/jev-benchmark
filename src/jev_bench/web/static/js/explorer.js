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
