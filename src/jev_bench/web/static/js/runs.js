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
let epoch = 0;

async function main() {
  await initLayout();
  document.getElementById("compare").addEventListener("click", () => go("./"));
  document.getElementById("explore").addEventListener("click", () => go("explorer.html"));
  await refresh();
}

function go(page) {
  const chosen = runs.filter((run) => selected.has(run.id));
  const params = new URLSearchParams({ runs: chosen.map((run) => run.id).join(",") });
  if (page !== "./") params.set("generations", [...new Set(chosen.flatMap((run) => run.generation_ids))].join(","));
  location.href = `${page}?${params}`;
}

async function refresh() {
  const requestEpoch = epoch;
  try {
    const fetched = (await api.runs()).map((view) => view.meta);
    if (requestEpoch !== epoch) return;
    runs = fetched;
    render();
    clearTimeout(timer);
    timer = runs.some((run) => run.status === "running") ? setTimeout(refresh, POLL_MS) : null;
  } catch (error) {
    toastError(error);
  }
}

function stopPolling() {
  epoch += 1;
  clearTimeout(timer);
  timer = null;
}

window.addEventListener("pagehide", stopPolling);
window.addEventListener("pageshow", (event) => {
  if (event.persisted) refresh().catch(toastError);
});

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
  const running = run.status === "running";
  const box = h("input", { class: "form-check-input", type: "checkbox", checked: selected.has(run.id), disabled: running, "aria-label": `select ${run.id}`, onchange: (event) => toggle(run.id, event.target.checked) });
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
