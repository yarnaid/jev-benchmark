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
let epoch = 0;

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
  const requestEpoch = epoch;
  try {
    const views = await api.generations();
    if (requestEpoch !== epoch) return;
    render(views);
    schedule(views.some((view) => view.meta.status === "running"));
  } catch (error) {
    if (requestEpoch === epoch) schedule(false);
    toastError(error);
  }
}

function schedule(active) {
  clearTimeout(timer);
  timer = active ? setTimeout(refresh, POLL_MS) : null;
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
