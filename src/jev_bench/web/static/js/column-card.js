/**
 * One Benchmark column card: the quality of the column's current run (filled by benchmark.js from the
 * comparison, see quality.js), model and request-mode pickers, the estimated cost of a run, Run/Cancel,
 * and the live or final statistics of that run. Every statistic carries a (?) tooltip. When the card's
 * slot holds several columns (Kev and Embeddings), the header shows a segmented control to swap them.
 * The card's accent color is its column's chart color (`accent-<column id>` in app.css).
 * Exports: columnCard, statsBlock, estimateBlock.
 */
import { h, icon } from "./dom.js";
import { duration, fixed, money, num, perMillion } from "./format.js";
import { withHelp } from "./glossary.js";
import { progressBar, statusBadge } from "./widgets.js";

export function columnCard(column, { model, mode, onModel, onMode, onRun, onCancel, running, siblings = [column], onSwap }) {
  const select = h("select", { class: "form-select form-select-sm", "aria-label": `${column.title} model`, onchange: (event) => onModel(event.target.value) }, column.models.map((item) => h("option", { value: item.id, selected: item.id === model }, modelLabel(item))));
  const run = h("button", { class: "btn btn-primary btn-sm", type: "button", onclick: () => onRun(select.value) }, icon("play-fill"), " Run");
  const cancel = h("button", { class: "btn btn-outline-danger btn-sm", type: "button", id: `cancel-${column.id}`, disabled: !running, onclick: onCancel }, icon("stop-fill"), " Cancel");
  const warning = column.error ? h("div", { class: "alert alert-warning py-1 px-2 small mb-0" }, icon("exclamation-triangle"), ` Catalog unavailable: ${column.error}`) : null;
  const header = h("div", { class: "card-header d-flex align-items-center gap-2" }, h("span", { class: "series-dot", "aria-hidden": "true" }), cardTitle(column, siblings, onSwap), h("span", { class: "badge text-bg-light border" }, column.kind));
  const body = h("div", { class: "card-body d-flex flex-column gap-2" }, warning, h("div", { class: "quality-slot d-flex flex-column gap-2", id: `quality-${column.id}` }), h("div", { class: "write-only" }, h("label", { class: "form-label small mb-1" }, "Model"), select, extras(column, mode, onMode)), h("div", { class: "estimate small write-only", id: `estimate-${column.id}` }), h("div", { class: "d-flex gap-2 write-only" }, run, cancel), h("div", { id: `stats-${column.id}` }));
  return h("div", { class: `card h-100 column-card shadow-sm accent-${column.id}` }, header, body);
}

function cardTitle(column, siblings, onSwap) {
  if (siblings.length < 2) return h("span", { class: "fw-semibold me-auto" }, column.title);
  const option = (sibling) => {
    const id = `slot-${column.slot}-${sibling.id}`;
    return [h("input", { type: "radio", class: "btn-check", name: `slot-${column.slot}`, id, checked: sibling.id === column.id, onchange: () => onSwap(sibling.id) }), h("label", { class: "btn btn-outline-secondary btn-sm fw-semibold", for: id }, sibling.title)];
  };
  return h("div", { class: "btn-group me-auto", role: "group", "aria-label": "Column shown in this slot" }, siblings.map(option));
}

function modelLabel(model) {
  return `${model.name} — ${perMillion(model.prompt_price_per_m)} / ${perMillion(model.completion_price_per_m)} per 1M`;
}

function extras(column, mode, onMode) {
  if (column.kind === "chat") return modeToggle(column, mode, onMode);
  if (column.kind === "embeddings") return h("div", { class: "small text-body-secondary mt-2" }, icon("thermometer-half"), ` τ = ${column.embedding_temperature} · ${column.emails_per_request} emails per request`);
  return null;
}

function modeToggle(column, mode, onMode) {
  const option = (value, text) => {
    const id = `mode-${column.id}-${value}`;
    return [h("input", { type: "radio", class: "btn-check", name: `mode-${column.id}`, id, value, checked: mode === value, onchange: () => onMode(value) }), h("label", { class: "btn btn-outline-secondary btn-sm", for: id }, text)];
  };
  return h("div", { class: "btn-group mt-2 w-100", role: "group", "aria-label": "Request mode" }, option("per_email", "Per email"), option("all_in_one", "All in one"));
}

export function estimateBlock(estimate) {
  if (estimate === null) return h("span", { class: "text-body-secondary" }, "Select a generation to estimate the cost.");
  if (estimate instanceof Error) return h("span", { class: "text-body-secondary" }, icon("exclamation-circle"), ` No estimate: ${estimate.message}`);
  const history = estimate.history_cost === null ? null : h("span", { class: "fw-semibold" }, `≈ ${money(estimate.history_cost)}`, h("span", { class: "fw-normal text-body-secondary" }, " from past runs"));
  const tokens = estimate.token_cost === null ? "no price information" : `≤ ${estimate.token_cost > 0 && estimate.token_cost < 0.00001 ? "$0.00001" : money(estimate.token_cost)} by tokens`;
  const detail = `${tokens} · ${num(estimate.n_emails)} emails · ${num(estimate.n_requests)} requests`;
  return h("div", { class: "estimate-box" }, h("div", { class: "stat-label mb-1" }, withHelp("Estimated cost", "estimate")), history, h("div", { class: history ? "text-body-secondary" : "fw-semibold" }, detail));
}

function statRows(run, live) {
  const rows = [
    ["Status", "status", statusBadge(run.status)],
    ["Elapsed", "elapsed", duration(live ? live.elapsed_s : run.duration_s)],
    ["Cost", "cost", money(live ? live.cost : run.total_cost)],
    ["Errors", "errors", num(live ? live.errors : run.n_errors)],
    ["Requests", "requests", `${num(run.n_requests)}${run.n_splits ? ` (+${run.n_splits} split)` : ""}`],
    ["Tokens in / out", "tokens", `${num(run.input_tokens)} / ${num(run.output_tokens)}`],
    ["Latency p50 / p95", "latency", `${fixed(run.latency_p50_ms, 0)} / ${fixed(run.latency_p95_ms, 0)} ms`],
  ];
  if (run.kind !== "embeddings") return rows;
  return [...rows, ["Cache hits", "cache_hits", `${num(run.cache_hits)} / ${num(run.cache_hits + run.cache_misses)}`], ["Cold cost", "cold_cost", money(run.cold_cost)]];
}

export function statsBlock(run, live) {
  if (!run) return h("p", { class: "small text-body-secondary mb-0" }, "No run on the selected generations yet.");
  const done = live ? live.done : run.n_done;
  const items = statRows(run, live).flatMap(([label, key, value]) => [h("dt", { class: "col-6 stat-label" }, withHelp(label, key)), h("dd", { class: "col-6 stat-value mb-1" }, value)]);
  return h(
    "div",
    {},
    progressBar(done, run.n_emails, { animated: run.status === "running" }),
    h("dl", { class: "row small mb-0 mt-2" }, items),
    h("div", { class: "small text-body-secondary text-truncate", title: run.id }, `${run.model} · ${run.mode} · ${run.id}`),
    run.error ? h("div", { class: "alert alert-danger py-1 px-2 small mt-2 mb-0" }, run.error) : null,
  );
}
