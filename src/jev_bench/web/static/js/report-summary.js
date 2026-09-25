/**
 * The top of a comparison report: the heading with an "Explore emails" link, and the raters summary.
 * The summary shows each rater's color, emails, errors, time and cost (both also relative to the Jev run
 * in the comparison: ×1 = Jev), cost per email, cold cost, requests and latency. Every column header has
 * a (?) tooltip.
 * Exports: raterLabel, reportHeader, summaryTable, swatch, table.
 */
import { h, icon } from "./dom.js";
import { duration, fixed, money, num, times } from "./format.js";
import { withHelp } from "./glossary.js";

const HEAD = [
  ["Rater"],
  ["Emails", "emails"],
  ["Errors", "errors"],
  ["Duration", "duration"],
  ["Time vs Jev", "time_vs_jev"],
  ["Cost", "cost"],
  ["Cost vs Jev", "cost_vs_jev"],
  ["Cost / email", "cost_per_email"],
  ["Cold cost", "cold_cost"],
  ["Requests", "requests"],
  ["Latency p50 / p95", "latency"],
];

export function raterLabel(rater) {
  if (!rater.run) return rater.label;
  const model = rater.run.model.split("/").pop();
  return `${rater.run.column} · ${model}${rater.run.mode === "all_in_one" ? " · all-in-one" : ""}`;
}

export function reportHeader(report) {
  const runs = report.raters.filter((rater) => rater.run).map((rater) => rater.run);
  const generations = [...new Set(runs.flatMap((run) => run.generation_ids))];
  const explore = `/explorer.html?${new URLSearchParams({ generations: generations.join(","), runs: runs.map((run) => run.id).join(",") })}`;
  return h("div", { class: "d-flex flex-wrap align-items-center gap-2 mb-2" }, h("h2", { class: "h5 mb-0 me-auto" }, icon("graph-up"), " Comparison"), h("a", { class: "btn btn-sm btn-outline-primary", href: explore }, icon("search"), " Explore emails"));
}

export function swatch(color) {
  return h("span", { class: "series-dot me-2", style: `background: ${color}`, "aria-hidden": "true" });
}

export function table(head, rows, extraClass = "") {
  const cells = head.map(([text, key]) => h("th", { class: "small text-nowrap" }, key ? withHelp(text, key) : text));
  return h("table", { class: `table table-sm align-middle mb-2 ${extraClass}` }, h("thead", {}, h("tr", {}, cells)), h("tbody", {}, rows));
}

export function summaryTable(raters, labels, colors) {
  const jev = raters.find((rater) => rater.run?.kind === "decisions")?.run ?? null;
  const rows = raters.map((rater) => summaryRow(rater, labels[rater.id], colors.get(rater.id), jev));
  return h("div", { class: "table-responsive mb-3" }, table(HEAD, rows, "summary-table"));
}

function summaryRow(rater, label, color, jev) {
  const name = h("td", { class: "text-nowrap" }, swatch(color), label);
  return h("tr", {}, name, runCells(rater, jev).map((cell) => h("td", { class: "font-monospace small text-nowrap" }, cell)));
}

function runCells(rater, jev) {
  const run = rater.run;
  if (!run) return [num(rater.n_items), ...Array(HEAD.length - 2).fill("—")];
  const perEmail = run.n_done ? run.total_cost / run.n_done : null;
  const requests = `${num(run.n_requests)}${run.n_splits ? ` (+${run.n_splits})` : ""}`;
  return [num(run.n_done), num(run.n_errors), duration(run.duration_s), times(run.duration_s, jev?.duration_s), money(run.total_cost), times(run.total_cost, jev?.total_cost), money(perEmail), run.kind === "embeddings" ? money(run.cold_cost) : "—", requests, `${fixed(run.latency_p50_ms, 0)} / ${fixed(run.latency_p95_ms, 0)} ms`];
}
