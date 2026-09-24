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
