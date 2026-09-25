/**
 * Renders a ComparisonReport: a raters summary table, warnings, then one card per question with an
 * answer-count chart, per-rater statistics and pairwise metrics (95% bootstrap CIs) plus Fleiss' kappa.
 * Score questions show a 0-100 mean score; multi-label questions show labels per email, exact-set
 * match, Jaccard, F1 and the macro kappa at the threshold the report was computed with.
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
  const multi = question.type === "multi";
  const fleiss = `Fleiss κ${multi ? " (mean over labels)" : ""} ${fixed(question.fleiss_kappa, 3)}`;
  const tables = multi ? [multiRaterTable(question, labels), multiPairTable(question, labels)] : [raterTable(question, labels), pairTable(question, labels)];
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
        multi ? thresholdBadge(question.threshold) : null,
        h("span", { class: "ms-auto small text-body-secondary" }, fleiss),
      ),
      h("div", { class: "card-body" }, h("div", { class: "chart-box mb-3" }, canvas), tables),
    ),
  );
  return { element, draw: () => question.raters.length && countsChart(canvas, question, labels) };
}

function raterTable(question, labels) {
  const extra = question.type === "score" ? "Mean score (0–100)" : question.type === "noul" ? "Mean P(yes)" : null;
  const head = ["Rater", "n", "Entropy", "Confidence", ...(extra ? [extra] : [])];
  const rows = question.raters.map((stats) => {
    const value = question.type === "score" ? fixed(stats.mean_score, 0) : fixed(stats.mean.yes, 3);
    return h("tr", {}, h("td", { class: "small" }, labels[stats.rater] ?? stats.rater), h("td", {}, num(stats.n)), h("td", {}, fixed(stats.mean_entropy, 3)), h("td", {}, pct(stats.mean_confidence)), extra ? h("td", {}, value) : null);
  });
  return h("div", { class: "table-responsive" }, table(head, rows));
}

function interval(bounds, format) {
  return bounds ? h("small", { class: "text-body-secondary" }, ` [${format(bounds[0])}, ${format(bounds[1])}]`) : null;
}

function thresholdBadge(threshold) {
  const title = "A label counts as applied when its probability is at least this share of the most probable label";
  return h("span", { class: "badge text-bg-info", title }, `≥ ${Math.round(threshold * 100)}% of top`);
}

function multiRaterTable(question, labels) {
  const head = ["Rater", "n", "Labels / email", "Entropy", "Confidence"];
  const rows = question.raters.map((stats) =>
    h(
      "tr",
      {},
      h("td", { class: "small" }, labels[stats.rater] ?? stats.rater),
      h("td", {}, num(stats.n)),
      h("td", {}, fixed(stats.mean_labels, 2)),
      h("td", {}, fixed(stats.mean_entropy, 3)),
      h("td", {}, pct(stats.mean_confidence)),
    ),
  );
  return h("div", { class: "table-responsive" }, table(head, rows));
}

function noPairs() {
  return h("p", { class: "small text-body-secondary mb-0" }, "No overlapping raters for this question.");
}

function pairName(pair, labels) {
  return h("td", { class: "small" }, `${labels[pair.a] ?? pair.a} ↔ ${labels[pair.b] ?? pair.b}`);
}

function multiPairTable(question, labels) {
  if (!question.pairs.length) return noPairs();
  const head = ["Pair", "n", "Exact match", "Jaccard", "F1", "κ (macro)", "JSD", "Brier"];
  const two = (value) => fixed(value, 2);
  const rows = question.pairs.map((pair) =>
    h(
      "tr",
      {},
      pairName(pair, labels),
      h("td", {}, num(pair.n)),
      h("td", {}, pct(pair.agreement), interval(pair.agreement_ci, pct)),
      h("td", {}, fixed(pair.jaccard, 3), interval(pair.jaccard_ci, two)),
      h("td", {}, fixed(pair.f1, 3)),
      h("td", {}, fixed(pair.kappa, 3), interval(pair.kappa_ci, two)),
      h("td", {}, fixed(pair.jsd, 3)),
      h("td", {}, fixed(pair.brier, 3)),
    ),
  );
  return h("div", { class: "table-responsive" }, table(head, rows));
}

function pairTable(question, labels) {
  if (!question.pairs.length) return noPairs();
  const head = ["Pair", "n", "Agreement", "κ", "JSD", "r", "Brier"];
  const rows = question.pairs.map((pair) =>
    h(
      "tr",
      {},
      pairName(pair, labels),
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
