/**
 * Renders a ComparisonReport: the heading and raters summary (report-summary.js), warnings, then one card
 * per question with an answer-count chart, per-rater statistics and pairwise metrics (95% bootstrap CIs)
 * plus Fleiss' kappa. Score questions show a 0-100 mean score; multi-label questions show labels per
 * email, exact-set match, Jaccard, F1 and the macro kappa at the threshold the report was computed with.
 * Raters keep one validated color (palette.js) across the summary, the tables and the charts, and every
 * metric has a (?) tooltip.
 * Exports: renderReport.
 */
import { countsChart, destroyCharts } from "./charts.js";
import { clear, h, icon } from "./dom.js";
import { fixed, num, pct } from "./format.js";
import { hideTooltips, withHelp } from "./glossary.js";
import { raterColors } from "./palette.js";
import { raterLabel, reportHeader, summaryTable, swatch, table } from "./report-summary.js";

export function renderReport(container, report) {
  destroyCharts();
  hideTooltips(container);
  const labels = Object.fromEntries(report.raters.map((rater) => [rater.id, raterLabel(rater)]));
  const colors = raterColors(report.raters, document.documentElement.dataset.bsTheme);
  const view = { labels, colors };
  const cards = report.questions.map((question) => questionCard(question, view));
  const warnings = report.warnings.map((warning) => h("div", { class: "alert alert-warning py-1 small" }, icon("exclamation-triangle"), ` ${warning}`));
  clear(container, reportHeader(report), summaryTable(report.raters, labels, colors), warnings, h("div", { class: "row g-3" }, cards.map((card) => card.element)));
  for (const card of cards) card.draw();
}

function questionCard(question, view) {
  const canvas = h("canvas", { role: "img", "aria-label": `${question.id} answer counts` });
  const multi = question.type === "multi";
  const fleiss = withHelp(`Fleiss κ${multi ? " (mean over labels)" : ""} ${fixed(question.fleiss_kappa, 3)}`, multi ? "fleiss_multi" : "fleiss");
  const tables = multi ? [multiRaterTable(question, view), multiPairTable(question, view)] : [raterTable(question, view), pairTable(question, view)];
  const threshold = multi ? h("span", { class: "badge text-bg-info" }, `≥ ${Math.round(question.threshold * 100)}% of top`) : null;
  const headerRow = h("div", { class: "card-header d-flex align-items-center gap-2" }, h("code", { class: "fw-semibold" }, question.id), h("span", { class: "badge text-bg-light border" }, question.type), threshold, h("span", { class: "ms-auto small text-body-secondary" }, fleiss));
  const chart = [h("div", { class: "small text-body-secondary mb-1" }, withHelp("Answer counts", "answer_counts")), h("div", { class: "chart-box mb-3" }, canvas)];
  const element = h("div", { class: "col-12 col-xl-6" }, h("div", { class: "card h-100 shadow-sm question-card" }, headerRow, h("div", { class: "card-body" }, chart, tables)));
  return { element, draw: () => question.raters.length && countsChart(canvas, question, view.labels, view.colors) };
}

function raterName(id, view) {
  return h("td", { class: "small text-nowrap" }, swatch(view.colors.get(id)), view.labels[id] ?? id);
}

function raterTable(question, view) {
  const extra = question.type === "score" ? ["Mean score (0–100)", "mean_score"] : question.type === "noul" ? ["Mean P(yes)", "mean_yes"] : null;
  const head = [["Rater"], ["n", "n"], ["Entropy", "entropy"], ["Confidence", "confidence"], ...(extra ? [extra] : [])];
  const rows = question.raters.map((stats) => {
    const value = question.type === "score" ? fixed(stats.mean_score, 0) : fixed(stats.mean.yes, 3);
    return h("tr", {}, raterName(stats.rater, view), h("td", {}, num(stats.n)), h("td", {}, fixed(stats.mean_entropy, 3)), h("td", {}, pct(stats.mean_confidence)), extra ? h("td", {}, value) : null);
  });
  return h("div", { class: "table-responsive" }, table(head, rows));
}

function multiRaterTable(question, view) {
  const head = [["Rater"], ["n", "n"], ["Labels / email", "labels_per_email"], ["Entropy", "entropy"], ["Confidence", "confidence"]];
  const rows = question.raters.map((stats) => h("tr", {}, raterName(stats.rater, view), h("td", {}, num(stats.n)), h("td", {}, fixed(stats.mean_labels, 2)), h("td", {}, fixed(stats.mean_entropy, 3)), h("td", {}, pct(stats.mean_confidence))));
  return h("div", { class: "table-responsive" }, table(head, rows));
}

function interval(bounds, format) {
  return bounds ? h("small", { class: "d-block text-body-secondary text-nowrap" }, `[${format(bounds[0])}, ${format(bounds[1])}]`) : null;
}

function noPairs() {
  return h("p", { class: "small text-body-secondary mb-0" }, "No overlapping raters for this question.");
}

function pairName(pair, view) {
  const name = (id) => [swatch(view.colors.get(id)), view.labels[id] ?? id];
  return h("td", { class: "small" }, name(pair.a), " ↔ ", name(pair.b));
}

const two = (value) => fixed(value, 2);

function pairTable(question, view) {
  if (!question.pairs.length) return noPairs();
  const head = [["Pair"], ["n", "n"], ["Agreement", "agreement"], ["κ", "kappa"], ["JSD", "jsd"], ["r", "pearson"], ["Brier", "brier"]];
  const rows = question.pairs.map((pair) =>
    h("tr", {}, pairName(pair, view), h("td", {}, num(pair.n)), h("td", {}, pct(pair.agreement), interval(pair.agreement_ci, pct)), h("td", {}, fixed(pair.kappa, 3), interval(pair.kappa_ci, two)), h("td", {}, fixed(pair.jsd, 3)), h("td", {}, fixed(pair.pearson, 3)), h("td", {}, fixed(pair.brier, 3))),
  );
  return h("div", { class: "table-responsive" }, table(head, rows));
}

function multiPairTable(question, view) {
  if (!question.pairs.length) return noPairs();
  const head = [["Pair"], ["n", "n"], ["Exact match", "exact_match"], ["Jaccard", "jaccard"], ["F1", "f1"], ["κ (macro)", "kappa_macro"], ["JSD", "jsd"], ["Brier", "brier"]];
  const rows = question.pairs.map((pair) =>
    h(
      "tr",
      {},
      pairName(pair, view),
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
