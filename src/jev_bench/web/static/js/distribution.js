/**
 * Probability visualisation for the Explorer detail panel: a table cell with a bar and the value;
 * the tooltip adds the cosine similarity for embedding runs, and an optional `tick` (a probability,
 * e.g. a multi-label question's applied limit) is drawn as a mark on the bar.
 * Exports: probabilityCell.
 */
import { h } from "./dom.js";
import { fixed } from "./format.js";

export function probabilityCell(probability, { similarity = null, highlight = false, tick = null } = {}) {
  const value = probability ?? 0;
  const title = similarity === null ? `p = ${fixed(value, 3)}` : `p = ${fixed(value, 3)} · cos = ${fixed(similarity, 3)}`;
  const mark = tick === null || !Number.isFinite(tick) ? null : h("div", { class: "prob-tick", style: `left: ${Math.round(tick * 100)}%`, "aria-hidden": "true" });
  const bar = h("div", { class: "prob-track flex-grow-1" }, h("div", { class: "prob-bar", style: `width: ${Math.round(value * 100)}%` }), mark);
  return h("td", { class: highlight ? "fw-semibold" : "", title }, h("div", { class: "d-flex align-items-center gap-2" }, bar, h("span", { class: "small font-monospace" }, fixed(value, 2))));
}
