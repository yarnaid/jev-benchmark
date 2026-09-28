/**
 * Headline quality of a run against the hard raters, from report.quality of /api/compare: the mean κ over
 * questions, colored by its Landis–Koch band (above 0.8 almost perfect, 0.6 substantial, 0.4 moderate,
 * 0.2 fair, 0 slight, below 0 poor), with the mean agreement, questions and emails behind it.
 * kappaBand and cardQuality are pure: the band of a κ, and what a column card shows for its run (nothing,
 * "scored when it completes", "not in the comparison", "nothing to score against", or its scores).
 * qualityTargets lists the targets scored in a report, in a fixed order; targetHeading is a target's
 * column heading and glossary key; qualityCell is one summary-table cell.
 * Exports: kappaBand, cardQuality, qualityBlock, qualityTargets, targetHeading, qualityCell.
 */
import { h, icon } from "./dom.js";
import { fixed, num, pct } from "./format.js";
import { withHelp } from "./glossary.js";

const BANDS = [
  [0.8, "almost perfect", "success"],
  [0.6, "substantial", "success"],
  [0.4, "moderate", "warning"],
  [0.2, "fair", "warning"],
  [0, "slight", "danger"],
];
const TARGETS = { reference: ["κ vs reference", "quality_reference"], human: ["κ vs human", "quality_human"] };
const NOTES = {
  running: ["hourglass-split", "Quality is scored when the run completes."],
  absent: ["info-circle", "This run is not in the comparison below."],
  unscored: ["info-circle", "No reference answers or human labels to score against."],
};
const UNKNOWN = { label: "—", tone: "secondary" };

export function kappaBand(kappa) {
  if (typeof kappa !== "number" || !Number.isFinite(kappa)) return null;
  if (kappa < 0) return { label: "poor", tone: "danger" };
  const [, label, tone] = BANDS.find(([floor]) => kappa > floor) ?? BANDS.at(-1);
  return { label, tone };
}

export function cardQuality(run, report) {
  if (!run) return { state: "none" };
  if (run.status === "running") return { state: "running" };
  if (!report) return { state: "none" };
  if (!report.raters.some((rater) => rater.id === run.id)) return { state: "absent" };
  const scores = report.quality.filter((score) => score.rater === run.id);
  return scores.length ? { state: "scored", scores } : { state: "unscored" };
}

export function qualityBlock(view) {
  if (view.state === "scored") return view.scores.map((score, index) => scoreBox(score, index === 0));
  const note = NOTES[view.state];
  return note ? h("div", { class: "small text-body-secondary" }, icon(note[0]), ` ${note[1]}`) : null;
}

function scoreBox(score, primary) {
  const band = kappaBand(score.kappa) ?? UNKNOWN;
  const [label, key] = targetHeading(score.target);
  const detail = `agreement ${pct(score.agreement)} · ${num(score.n_questions)} questions · ${num(score.n_emails)} emails`;
  return h(
    "div",
    { class: `quality-box${primary ? "" : " quality-secondary"} bg-${band.tone}-subtle border border-${band.tone}-subtle` },
    h("div", { class: "stat-label" }, withHelp(label, key)),
    h("div", { class: `quality-kappa text-${band.tone}-emphasis` }, fixed(score.kappa, 3), h("span", { class: "quality-band" }, band.label)),
    h("div", { class: "small text-body-secondary" }, detail),
  );
}

export function qualityTargets(quality) {
  return Object.keys(TARGETS).filter((target) => quality.some((score) => score.target === target));
}

export function targetHeading(target) {
  return TARGETS[target] ?? [`κ vs ${target}`, "quality_reference"];
}

export function qualityCell(quality, raterId, target) {
  const score = quality.find((item) => item.rater === raterId && item.target === target);
  if (!score) return "—";
  const band = kappaBand(score.kappa) ?? UNKNOWN;
  return h("span", { class: `fw-semibold text-${band.tone}-emphasis`, title: `${band.label} · agreement ${pct(score.agreement)}` }, fixed(score.kappa, 3));
}
