/**
 * Explorer detail panel content: the email header, the (hostile-by-design) body as plain text, and one
 * card per question with each run's probability per option, the generator reference, a 0-100 score
 * row for score questions, and the human-label control (a select, or checkboxes for a multi-label
 * question, whose card also shows the threshold; per run, a tick marks threshold x the top probability
 * and labels at or above it are bold).
 * Exports: emailDetail.
 */
import { appliedLimit, asLabels } from "./answers.js";
import { probabilityCell } from "./distribution.js";
import { h, icon } from "./dom.js";
import { fixed, when } from "./format.js";

export function emailDetail(detail, { row = null, threshold = null, onSave }) {
  return [emailHeader(detail.email), h("div", { class: "email-body mb-3" }, detail.email.body), labellingForm(detail, row, threshold, onSave)];
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

function labellingForm(detail, row, threshold, onSave) {
  const choices = { ...detail.human };
  const context = { detail, row, threshold, choices, runIds: Object.keys(detail.predictions) };
  const sections = detail.questions.questions.map((question) => questionSection(question, context));
  const save = h("button", { class: "btn btn-primary", type: "button", onclick: () => onSave(detail.email.id, choices, detail.human) }, icon("save"), " Save labels");
  return h("div", {}, sections, h("div", { class: "d-flex justify-content-end mt-2" }, save));
}

function questionSection(question, { detail, row, threshold, choices, runIds }) {
  const cut = question.type === "multi" ? (threshold ?? question.threshold) : null;
  const reference = asLabels(detail.email.reference_answers[question.id]);
  const head = ["Option", "Ref", ...runIds.map((id) => detail.run_labels[id] ?? id)];
  const rows = Object.entries(question.options).map(([option, text]) =>
    h("tr", {}, h("td", { class: "small", title: text }, option), h("td", { class: "text-center" }, reference.includes(option) ? icon("check-lg") : ""), runIds.map((id) => runCell(detail.predictions[id], question.id, option, cut))),
  );
  const foot = question.type === "score" ? scoreRow(question.id, row, runIds) : null;
  const table = h("table", { class: "table table-sm mb-2" }, h("thead", {}, h("tr", {}, head.map((cell) => h("th", { class: "small" }, cell)))), h("tbody", {}, rows), foot);
  const badge = cut === null ? null : h("span", { class: "badge text-bg-info", title: "A label counts as applied when its probability is at least this share of the most probable label" }, `≥ ${Math.round(cut * 100)}% of top`);
  const human = h("div", { class: "d-flex align-items-center gap-2" }, h("span", { class: "small text-nowrap" }, icon("person"), " Human"), humanControl(question, detail.human[question.id], choices));
  return h(
    "div",
    { class: "card mb-2" },
    h("div", { class: "card-header py-1 d-flex align-items-center gap-2" }, h("code", {}, question.id), badge, h("span", { class: "small text-body-secondary text-truncate" }, question.instructions)),
    h("div", { class: "card-body py-2" }, h("div", { class: "table-responsive" }, table), human),
  );
}

function scoreRow(questionId, row, runIds) {
  if (!row) return null;
  const cell = (score) => h("td", { class: "small font-monospace" }, fixed(score, 0));
  return h("tfoot", {}, h("tr", {}, h("td", { class: "small fw-semibold" }, "score (0–100)"), cell(row.reference_scores?.[questionId]), runIds.map((id) => cell(row.scores?.[id]?.[questionId]))));
}

const topOption = (distribution) => Object.entries(distribution).reduce((best, entry) => (entry[1] > best[1] ? entry : best))[0];

function runCell(prediction, questionId, option, cut) {
  if (!prediction) return h("td", { class: "small text-body-secondary" }, "—");
  if (prediction.error) return h("td", { class: "small text-danger", title: prediction.error }, "error");
  const distribution = prediction.answers?.[questionId];
  if (!distribution) return h("td", { class: "small text-body-secondary" }, "—");
  const similarity = prediction.similarities?.[questionId]?.[option] ?? null;
  if (cut === null) return probabilityCell(distribution[option], { similarity, highlight: topOption(distribution) === option });
  const limit = appliedLimit(distribution, cut);
  return probabilityCell(distribution[option], { similarity, highlight: (distribution[option] ?? 0) >= limit, tick: limit });
}

function humanControl(question, current, choices) {
  return question.type === "multi" ? humanLabels(question, current, choices) : humanSelect(question, current, choices);
}

function humanSelect(question, current, choices) {
  const onchange = (event) => {
    choices[question.id] = event.target.value || null;
  };
  const options = Object.entries(question.options).map(([option, text]) => h("option", { value: option, selected: current === option, title: text }, option));
  return h("select", { class: "form-select form-select-sm", "aria-label": `Human label for ${question.id}`, onchange }, h("option", { value: "" }, "— not labelled —"), options);
}

function humanLabels(question, current, choices) {
  const chosen = new Set(asLabels(current));
  const toggle = (option, checked) => {
    if (checked) chosen.add(option);
    else chosen.delete(option);
    choices[question.id] = [...chosen];
  };
  const boxes = Object.entries(question.options).map(([option, text], index) => {
    const id = `human-${question.id}-${index}`;
    const input = h("input", { class: "form-check-input", type: "checkbox", id, checked: chosen.has(option), onchange: (event) => toggle(option, event.target.checked) });
    return h("div", { class: "form-check form-check-inline mb-0" }, input, h("label", { class: "form-check-label small", for: id, title: text }, option));
  });
  return h("div", { class: "d-flex flex-wrap", role: "group", "aria-label": `Human labels for ${question.id}` }, boxes);
}
