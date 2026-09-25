/**
 * Explorer detail panel content: the email header (one icon per field), the (hostile-by-design) body as
 * plain text, and one compact card per question. In each card:
 * - options of choice and multi-label questions are ordered by option-order.js (reference first, then
 *   the most probable), and unlikely options are folded behind a "Show N more" row. Scales and yes/no
 *   keep their natural order;
 * - the reference row is highlighted with a ✓;
 * - each run's answer (its top option, or its applied labels at threshold × top) is bold, green when it
 *   is in the reference and red when it is not;
 * - multi-label bars carry a tick at the run's applied limit, score questions get a 0-100 score row, and
 *   the human-label control is a select or checkboxes.
 * Run columns use the caller's short `labels` (the full run label is the header's tooltip).
 * Exports: emailDetail.
 */
import { appliedLimit, asLabels } from "./answers.js";
import { probabilityCell } from "./distribution.js";
import { h, icon } from "./dom.js";
import { fixed, when } from "./format.js";
import { helpIcon, withHelp } from "./glossary.js";
import { orderOptions } from "./option-order.js";

const HEADER_FIELDS = [
  ["From", "person-circle"],
  ["To", "envelope"],
  ["Cc", "people"],
  ["Sent", "clock"],
  ["Generator", "cpu"],
  ["Traits", "tags"],
  ["Id", "hash"],
];

export function emailDetail(detail, { row = null, threshold = null, labels = {}, onSave }) {
  return [emailHeader(detail.email), h("div", { class: "email-body mb-3" }, detail.email.body), labellingForm(detail, { row, threshold, labels }, onSave)];
}

function emailHeader(email) {
  const party = (value) => (value.name ? `${value.name} <${value.address}>` : value.address);
  const traits = Object.entries(email.traits).map(([name, value]) => `${name}=${value}`).join(", ");
  const values = [party(email.sender), email.to.map(party).join(", "), email.cc.map(party).join(", ") || "—", when(email.sent_at), email.generator_model, traits || "—", email.id];
  const cells = HEADER_FIELDS.flatMap(([label, name], index) => [h("dt", { class: "text-body-secondary fw-normal text-nowrap" }, icon(name), ` ${label}`), h("dd", { class: "mb-0 text-break" }, values[index])]);
  return h("dl", { class: "email-meta small mb-2" }, cells);
}

function labellingForm(detail, view, onSave) {
  const choices = { ...detail.human };
  const context = { ...view, detail, choices, runIds: Object.keys(detail.predictions) };
  const sections = detail.questions.questions.map((question) => questionSection(question, context));
  const save = h("button", { class: "btn btn-primary", type: "button", onclick: () => onSave(detail.email.id, choices, detail.human) }, icon("save"), " Save labels");
  return h("div", {}, sections, h("div", { class: "d-flex justify-content-end mt-2" }, save));
}

function answerOf(prediction, questionId) {
  return prediction && !prediction.error ? prediction.answers?.[questionId] ?? null : null;
}

function layout(question, reference, distributions) {
  if (question.type === "score" || question.type === "noul") return { shown: Object.keys(question.options), folded: [] };
  return orderOptions(Object.keys(question.options), reference, distributions);
}

function questionSection(question, context) {
  const { detail, runIds } = context;
  const cut = question.type === "multi" ? (context.threshold ?? question.threshold) : null;
  const reference = asLabels(detail.email.reference_answers[question.id]);
  const { shown, folded } = layout(question, reference, runIds.map((id) => answerOf(detail.predictions[id], question.id)));
  const optionRow = (option) => optionRowFor(question, option, reference, cut, context);
  const head = h("thead", {}, h("tr", {}, h("th", { class: "small" }, "Option"), h("th", { class: "small text-center" }, withHelp("Ref", "reference")), runIds.map((id) => runHeader(id, context))));
  const foldedBody = h("tbody", { class: "d-none" }, folded.map(optionRow));
  const table = h("table", { class: "table table-sm mb-2 detail-table" }, head, h("tbody", {}, shown.map(optionRow)), foldToggle(folded.length, foldedBody, runIds.length), foldedBody, question.type === "score" ? scoreRow(question.id, context.row, runIds) : null);
  const badge = cut === null ? null : h("span", { class: "badge text-bg-info" }, `≥ ${Math.round(cut * 100)}% of top `, helpIcon("threshold"));
  const human = h("div", { class: "d-flex align-items-center gap-2" }, h("span", { class: "small text-nowrap" }, icon("person"), " ", withHelp("Human", "human")), humanControl(question, detail.human[question.id], context.choices));
  return h(
    "div",
    { class: "card mb-2 detail-card" },
    h("div", { class: "card-header py-1 d-flex align-items-center gap-2" }, h("code", {}, question.id), badge, h("span", { class: "small text-body-secondary text-truncate", title: question.instructions }, question.instructions)),
    h("div", { class: "card-body py-2" }, h("div", { class: "table-responsive" }, table), human),
  );
}

function runHeader(id, { detail, labels }) {
  const full = detail.run_labels[id] ?? id;
  return h("th", { class: "small", title: full }, labels[id] ?? full);
}

function foldToggle(count, foldedBody, runs) {
  if (!count) return null;
  const button = h("button", { class: "btn btn-link btn-sm p-0", type: "button", onclick: () => toggle(foldedBody, button, count) }, icon("chevron-down"), ` Show ${count} more (below 5% everywhere)`);
  return h("tbody", {}, h("tr", {}, h("td", { colspan: runs + 2, class: "py-1" }, button)));
}

function toggle(foldedBody, button, count) {
  const hidden = foldedBody.classList.toggle("d-none");
  button.replaceChildren(icon(hidden ? "chevron-down" : "chevron-up"), hidden ? ` Show ${count} more (below 5% everywhere)` : " Show fewer");
}

function optionRowFor(question, option, reference, cut, { detail, runIds }) {
  const isReference = reference.includes(option);
  const label = h("td", { class: "small", title: question.options[option] }, option);
  const check = h("td", { class: "text-center" }, isReference ? icon("check-lg") : "");
  const cells = runIds.map((id) => runCell(detail.predictions[id], question.id, option, cut, isReference));
  return h("tr", { class: isReference ? "row-reference" : "" }, label, check, cells);
}

function scoreRow(questionId, row, runIds) {
  if (!row) return null;
  const cell = (score) => h("td", { class: "small font-monospace" }, fixed(score, 0));
  return h("tfoot", {}, h("tr", {}, h("td", { class: "small fw-semibold text-nowrap" }, withHelp("Score", "score")), cell(row.reference_scores?.[questionId]), runIds.map((id) => cell(row.scores?.[id]?.[questionId]))));
}

const topOption = (distribution) => Object.entries(distribution).reduce((best, entry) => (entry[1] > best[1] ? entry : best))[0];

function runCell(prediction, questionId, option, cut, isReference) {
  if (!prediction) return h("td", { class: "small text-body-secondary" }, "—");
  if (prediction.error) return h("td", { class: "small text-danger", title: prediction.error }, "error");
  const distribution = prediction.answers?.[questionId];
  if (!distribution) return h("td", { class: "small text-body-secondary" }, "—");
  const similarity = prediction.similarities?.[questionId]?.[option] ?? null;
  const limit = cut === null ? null : appliedLimit(distribution, cut);
  const chosen = limit === null ? topOption(distribution) === option : (distribution[option] ?? 0) >= limit;
  const verdict = chosen ? (isReference ? "cell-match" : "cell-mismatch") : "";
  return probabilityCell(distribution[option], { similarity, highlight: chosen, tick: limit, extraClass: verdict });
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
  const toggleLabel = (option, checked) => {
    if (checked) chosen.add(option);
    else chosen.delete(option);
    choices[question.id] = [...chosen];
  };
  const boxes = Object.entries(question.options).map(([option, text], index) => {
    const id = `human-${question.id}-${index}`;
    const input = h("input", { class: "form-check-input", type: "checkbox", id, checked: chosen.has(option), onchange: (event) => toggleLabel(option, event.target.checked) });
    return h("div", { class: "form-check form-check-inline mb-0" }, input, h("label", { class: "form-check-label small", for: id, title: text }, option));
  });
  return h("div", { class: "d-flex flex-wrap", role: "group", "aria-label": `Human labels for ${question.id}` }, boxes);
}
