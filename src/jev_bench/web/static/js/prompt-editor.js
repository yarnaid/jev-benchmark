/**
 * One editable prompt template of the Analyze page: a labelled textarea prefilled with the saved edit or
 * the default, an "edited" badge, "Reset to default", and placeholder chips that insert $name at the
 * cursor. `onChange` receives the text, or null when it equals the default (so the server's default is
 * used and nothing is stored).
 * Exports: PLACEHOLDER_HELP, promptEditor.
 */
import { h, icon } from "./dom.js";

export const PLACEHOLDER_HELP = {
  generations: "The selected generations and how many emails each has.",
  runs: "Each run's model, speed, cost and tokens, named R1, R2, …",
  questions: "Every question with its type and options.",
  report: "The full comparison report (agreement, κ, Brier, …) as compact JSON.",
  emails: "One table per question with every run's answer for every email.",
  disputed: "The emails the runs disagree on most, in full.",
};

export function promptEditor({ id, label, value, fallback, rows = 8, onChange }) {
  const area = h("textarea", { class: "form-control font-monospace prompt-area", id, rows, spellcheck: "false" }, value ?? fallback);
  const badge = h("span", { class: "badge text-bg-warning" }, "edited");
  const changed = () => {
    const edited = area.value !== fallback;
    badge.classList.toggle("d-none", !edited);
    onChange(edited ? area.value : null);
  };
  area.addEventListener("input", changed);
  badge.classList.toggle("d-none", area.value === fallback);
  const reset = h("button", { class: "btn btn-link btn-sm p-0 ms-auto", type: "button", onclick: () => replace(area, fallback, changed) }, icon("arrow-counterclockwise"), " Reset to default");
  const heading = h("div", { class: "d-flex align-items-center gap-2 mb-1" }, h("label", { class: "form-label small fw-semibold mb-0", for: id }, label), badge, reset);
  return h("div", { class: "mb-3" }, heading, area, chips(area, changed));
}

function chips(area, changed) {
  const chip = ([name, text]) => h("button", { class: "btn btn-outline-secondary btn-sm font-monospace placeholder-chip", type: "button", "data-bs-toggle": "tooltip", "data-bs-title": text, "aria-label": `Insert $${name}: ${text}`, onclick: () => insert(area, `$${name}`, changed) }, `$${name}`);
  return h("div", { class: "d-flex flex-wrap gap-1 mt-1" }, Object.entries(PLACEHOLDER_HELP).map(chip));
}

function insert(area, text, changed) {
  area.setRangeText(text, area.selectionStart, area.selectionEnd, "end");
  area.focus();
  changed();
}

function replace(area, text, changed) {
  area.value = text;
  changed();
}
