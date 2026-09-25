/**
 * The Analyze page's setup card: analyst model (free text with suggestions), how many disputed emails to
 * include in full, the two editable prompts (prompt-editor.js), the estimate and the Run button; and the
 * estimate view (upper-bound cost, tokens against the context window, and a warning when it does not fit).
 * Exports: setupCard, estimateView.
 */
import { h, icon } from "./dom.js";
import { money, num } from "./format.js";
import { withHelp } from "./glossary.js";
import { promptEditor } from "./prompt-editor.js";

export function setupCard({ defaults, models, model, disputed, prompts, onModel, onDisputed, onPrompt, onRun }) {
  const suggestions = h("datalist", { id: "analyst-models" }, models.map((id) => h("option", { value: id })));
  const modelInput = h("input", { class: "form-control form-control-sm font-monospace", id: "analyst-model", list: "analyst-models", value: model, spellcheck: "false", autocomplete: "off", oninput: (event) => onModel(event.target.value) });
  const disputedInput = h("input", { class: "form-control form-control-sm", id: "disputed-emails", type: "number", min: 0, max: 100, step: 1, value: disputed, onchange: (event) => onDisputed(event.target) });
  const fields = h("div", { class: "row g-2 mb-3" }, field("col-sm-8", "analyst-model", withHelp("Analyst model", "analyst_model"), modelInput, suggestions), field("col-sm-4", "disputed-emails", withHelp("Disputed emails", "disputed_emails"), disputedInput));
  const editors = [
    promptEditor({ id: "system-prompt", label: "System instructions", value: prompts.system, fallback: defaults.system_prompt, rows: 10, onChange: (value) => onPrompt("system", value) }),
    promptEditor({ id: "user-prompt", label: "Data message", value: prompts.user, fallback: defaults.user_prompt, rows: 8, onChange: (value) => onPrompt("user", value) }),
  ];
  const run = h("button", { class: "btn btn-primary", id: "run-analysis", type: "button", disabled: true, onclick: onRun }, icon("play-fill"), " Run analysis");
  const body = h("div", { class: "card-body" }, fields, h("div", { class: "small text-body-secondary mb-2" }, withHelp("Instructions", "prompts")), editors, h("div", { class: "mb-3", id: "analysis-estimate" }), run);
  return h("div", { class: "card shadow-sm" }, h("div", { class: "card-header fw-semibold" }, icon("sliders"), " Setup"), body);
}

function field(width, id, label, ...controls) {
  return h("div", { class: width }, h("label", { class: "form-label small mb-1", for: id }, label), controls);
}

export function estimateView(estimate) {
  if (estimate === null) return h("span", { class: "small text-body-secondary" }, "Select at least one run to estimate the cost.");
  if (estimate instanceof Error) return h("div", { class: "small text-danger" }, icon("exclamation-circle"), ` ${estimate.message}`);
  const cost = estimate.cost === null ? "no price information" : `≤ ${money(estimate.cost)}`;
  const tokens = `~${num(estimate.input_tokens)} input tokens + up to ${num(estimate.max_output_tokens)} output · `;
  const warning = estimate.fits ? null : h("div", { class: "alert alert-warning py-1 px-2 mt-2 mb-0" }, icon("exclamation-triangle"), " Too long for this model's context window: include fewer disputed emails or runs, or choose a model with a larger context.");
  return h(
    "div",
    { class: "estimate-box small" },
    h("div", { class: "stat-label mb-1" }, withHelp("Estimated cost", "analysis_estimate")),
    h("div", { class: "fw-semibold" }, cost),
    h("div", { class: "text-body-secondary" }, tokens, withHelp(`context ${num(estimate.context_length)}`, "context_window")),
    h("div", { class: "text-body-secondary" }, `${num(estimate.n_emails)} emails · ${num(estimate.n_disputed)} disputed in full · ${estimate.model}`),
    warning,
  );
}
