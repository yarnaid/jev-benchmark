/**
 * Reusable widgets built with h(): status badges, progress bars, empty states, a checkbox dropdown
 * ("checklist") taking items [{ value, text, hint }] and calling onChange(selectedValues), and a
 * multi-label threshold slider (50-100 % of the top probability, step 5) calling onChange(fraction)
 * after a debounce. thresholdRange widens that range so any configured threshold in (0, 1] is an exact
 * slider position: the minimum drops below 50 % when needed (never to 0) and the step becomes 1 % when
 * the value is not a multiple of 5 %.
 * Exports: statusBadge, progressBar, emptyState, checklist, thresholdRange, thresholdSlider.
 */
import { h, icon } from "./dom.js";

const STATUS_STYLES = { running: "primary", completed: "success", cancelled: "secondary", failed: "danger", interrupted: "warning" };

export function statusBadge(status) {
  return h("span", { class: `badge text-bg-${STATUS_STYLES[status] ?? "secondary"}` }, status);
}

export function progressBar(done, total, { animated = false } = {}) {
  const percent = total ? Math.round((100 * done) / total) : 0;
  const bar = h(
    "div",
    {
      class: `progress-bar${animated ? " progress-bar-striped progress-bar-animated" : ""}`,
      style: `width: ${percent}%`,
      role: "progressbar",
      "aria-valuenow": percent,
      "aria-valuemin": 0,
      "aria-valuemax": 100,
    },
    `${done}/${total}`,
  );
  return h("div", { class: "progress" }, bar);
}

export function emptyState(text, iconName = "inbox") {
  return h("div", { class: "text-center text-body-secondary py-5" }, h("div", { class: "fs-1" }, icon(iconName)), h("p", { class: "mb-0" }, text));
}

export function checklist({ label, items, selected, onChange }) {
  const chosen = new Set(selected);
  const summary = h("span", {});
  const updateSummary = () => {
    summary.textContent = chosen.size ? `${label}: ${chosen.size} selected` : `${label}: none`;
  };
  const toggle = (value, checked) => {
    if (checked) chosen.add(value);
    else chosen.delete(value);
    updateSummary();
    onChange([...chosen]);
  };
  const options = items.map((item, index) => {
    const id = `${label.toLowerCase().replace(/\W+/g, "-")}-${index}`;
    const input = h("input", { class: "form-check-input mt-1", type: "checkbox", id, checked: chosen.has(item.value), onchange: (event) => toggle(item.value, event.target.checked) });
    const text = h("span", {}, item.text, item.hint ? h("small", { class: "d-block text-body-secondary" }, item.hint) : null);
    return h("li", {}, h("label", { class: "dropdown-item d-flex gap-2 align-items-start", for: id }, input, text));
  });
  updateSummary();
  const empty = h("li", { class: "dropdown-item-text text-body-secondary" }, "Nothing to select yet");
  return h(
    "div",
    { class: "dropdown" },
    h("button", { class: "btn btn-outline-primary btn-sm dropdown-toggle", type: "button", "data-bs-toggle": "dropdown", "data-bs-auto-close": "outside", "aria-expanded": "false" }, icon("collection"), " ", summary),
    h("ul", { class: "dropdown-menu shadow-sm checklist-menu" }, options.length ? options : empty),
  );
}

export function thresholdRange(value) {
  const percent = Math.round(value * 100);
  const min = Math.max(1, Math.min(50, percent - (percent % 5)));
  return { min, max: 100, step: percent % 5 === 0 ? 5 : 1, value: percent };
}

export function thresholdSlider({ value, onChange, delayMs = 300 }) {
  const percent = (fraction) => `${Math.round(fraction * 100)}%`;
  const output = h("output", { class: "small font-monospace text-nowrap" }, percent(value));
  let timer = null;
  const oninput = (event) => {
    const next = Number(event.target.value) / 100;
    output.textContent = percent(next);
    clearTimeout(timer);
    timer = setTimeout(() => onChange(next), delayMs);
  };
  const input = h("input", { type: "range", class: "form-range threshold-range", ...thresholdRange(value), "aria-label": "Label threshold", oninput });
  const label = h("span", { class: "small text-nowrap" }, icon("sliders"), " Label threshold");
  const hint = "A label counts as applied when its probability is at least this share of the most probable label";
  return h("div", { class: "d-flex align-items-center gap-2", title: hint }, label, input, output);
}
