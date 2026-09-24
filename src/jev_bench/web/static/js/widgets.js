/**
 * Reusable widgets built with h(): status badges, progress bars, empty states and a checkbox dropdown
 * ("checklist") taking items [{ value, text, hint }] and calling onChange(selectedValues).
 * Exports: statusBadge, progressBar, emptyState, checklist.
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
