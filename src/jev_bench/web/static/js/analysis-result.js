/**
 * The Analyze page's result card and history table. The card shows one analysis: status, model, the
 * legend of run names (R1 = …), live progress (characters received) or duration, tokens and cost, the
 * rendered Markdown report (a live preview while it runs; e001… refs link to the Explorer), copy and
 * download buttons, and the prompts that were sent (rendered only when opened; `promptsOpen` keeps them
 * open across the re-renders of a running analysis).
 * Exports: resultCard, historyTable.
 */
import { explorerHref } from "./analysis-links.js";
import { h, icon } from "./dom.js";
import { duration, money, num, shortModel, when } from "./format.js";
import { withHelp } from "./glossary.js";
import { renderMarkdown } from "./markdown-render.js";
import { emptyState, statusBadge } from "./widgets.js";

export function resultCard(view, { runLabel, onCancel, onCopy, onDownload, promptsOpen = false, onPromptsToggle = () => {} }) {
  if (!view) return emptyState("No analysis yet: choose the runs, check the estimate and start one.", "stars");
  const { meta, progress } = view;
  const running = meta.status === "running";
  const actions = running
    ? h("button", { class: "btn btn-outline-danger btn-sm ms-auto", type: "button", onclick: () => onCancel(meta.id) }, icon("stop-fill"), " Cancel")
    : h("div", { class: "btn-group btn-group-sm ms-auto" }, h("button", { class: "btn btn-outline-secondary", type: "button", disabled: !meta.result, onclick: () => onCopy(meta) }, icon("clipboard"), " Copy"), h("button", { class: "btn btn-outline-secondary", type: "button", disabled: !meta.result, onclick: () => onDownload(meta) }, icon("download"), " .md"));
  const header = h("div", { class: "card-header d-flex flex-wrap align-items-center gap-2" }, icon("stars"), h("span", { class: "fw-semibold" }, shortModel(meta.model)), statusBadge(meta.status), h("span", { class: "small text-body-secondary" }, when(meta.created_at)), actions);
  const body = h("div", { class: "card-body" }, legend(meta, runLabel), stats(meta, progress), meta.error ? h("div", { class: "alert alert-danger py-1 px-2 small" }, icon("exclamation-triangle"), ` ${meta.error}`) : null, report(meta, running), promptsSent(meta, promptsOpen, onPromptsToggle));
  return h("div", { class: "card shadow-sm analysis-card" }, header, body);
}

function legend(meta, runLabel) {
  const item = (id, index) => h("span", { class: "badge bg-body-tertiary text-body border fw-normal" }, h("span", { class: "fw-semibold" }, `R${index + 1}`), ` ${runLabel(id)}`);
  return h("div", { class: "d-flex flex-wrap align-items-center gap-1 mb-2 small" }, h("span", { class: "text-body-secondary me-1" }, withHelp("Runs", "run_names")), meta.run_ids.map(item));
}

function stats(meta, progress) {
  const cost = meta.status === "running" ? "—" : `${meta.cost_estimated ? "≈ " : ""}${money(meta.cost)}`;
  const time = progress ? `${duration(progress.elapsed_s)} · ${num(progress.done)} characters` : duration(meta.duration_s);
  const items = [
    ["Emails", `${num(meta.n_emails)} (${num(meta.n_disputed)} in full)`],
    ["Time", time],
    ["Tokens in / out", `${num(meta.input_tokens)} / ${num(meta.output_tokens)}`],
    ["Cost", cost],
  ];
  return h("div", { class: "d-flex flex-wrap gap-3 small text-body-secondary mb-3" }, items.map(([label, value]) => h("span", {}, `${label}: `, h("span", { class: "text-body fw-semibold" }, value))));
}

function report(meta, running) {
  if (meta.result) return h("div", { class: `markdown-body${running ? " is-streaming" : ""}` }, markdownOrText(meta));
  if (running) return h("p", { class: "text-body-secondary" }, h("span", { class: "spinner-border spinner-border-sm me-2", role: "status" }), "Waiting for the first words… (the analyst reads everything first)");
  return h("p", { class: "text-body-secondary" }, "No text was returned.");
}

function markdownOrText(meta) {
  try {
    return renderMarkdown(meta.result, { refHref: (ref) => explorerHref(meta, ref) });
  } catch (error) {
    console.warn("report shown as plain text", error);
    return h("pre", { class: "md-code" }, meta.result);
  }
}

function promptsSent(meta, open, onToggle) {
  const details = h("details", { class: "mt-3 small", open }, h("summary", { class: "text-body-secondary" }, icon("file-earmark-text"), " Prompts sent to the analyst"));
  let filled = false;
  const fill = () => {
    if (details.open && !filled) fillPrompts(details, meta);
    filled ||= details.open;
  };
  details.addEventListener("toggle", () => {
    fill();
    onToggle(details.open);
  });
  fill();
  return details;
}

function fillPrompts(details, meta) {
  const block = (title, text) => [h("div", { class: "fw-semibold mt-2" }, title), h("pre", { class: "md-code prompt-sent" }, text)];
  details.append(...block("System", meta.system_prompt ?? ""), ...block("User", meta.user_prompt ?? ""));
}

export function historyTable(views, { selectedId, onSelect }) {
  if (!views.length) return null;
  const head = ["Started", "Analyst", "Runs", "Emails", "Status", "Cost", "Duration"].map((text) => h("th", { class: "small" }, text));
  const rows = views.map(({ meta }) =>
    h(
      "tr",
      { class: `clickable${meta.id === selectedId ? " table-active" : ""}`, tabindex: 0, onclick: () => onSelect(meta.id), onkeydown: (event) => event.key === "Enter" && onSelect(meta.id) },
      h("td", { class: "small text-nowrap" }, when(meta.created_at)),
      h("td", { class: "small" }, shortModel(meta.model)),
      h("td", { class: "small" }, num(meta.run_ids.length)),
      h("td", { class: "small" }, num(meta.n_emails)),
      h("td", {}, statusBadge(meta.status)),
      h("td", { class: "small" }, meta.status === "running" ? "—" : money(meta.cost)),
      h("td", { class: "small" }, duration(meta.duration_s)),
    ),
  );
  return h("div", { class: "card shadow-sm" }, h("div", { class: "card-header fw-semibold" }, icon("clock-history"), " Past analyses"), h("div", { class: "table-responsive" }, h("table", { class: "table table-sm table-hover mb-0" }, h("thead", {}, h("tr", {}, head)), h("tbody", {}, rows))));
}
