/**
 * Shared page chrome: navbar (pages, API-key badge, theme toggle), API-key dialog (OpenRouter key and
 * optional Hugging Face token, with how to get each), toasts, the delegated Bootstrap tooltips behind every (?) icon, and a job starter that opens the
 * key dialog when the server asks for a key.
 * THEME_EVENT fires on window after the light/dark theme is toggled.
 * Exports: THEME_EVENT, initLayout, keySteps, openKeyModal, toastError, toastSuccess, startJob.
 */
import { api, needsKey } from "./api.js";
import { clear, h, icon } from "./dom.js";
import { forgetHfToken, forgetKey, getKey, HF_STEPS, isHfToken, KEY_EVENT, KEY_STEPS, setHfToken, setKey } from "./key.js";
import { readPref, writePref } from "./storage.js";

export const THEME_EVENT = "jev-bench:theme-changed";

const PAGES = [
  ["/", "Benchmark", "speedometer2"],
  ["/generations.html", "Generations", "envelope-paper"],
  ["/explorer.html", "Explorer", "search"],
  ["/runs.html", "Runs", "clock-history"],
  ["/analyze.html", "Analyze", "stars"],
  ["/help.html", "Help", "question-circle"],
];

export async function initLayout() {
  applyTheme(readPref("theme"));
  const toasts = h("div", { class: "toast-container position-fixed bottom-0 end-0 p-3", id: "toasts" });
  document.body.prepend(navbar(), keyModal(), toasts);
  window.addEventListener(KEY_EVENT, () => refreshKeyBadge());
  new bootstrap.Tooltip(document.body, { selector: '[data-bs-toggle="tooltip"]', trigger: "hover focus" });
  await refreshKeyBadge();
}

function navbar() {
  const here = location.pathname === "/index.html" ? "/" : location.pathname;
  const links = PAGES.map(([href, label, name]) =>
    h("li", { class: "nav-item" }, h("a", { class: `nav-link${href === here ? " active" : ""}`, href }, icon(name), h("span", { class: "nav-label" }, ` ${label}`))),
  );
  const theme = h("button", { class: "btn btn-sm btn-outline-secondary", type: "button", title: "Toggle theme", "aria-label": "Toggle theme", onclick: toggleTheme }, icon("circle-half"));
  return h(
    "nav",
    { class: "navbar navbar-expand jb-nav mb-3", "data-bs-theme": "dark" },
    h(
      "div",
      { class: "container-fluid" },
      h("a", { class: "navbar-brand fw-semibold", href: "/" }, icon("bar-chart-steps"), " jev-bench"),
      h("ul", { class: "navbar-nav me-auto flex-wrap" }, links),
      h("div", { class: "d-flex gap-2 align-items-center" }, h("span", { id: "key-badge" }), theme),
    ),
  );
}

async function refreshKeyBadge() {
  const serverKey = await api.status().then((status) => status.server_key, () => false);
  clear(document.getElementById("key-badge"), keyButton(Boolean(getKey()), serverKey));
}

function keyButton(stored, serverKey) {
  const [style, name, text, title] = stored
    ? ["btn-success", "key-fill", " your key", serverKey ? "Your key overrides the server key for jobs you start" : "Your key is used for jobs you start"]
    : serverKey
      ? ["btn-outline-success", "shield-lock", " server key", "The server's key is used; click to use your own key instead"]
      : ["btn-warning", "key", " set API key", "No key yet: runs, generations and analyses need an OpenRouter key"];
  return h("button", { class: `btn btn-sm ${style}`, type: "button", title, onclick: openKeyModal }, icon(name), text);
}

function keyModal() {
  const input = h("input", { class: "form-control font-monospace", type: "password", autocomplete: "off", placeholder: "sk-or-v1-…", "aria-label": "OpenRouter API key" });
  const modal = h("div", { class: "modal fade", id: "key-modal", tabindex: "-1", "aria-labelledby": "key-modal-title", "aria-hidden": "true" });
  const hide = () => {
    document.activeElement?.blur();
    bootstrap.Modal.getOrCreateInstance(modal).hide();
  };
  const hfInput = h("input", { class: "form-control font-monospace", type: "password", autocomplete: "off", placeholder: "hf_…", id: "hf-token-input", "aria-label": "Hugging Face token" });
  const save = () => {
    const [key, token] = [input.value.trim(), hfInput.value.trim()];
    if (!key && !token) return;
    if (token && !isHfToken(token)) {
      toastError("That is not a Hugging Face token (it starts with hf_). Nothing was saved.");
      return;
    }
    if (key) setKey(key);
    if (token) setHfToken(token);
    input.value = "";
    hfInput.value = "";
    hide();
  };
  for (const field of [input, hfInput]) {
    field.addEventListener("keydown", (event) => {
      if (event.key === "Enter") save();
    });
  }
  const hfNote = "Optional, for the Kev column only: a Hugging Face token raises Kev's free GPU quota. It is stored only in this browser, sent only when you start a run, and overrides the server's HF_TOKEN.";
  const note = "A key entered here is stored only in this browser (localStorage) and sent only when you start a run, a generation or an analysis. It overrides the server's key, if the server has one.";
  modal.append(
    h(
      "div",
      { class: "modal-dialog modal-dialog-centered" },
      h(
        "div",
        { class: "modal-content" },
        h("div", { class: "modal-header" }, h("h5", { class: "modal-title", id: "key-modal-title" }, icon("key"), " OpenRouter API key"), h("button", { type: "button", class: "btn-close", "data-bs-dismiss": "modal", "aria-label": "Close" })),
        h(
          "div",
          { class: "modal-body" },
          h("p", { class: "small text-body-secondary" }, note),
          input,
          keyHowTo(),
          h("hr"),
          h("label", { class: "form-label fw-semibold small", for: "hf-token-input" }, "Hugging Face token (optional)"),
          h("p", { class: "small text-body-secondary" }, hfNote),
          hfInput,
          h("details", { class: "mt-3 small" }, h("summary", { class: "fw-semibold" }, icon("question-circle"), " How to get a Hugging Face token"), keySteps(HF_STEPS)),
        ),
        h(
          "div",
          { class: "modal-footer" },
          h("button", { type: "button", class: "btn btn-outline-danger me-auto", onclick: () => { forgetKey(); forgetHfToken(); hide(); } }, icon("trash"), " Forget keys"),
          h("button", { type: "button", class: "btn btn-primary", onclick: save }, icon("check2"), " Save"),
        ),
      ),
    ),
  );
  return modal;
}

export function keySteps(steps = KEY_STEPS) {
  const items = steps.map(([label, href, text]) => h("li", { class: "mb-1" }, h("a", { href, target: "_blank", rel: "noopener noreferrer" }, label, " ", icon("box-arrow-up-right")), h("span", { class: "d-block text-body-secondary" }, text)));
  return h("ol", { class: "mt-2 mb-0 ps-3" }, items);
}

function keyHowTo() {
  return h("details", { class: "mt-3 small" }, h("summary", { class: "fw-semibold" }, icon("question-circle"), " How to get an OpenRouter key"), keySteps());
}

export function openKeyModal() {
  bootstrap.Modal.getOrCreateInstance(document.getElementById("key-modal")).show();
}

function toast(text, variant) {
  const element = h(
    "div",
    { class: `toast align-items-center text-bg-${variant} border-0`, role: "alert", "aria-live": "assertive", "aria-atomic": "true" },
    h("div", { class: "d-flex" }, h("div", { class: "toast-body" }, text), h("button", { type: "button", class: "btn-close btn-close-white me-2 m-auto", "data-bs-dismiss": "toast", "aria-label": "Close" })),
  );
  document.getElementById("toasts").append(element);
  element.addEventListener("hidden.bs.toast", () => element.remove());
  bootstrap.Toast.getOrCreateInstance(element, { delay: 8000 }).show();
}

export function toastError(error) {
  toast(error instanceof Error ? error.message : String(error), "danger");
}

export function toastSuccess(text) {
  toast(text, "success");
}

export async function startJob(action) {
  try {
    return await action();
  } catch (error) {
    if (needsKey(error)) openKeyModal();
    else toastError(error);
    return null;
  }
}

function applyTheme(stored) {
  const prefersDark = window.matchMedia("(prefers-color-scheme: dark)").matches;
  document.documentElement.dataset.bsTheme = stored ?? (prefersDark ? "dark" : "light");
}

function toggleTheme() {
  const next = document.documentElement.dataset.bsTheme === "dark" ? "light" : "dark";
  writePref("theme", next);
  applyTheme(next);
  window.dispatchEvent(new Event(THEME_EVENT));
}
