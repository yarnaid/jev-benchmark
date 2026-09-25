/**
 * Shared page chrome: navbar (pages, API-key badge, theme toggle), API-key dialog, toasts and a job starter
 * that opens the key dialog when the server asks for a key.
 * Exports: initLayout, openKeyModal, toastError, toastSuccess, startJob.
 */
import { api, needsKey } from "./api.js";
import { clear, h, icon } from "./dom.js";
import { forgetKey, getKey, KEY_EVENT, setKey } from "./key.js";
import { readPref, writePref } from "./storage.js";

const PAGES = [
  ["/", "Benchmark", "speedometer2"],
  ["/generations.html", "Generations", "envelope-paper"],
  ["/explorer.html", "Explorer", "search"],
  ["/runs.html", "Runs", "clock-history"],
];

export async function initLayout() {
  applyTheme(readPref("theme"));
  const toasts = h("div", { class: "toast-container position-fixed bottom-0 end-0 p-3", id: "toasts" });
  document.body.prepend(navbar(), keyModal(), toasts);
  window.addEventListener(KEY_EVENT, () => refreshKeyBadge());
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
    { class: "navbar navbar-expand bg-body-tertiary border-bottom mb-3" },
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
  const save = () => {
    const value = input.value.trim();
    if (!value) return;
    setKey(value);
    input.value = "";
    hide();
  };
  input.addEventListener("keydown", (event) => {
    if (event.key === "Enter") save();
  });
  const note = "A key entered here is stored only in this browser (localStorage) and sent only when you start a run, a generation or an analysis. It overrides the server's key, if the server has one.";
  modal.append(
    h(
      "div",
      { class: "modal-dialog modal-dialog-centered" },
      h(
        "div",
        { class: "modal-content" },
        h("div", { class: "modal-header" }, h("h5", { class: "modal-title", id: "key-modal-title" }, icon("key"), " OpenRouter API key"), h("button", { type: "button", class: "btn-close", "data-bs-dismiss": "modal", "aria-label": "Close" })),
        h("div", { class: "modal-body" }, h("p", { class: "small text-body-secondary" }, note), input, keyHowTo()),
        h(
          "div",
          { class: "modal-footer" },
          h("button", { type: "button", class: "btn btn-outline-danger me-auto", onclick: () => { forgetKey(); hide(); } }, icon("trash"), " Forget"),
          h("button", { type: "button", class: "btn btn-primary", onclick: save }, icon("check2"), " Save"),
        ),
      ),
    ),
  );
  return modal;
}

const KEY_LINKS = [
  ["Create a key", "https://openrouter.ai/keys", "Sign in to OpenRouter, open Keys and create a key (it starts with sk-or-v1-)."],
  ["Add credits", "https://openrouter.ai/settings/credits", "Runs are paid per use from your OpenRouter credits."],
  ["Optional: your provider keys", "https://openrouter.ai/workspaces/default/byok", "Bring your own Anthropic or OpenAI key (BYOK) to be billed by the provider directly."],
];

function keyHowTo() {
  const steps = KEY_LINKS.map(([label, href, text]) => h("li", { class: "mb-1" }, h("a", { href, target: "_blank", rel: "noopener noreferrer" }, label, " ", icon("box-arrow-up-right")), h("span", { class: "d-block text-body-secondary" }, text)));
  return h("details", { class: "mt-3 small" }, h("summary", { class: "fw-semibold" }, icon("question-circle"), " How to get an OpenRouter key"), h("ol", { class: "mt-2 mb-0 ps-3" }, steps));
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
}
