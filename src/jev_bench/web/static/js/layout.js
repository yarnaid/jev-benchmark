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
  const badge = document.getElementById("key-badge");
  const serverKey = await api.status().then((status) => status.server_key, () => false);
  if (serverKey) {
    clear(badge, h("span", { class: "badge text-bg-success", title: "The server has OPENROUTER_API_KEY" }, icon("shield-lock"), " server key"));
    return;
  }
  const stored = Boolean(getKey());
  const style = stored ? "btn-success" : "btn-warning";
  clear(badge, h("button", { class: `btn btn-sm ${style}`, type: "button", onclick: openKeyModal }, icon("key"), stored ? " key saved" : " set API key"));
}

function keyModal() {
  const input = h("input", { class: "form-control font-monospace", type: "password", autocomplete: "off", placeholder: "sk-or-v1-…", "aria-label": "OpenRouter API key" });
  const modal = h("div", { class: "modal fade", id: "key-modal", tabindex: "-1", "aria-labelledby": "key-modal-title", "aria-hidden": "true" });
  const hide = () => bootstrap.Modal.getOrCreateInstance(modal).hide();
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
  const note = "The server has no OPENROUTER_API_KEY. A key entered here is stored only in this browser (localStorage) and is sent only when you start a run or a generation.";
  modal.append(
    h(
      "div",
      { class: "modal-dialog modal-dialog-centered" },
      h(
        "div",
        { class: "modal-content" },
        h("div", { class: "modal-header" }, h("h5", { class: "modal-title", id: "key-modal-title" }, icon("key"), " OpenRouter API key"), h("button", { type: "button", class: "btn-close", "data-bs-dismiss": "modal", "aria-label": "Close" })),
        h("div", { class: "modal-body" }, h("p", { class: "small text-body-secondary" }, note), input),
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
