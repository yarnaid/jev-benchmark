### Task 22: UI shell — DOM builder, storage, key, API client, layout, widgets, styles, static contract test

**Files:**
- Create in `src/jev_bench/web/static/`: `css/app.css`, `js/dom.js`, `js/format.js`, `js/storage.js`,
  `js/key.js`, `js/api.js`, `js/layout.js`, `js/widgets.js`
- Test: `tests/test_web_static.py`

**Interfaces:**
- Consumes the HTTP API of Tasks 19–21 (paths and JSON shapes listed there).
- Produces ES modules for the page scripts of Tasks 23–24:
  - `dom.js`:
    - `h(tag, attrs = {}, ...children) -> Element`. Every string child becomes a text node;
      `on<event>` function attributes become listeners; `dataset` is merged; `true` gives an empty
      attribute; `null`, `undefined` and `false` are skipped;
    - `clear(element, ...children) -> Element`;
    - `icon(name) -> <i class="bi bi-name">`.
  - `format.js`: `money(usd)`, `duration(seconds)`, `pct(x)`, `fixed(x, digits = 3)`, `num(x)`,
    `when(iso)`, `perMillion(usd)`, `shortModel(id)`. Each returns `"—"` for `null` / `undefined`.
  - `storage.js`: `readPref(name, fallback = null)`, `writePref(name, value)`, `removePref(name)`. They
    are JSON-encoded under the `jev-bench.` prefix and never throw.
  - `key.js`: `KEY_EVENT`, `getKey()`, `setKey(value)`, `forgetKey()`.
  - `api.js`:
    - `ApiError` (`.status`);
    - `needsKey(error)`;
    - `api.status()`, `catalog()`, `generations()`, `generation(id)`, `createGeneration(body)`,
      `cancelGeneration(id)`, `runs(generationIds = [])`, `run(id)`, `createRun(body)`, `cancelRun(id)`,
      `compare(runIds)`, `emails(generationIds, runIds = [])`, `email(id, runIds = [])`,
      `putLabel(id, answers)`;
    - only `createGeneration` and `createRun` send `X-OpenRouter-Key`.
  - `layout.js`:
    - `initLayout()`: navbar with page links, key badge/button and theme toggle, plus the key modal and
      the toast container;
    - `openKeyModal()`, `toastError(error)`, `toastSuccess(text)`;
    - `startJob(action)`: runs `action`; on a missing-key 400 it opens the key modal, on other errors it
      shows a toast; returns `null` on failure.
  - `widgets.js`:
    - `statusBadge(status)`;
    - `progressBar(done, total, { animated })`;
    - `emptyState(text, iconName = "inbox")`;
    - `checklist({ label, items: [{value, text, hint}], selected, onChange })`: a dropdown of checkboxes
      that calls `onChange(values)`.

Rules (Global Constraints):
- no `innerHTML`, `outerHTML`, `insertAdjacentHTML` or `document.write`;
- no inline `<script>`, and no `on…=` attributes in HTML;
- every jsDelivr asset carries `integrity="sha384-…"` and `crossorigin="anonymous"`, with the version
  pinned via `@x.y.z`;
- `bootstrap` and `Chart` are globals from the CDN bundles.

- [ ] **Step 1: Write the failing static contract test**

`tests/test_web_static.py`:
```python
"""Contract tests for the static UI: no HTML-injection sinks, no inline scripts, pinned CDN assets with SRI,
and every local reference (script, stylesheet, ES-module import) resolves to an existing file."""

import re
from html.parser import HTMLParser
from pathlib import Path

import pytest

from jev_bench.web.app import STATIC_DIR

FORBIDDEN_SINKS = re.compile(r"\binnerHTML\b|\bouterHTML\b|insertAdjacentHTML|document\.write")
IMPORT = re.compile(r"""from\s+["'](\./[^"']+)["']""")
SHELL_MODULES = ("dom.js", "format.js", "storage.js", "key.js", "api.js", "layout.js", "widgets.js")


class TagCollector(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.tags: list[tuple[str, dict[str, str | None]]] = []
        self.inline_script = False
        self._in_script = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.tags.append((tag, dict(attrs)))
        self._in_script = tag == "script"

    def handle_endtag(self, tag: str) -> None:
        if tag == "script":
            self._in_script = False

    def handle_data(self, data: str) -> None:
        if self._in_script and data.strip():
            self.inline_script = True


def _pages() -> list[Path]:
    return sorted(STATIC_DIR.glob("*.html"))


def _parse(page: Path) -> TagCollector:
    collector = TagCollector()
    collector.feed(page.read_text(encoding="utf-8"))
    return collector


def test_shell_modules_exist() -> None:
    assert [name for name in SHELL_MODULES if not (STATIC_DIR / "js" / name).exists()] == []
    assert (STATIC_DIR / "css" / "app.css").exists()


def test_javascript_has_no_html_injection_sinks() -> None:
    offenders = [path.name for path in STATIC_DIR.glob("js/*.js") if FORBIDDEN_SINKS.search(path.read_text(encoding="utf-8"))]
    assert offenders == []


def test_module_imports_resolve() -> None:
    missing = [
        f"{path.name} -> {target}"
        for path in STATIC_DIR.glob("js/*.js")
        for target in IMPORT.findall(path.read_text(encoding="utf-8"))
        if not (path.parent / target).exists()
    ]
    assert missing == []


@pytest.mark.parametrize("page", _pages(), ids=lambda path: path.name)
def test_pages_follow_the_csp_contract(page: Path) -> None:
    collector = _parse(page)
    assert collector.inline_script is False
    for tag, attrs in collector.tags:
        assert not any(name.startswith("on") for name in attrs), f"inline handler in <{tag}>"
        url = attrs.get("src") or attrs.get("href") or ""
        if tag == "script":
            assert url, "inline <script> without src"
        if "cdn.jsdelivr.net" in url:
            assert re.search(r"@\d+\.\d+\.\d+/", url), f"unpinned CDN asset {url}"
            assert (attrs.get("integrity") or "").startswith("sha384-"), f"missing SRI on {url}"
            assert attrs.get("crossorigin") == "anonymous"
        if url.startswith("/") and not url.startswith("//"):
            assert (STATIC_DIR / url.lstrip("/")).exists(), f"missing local asset {url}"
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_web_static.py -v`
Expected: `test_shell_modules_exist` FAILS (the modules do not exist yet); the others pass trivially on the Task 19 shell.

- [ ] **Step 3: Write the styles and modules**

`src/jev_bench/web/static/css/app.css`:
```css
:root {
  --jb-font: "Inter", system-ui, -apple-system, "Segoe UI", sans-serif;
  --jb-mono: "JetBrains Mono", ui-monospace, SFMono-Regular, Menlo, monospace;
}

body {
  font-family: var(--jb-font);
  background: var(--bs-body-bg);
}

code,
.font-monospace,
.stat-value {
  font-family: var(--jb-mono) !important;
}

.nav-link.active {
  font-weight: 600;
}

.column-card .stat-label {
  font-size: 0.72rem;
  color: var(--bs-secondary-color);
  text-transform: uppercase;
  letter-spacing: 0.04em;
}

.column-card .stat-value {
  font-size: 0.9rem;
}

.chart-box {
  position: relative;
  height: 280px;
}

.email-body {
  white-space: pre-wrap;
  word-break: break-word;
  font-family: var(--jb-mono);
  font-size: 0.85rem;
  background: var(--bs-tertiary-bg);
  border-radius: 0.5rem;
  padding: 0.75rem;
  max-height: 40vh;
  overflow: auto;
}

.prob-track {
  background: var(--bs-secondary-bg);
  border-radius: 0.25rem;
  height: 0.5rem;
  min-width: 3rem;
}

.prob-bar {
  height: 100%;
  border-radius: 0.25rem;
  background: var(--bs-primary);
}

.cell-match {
  background: rgba(var(--bs-success-rgb), 0.12) !important;
}

.cell-mismatch {
  background: rgba(var(--bs-danger-rgb), 0.15) !important;
}

.table-sticky thead th {
  position: sticky;
  top: 0;
  z-index: 1;
  background: var(--bs-body-bg);
}

.checklist-menu {
  max-height: 60vh;
  overflow: auto;
  min-width: 18rem;
}

.clickable {
  cursor: pointer;
}

@media (max-width: 575.98px) {
  .navbar .nav-label {
    display: none;
  }
}
```

`src/jev_bench/web/static/js/dom.js`:
```js
/**
 * Tiny DOM builder: string children always become text nodes, so untrusted text is never parsed as HTML.
 * Exports: h, clear, icon.
 */

export function h(tag, attrs = {}, ...children) {
  const element = document.createElement(tag);
  for (const [name, value] of Object.entries(attrs ?? {})) setAttribute(element, name, value);
  appendChildren(element, children);
  return element;
}

export function clear(element, ...children) {
  element.replaceChildren();
  appendChildren(element, children);
  return element;
}

export function icon(name) {
  return h("i", { class: `bi bi-${name}`, "aria-hidden": "true" });
}

function setAttribute(element, name, value) {
  if (value === null || value === undefined || value === false) return;
  if (name.startsWith("on") && typeof value === "function") {
    element.addEventListener(name.slice(2).toLowerCase(), value);
  } else if (name === "dataset") {
    Object.assign(element.dataset, value);
  } else {
    element.setAttribute(name, value === true ? "" : String(value));
  }
}

function appendChildren(element, children) {
  for (const child of children.flat(Infinity)) {
    if (child === null || child === undefined || child === false) continue;
    element.append(child instanceof Node ? child : document.createTextNode(String(child)));
  }
}
```

`src/jev_bench/web/static/js/format.js`:
```js
/**
 * Display formatting helpers; every function returns "—" for null/undefined.
 * Exports: money, duration, pct, fixed, num, when, perMillion, shortModel.
 */

const missing = (value) => value === null || value === undefined;

export function money(usd) {
  if (missing(usd)) return "—";
  return `$${usd < 0.01 ? usd.toFixed(5) : usd.toFixed(4)}`;
}

export function duration(seconds) {
  if (missing(seconds)) return "—";
  if (seconds < 60) return `${seconds.toFixed(1)} s`;
  return `${Math.floor(seconds / 60)} min ${Math.round(seconds % 60)} s`;
}

export const pct = (value) => (missing(value) ? "—" : `${(value * 100).toFixed(1)}%`);
export const fixed = (value, digits = 3) => (missing(value) ? "—" : Number(value).toFixed(digits));
export const num = (value) => (missing(value) ? "—" : Number(value).toLocaleString("en-US"));
export const when = (iso) => (missing(iso) ? "—" : new Date(iso).toLocaleString());
export const perMillion = (usd) => (missing(usd) ? "—" : `$${Number(usd).toFixed(2)}`);
export const shortModel = (id) => (missing(id) ? "—" : id.split("/").pop());
```

`src/jev_bench/web/static/js/storage.js`:
```js
/**
 * Safe localStorage access: JSON values under the "jev-bench." prefix; private windows or blocked
 * storage fall back to defaults and never throw.
 * Exports: readPref, writePref, removePref.
 */

const PREFIX = "jev-bench.";

export function readPref(name, fallback = null) {
  try {
    const raw = localStorage.getItem(PREFIX + name);
    return raw === null ? fallback : JSON.parse(raw);
  } catch {
    return fallback;
  }
}

export function writePref(name, value) {
  try {
    localStorage.setItem(PREFIX + name, JSON.stringify(value));
  } catch (error) {
    console.warn("localStorage unavailable", error);
  }
}

export function removePref(name) {
  try {
    localStorage.removeItem(PREFIX + name);
  } catch (error) {
    console.warn("localStorage unavailable", error);
  }
}
```

`src/jev_bench/web/static/js/key.js`:
```js
/**
 * Browser-side OpenRouter key kept in localStorage, with a change event for the navbar badge.
 * Exports: KEY_EVENT, getKey, setKey, forgetKey.
 */
import { readPref, removePref, writePref } from "./storage.js";

const KEY_PREF = "openrouter-key";
export const KEY_EVENT = "jev-bench:key-changed";

export const getKey = () => readPref(KEY_PREF);

export function setKey(value) {
  writePref(KEY_PREF, value);
  window.dispatchEvent(new Event(KEY_EVENT));
}

export function forgetKey() {
  removePref(KEY_PREF);
  window.dispatchEvent(new Event(KEY_EVENT));
}
```

`src/jev_bench/web/static/js/api.js`:
```js
/**
 * JSON client for the /api endpoints; the browser-stored key is attached only to job-starting calls.
 * Exports: api, ApiError, needsKey.
 */
import { getKey } from "./key.js";

export class ApiError extends Error {
  constructor(status, detail) {
    super(typeof detail === "string" ? detail : JSON.stringify(detail));
    this.status = status;
  }
}

async function request(method, path, { body, withKey = false } = {}) {
  const headers = { Accept: "application/json" };
  if (body !== undefined) headers["Content-Type"] = "application/json";
  const key = withKey ? getKey() : null;
  if (key) headers["X-OpenRouter-Key"] = key;
  const response = await fetch(`/api${path}`, { method, headers, body: body === undefined ? undefined : JSON.stringify(body) });
  const isJson = (response.headers.get("content-type") ?? "").includes("json");
  const payload = isJson ? await response.json() : await response.text();
  if (!response.ok) throw new ApiError(response.status, payload?.detail ?? payload);
  return payload;
}

function query(params) {
  const entries = Object.entries(params).filter(([, value]) => value !== undefined && value !== "");
  const text = new URLSearchParams(entries).toString();
  return text ? `?${text}` : "";
}

const segment = (value) => encodeURIComponent(value);

export const api = {
  status: () => request("GET", "/status"),
  catalog: () => request("GET", "/catalog"),
  generations: () => request("GET", "/generations"),
  generation: (id) => request("GET", `/generations/${segment(id)}`),
  createGeneration: (body) => request("POST", "/generations", { body, withKey: true }),
  cancelGeneration: (id) => request("POST", `/generations/${segment(id)}/cancel`),
  runs: (generationIds = []) => request("GET", `/runs${query({ generations: generationIds.join(",") })}`),
  run: (id) => request("GET", `/runs/${segment(id)}`),
  createRun: (body) => request("POST", "/runs", { body, withKey: true }),
  cancelRun: (id) => request("POST", `/runs/${segment(id)}/cancel`),
  compare: (runIds) => request("GET", `/compare${query({ runs: runIds.join(",") })}`),
  emails: (generationIds, runIds = []) =>
    request("GET", `/emails${query({ generations: generationIds.join(","), runs: runIds.join(",") })}`),
  email: (id, runIds = []) => request("GET", `/emails/${segment(id)}${query({ runs: runIds.join(",") })}`),
  putLabel: (id, answers) => request("PUT", `/labels/${segment(id)}`, { body: { answers } }),
};

export const needsKey = (error) => error instanceof ApiError && error.status === 400 && error.message.includes("API key");
```

`src/jev_bench/web/static/js/layout.js`:
```js
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
```

`src/jev_bench/web/static/js/widgets.js`:
```js
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
  const bar = h("div", { class: `progress-bar${animated ? " progress-bar-striped progress-bar-animated" : ""}`, style: `width: ${percent}%` }, `${done}/${total}`);
  return h("div", { class: "progress", role: "progressbar", "aria-valuenow": percent, "aria-valuemin": 0, "aria-valuemax": 100 }, bar);
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
```

- [ ] **Step 4: Run the static test to verify it passes**

Run: `uv run pytest tests/test_web_static.py -v`
Expected: all PASS.

Run: `uv run ruff check --fix && uv run ruff format && uv run pyright`
Expected: `0 errors`.

- [ ] **Step 5: Commit**

```bash
git add src/jev_bench/web/static tests/test_web_static.py
git commit -m "feat(ui): vanilla JS shell with safe DOM builder, key dialog and static CSP contract test

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```
