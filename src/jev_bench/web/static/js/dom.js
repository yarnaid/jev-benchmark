/**
 * Tiny DOM builder: string children always become text nodes, so untrusted text is never parsed as
 * HTML. `on*` attributes must be functions (a string value throws, never becomes an inline handler
 * attribute) and URL attributes (href, src, action, formaction, xlink:href) are scheme-checked,
 * with an unsafe scheme (javascript:, data:, vbscript:, ...) silently dropped rather than set.
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

const URL_ATTRS = new Set(["href", "src", "action", "formaction", "xlink:href"]);
const SAFE_URL_PROTOCOLS = new Set(["http:", "https:", "mailto:", "tel:"]);

function isSafeUrl(value) {
  try {
    return SAFE_URL_PROTOCOLS.has(new URL(value, "http://relative.invalid").protocol);
  } catch {
    return false;
  }
}

function setAttribute(element, name, value) {
  if (value === null || value === undefined || value === false) return;
  const key = name.toLowerCase();
  if (key.startsWith("on")) {
    if (typeof value !== "function") throw new TypeError(`h(): ${name} must be a function`);
    element.addEventListener(key.slice(2), value);
    return;
  }
  if (name === "dataset") {
    Object.assign(element.dataset, value);
    return;
  }
  if (URL_ATTRS.has(key) && !isSafeUrl(String(value))) return;
  element.setAttribute(name, value === true ? "" : String(value));
}

function appendChildren(element, children) {
  for (const child of children.flat(Infinity)) {
    if (child === null || child === undefined || child === false) continue;
    element.append(child instanceof Node ? child : document.createTextNode(String(child)));
  }
}
