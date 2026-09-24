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
