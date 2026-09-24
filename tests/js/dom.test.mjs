// Manual gate (controller ruling, not part of the pytest suite): run with `node --test tests/js/`.
// Exercises src/jev_bench/web/static/js/dom.js against a tiny mock DOM defined below.
import assert from "node:assert/strict";
import test from "node:test";

class MockNode {}

class MockElement extends MockNode {
  constructor(tag) {
    super();
    this.tag = tag;
    this._attrs = new Map();
    this.dataset = {};
    this.children = [];
    this._listeners = {};
  }

  setAttribute(name, value) {
    this._attrs.set(name, String(value));
  }

  getAttribute(name) {
    return this._attrs.has(name) ? this._attrs.get(name) : null;
  }

  hasAttribute(name) {
    return this._attrs.has(name);
  }

  addEventListener(event, handler) {
    (this._listeners[event] ??= []).push(handler);
  }

  append(...nodes) {
    this.children.push(...nodes);
  }

  replaceChildren(...nodes) {
    this.children = [...nodes];
  }
}

class MockText extends MockNode {
  constructor(text) {
    super();
    this.text = text;
  }
}

globalThis.Node = MockNode;
globalThis.document = {
  createElement: (tag) => new MockElement(tag),
  createTextNode: (text) => new MockText(text),
};

const { h, clear, icon } = await import("../../src/jev_bench/web/static/js/dom.js");

test("a function on* attribute becomes an event listener, not an HTML attribute", () => {
  const handler = () => {};
  const el = h("button", { onclick: handler });
  assert.deepEqual(el._listeners.click, [handler]);
  assert.equal(el.hasAttribute("onclick"), false);
});

for (const name of ["onclick", "ONCLICK", "onClick"]) {
  test(`a string ${name} attribute throws TypeError instead of becoming an inline handler`, () => {
    assert.throws(() => h("button", { [name]: "doStuff()" }), TypeError);
  });
}

for (const name of ["HREF", "Src"]) {
  test(`an unsafe URL is dropped whatever the attribute name's case: ${name}`, () => {
    const el = h("a", { [name]: "javascript:alert(1)" });
    assert.equal(el.hasAttribute(name), false);
  });
}

const UNSAFE_HREFS = [
  "javascript:alert(1)",
  " JaVaScRiPt:alert(1)",
  "java\tscript:x",
  "data:text/html,<script>alert(1)</script>",
];

for (const value of UNSAFE_HREFS) {
  test(`unsafe href is dropped, not set: ${JSON.stringify(value)}`, () => {
    const el = h("a", { href: value });
    assert.equal(el.hasAttribute("href"), false);
  });
}

const SAFE_HREFS = ["https://x", "/a", "#frag", "mailto:a@b"];

for (const value of SAFE_HREFS) {
  test(`safe href is kept unchanged: ${JSON.stringify(value)}`, () => {
    const el = h("a", { href: value });
    assert.equal(el.getAttribute("href"), value);
  });
}

test("string children become text nodes, never parsed as markup", () => {
  const el = h("div", {}, "hello", "world");
  assert.equal(el.children.length, 2);
  assert.ok(el.children[0] instanceof MockText);
  assert.equal(el.children[0].text, "hello");
  assert.equal(el.children[1].text, "world");
});

test("null and false attributes are skipped; true gives an empty attribute", () => {
  const el = h("input", { disabled: true, hidden: false, title: null, type: "text" });
  assert.equal(el.getAttribute("disabled"), "");
  assert.equal(el.hasAttribute("hidden"), false);
  assert.equal(el.hasAttribute("title"), false);
  assert.equal(el.getAttribute("type"), "text");
});

test("clear() replaces existing children", () => {
  const el = h("div", {}, "old");
  clear(el, "new");
  assert.equal(el.children.length, 1);
  assert.equal(el.children[0].text, "new");
});

test("icon() builds a bootstrap-icon element", () => {
  const el = icon("key");
  assert.equal(el.tag, "i");
  assert.equal(el.getAttribute("class"), "bi bi-key");
});
