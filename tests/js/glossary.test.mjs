// Run with `node --test tests/js/`. Exercises the data and lookups of src/jev_bench/web/static/js/glossary.js.
import assert from "node:assert/strict";
import { readFileSync, readdirSync } from "node:fs";
import test from "node:test";

const { GLOSSARY, describe } = await import("../../src/jev_bench/web/static/js/glossary.js");

const JS_DIR = new URL("../../src/jev_bench/web/static/js/", import.meta.url);
const entries = GLOSSARY.flatMap((group) => group.entries);

test("every glossary key is unique", () => {
  const keys = entries.map(([key]) => key);
  assert.equal(new Set(keys).size, keys.length);
});

test("every entry has a label and a tooltip-sized text", () => {
  for (const [key, label, text] of entries) {
    assert.ok(label.length > 0, key);
    assert.ok(text.length >= 30 && text.length <= 300, `${key}: ${text.length} characters`);
  }
});

test("describe returns the entry and rejects unknown keys", () => {
  assert.equal(describe("jaccard").label, "Jaccard");
  assert.throws(() => describe("nope"), /glossary has no entry nope/);
});

test("every key the UI references exists in the glossary", () => {
  const known = new Set(entries.map(([key]) => key));
  const referenced = readdirSync(JS_DIR)
    .filter((name) => name.endsWith(".js"))
    .flatMap((name) => [...readFileSync(new URL(name, JS_DIR), "utf8").matchAll(/(?:withHelp\([^)]*?,|helpIcon\()\s*"([a-z_]+)"\)/g)].map((match) => [name, match[1]]));
  const missing = referenced.filter(([, key]) => !known.has(key));
  assert.deepEqual(missing, []);
});
