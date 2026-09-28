// Run with `node --test tests/js/`. Exercises which key headers src/jev_bench/web/static/js/api.js attaches.
import assert from "node:assert/strict";
import test from "node:test";

const store = new Map([
  ["jev-bench.openrouter-key", JSON.stringify("sk-or-v1-test")],
  ["jev-bench.hf-token", JSON.stringify("hf_test")],
]);
globalThis.localStorage = { getItem: (key) => store.get(key) ?? null, setItem: (key, value) => store.set(key, value), removeItem: (key) => store.delete(key) };
const sent = [];
globalThis.fetch = async (url, init) => {
  sent.push({ url, headers: init.headers });
  return { ok: true, headers: new Map([["content-type", "application/json"]]), json: async () => ({}) };
};

const { api } = await import("../../src/jev_bench/web/static/js/api.js");

const CASES = [
  ["createRun sends both keys", () => api.createRun({}), { "X-OpenRouter-Key": "sk-or-v1-test", "X-HF-Token": "hf_test" }],
  ["createGeneration sends only the OpenRouter key", () => api.createGeneration({}), { "X-OpenRouter-Key": "sk-or-v1-test" }],
  ["createAnalysis sends only the OpenRouter key", () => api.createAnalysis({}), { "X-OpenRouter-Key": "sk-or-v1-test" }],
  ["estimates send no key", () => api.estimateRun({}), {}],
  ["reads send no key", () => api.catalog(), {}],
];

for (const [name, call, expected] of CASES) {
  test(name, async () => {
    sent.length = 0;
    await call();
    const keys = Object.fromEntries(Object.entries(sent[0].headers).filter(([header]) => header.startsWith("X-")));
    assert.deepEqual(keys, expected);
  });
}
