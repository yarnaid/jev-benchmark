// Run with `node --test tests/js/`. Exercises src/jev_bench/web/static/js/option-order.js (pure, no DOM).
import assert from "node:assert/strict";
import test from "node:test";

const { orderOptions } = await import("../../src/jev_bench/web/static/js/option-order.js");

const OPTIONS = ["personal", "work", "news", "spam", "phishing", "scam"];

const CASES = [
  [
    "reference first, then by the highest probability of any run",
    { reference: ["spam"], runs: [{ spam: 0.6, phishing: 0.3, news: 0.1 }, { phishing: 0.5, scam: 0.2, spam: 0.3 }] },
    { shown: ["spam", "phishing", "scam", "news"], folded: ["personal", "work"] },
  ],
  [
    "several reference labels keep option order",
    { reference: ["phishing", "spam"], runs: [{ phishing: 0.9, spam: 0.1 }] },
    { shown: ["spam", "phishing"], folded: ["personal", "work", "news", "scam"] },
  ],
  [
    "a reference label below the cut is still shown",
    { reference: ["personal"], runs: [{ work: 0.99, personal: 0.01 }] },
    { shown: ["personal", "work"], folded: ["news", "spam", "phishing", "scam"] },
  ],
  [
    "ties keep option order",
    { reference: [], runs: [{ news: 0.4, work: 0.4, scam: 0.2 }] },
    { shown: ["work", "news", "scam"], folded: ["personal", "spam", "phishing"] },
  ],
  [
    "runs without an answer are ignored",
    { reference: ["work"], runs: [null, undefined, { work: 1 }] },
    { shown: ["work"], folded: ["personal", "news", "spam", "phishing", "scam"] },
  ],
  [
    "no runs and no reference folds everything",
    { reference: [], runs: [] },
    { shown: [], folded: OPTIONS },
  ],
];

for (const [name, { reference, runs }, expected] of CASES) {
  test(`orderOptions: ${name}`, () => assert.deepEqual(orderOptions(OPTIONS, reference, runs), expected));
}

test("orderOptions: the cut is configurable and inclusive", () => {
  const { shown } = orderOptions(["a", "b"], [], [{ a: 0.5, b: 0.2 }], 0.2);
  assert.deepEqual(shown, ["a", "b"]);
});
