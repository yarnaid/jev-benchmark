// Run with `node --test tests/js/`. Exercises src/jev_bench/web/static/js/answers.js (pure, no DOM).
import assert from "node:assert/strict";
import test from "node:test";

const { asLabels, setMatch, answerText, withScore, appliedLimit } = await import("../../src/jev_bench/web/static/js/answers.js");

const MATCHES = [
  ["same option", "spam", "spam", "match"],
  ["different option", "spam", "work", "mismatch"],
  ["same label set in any order", ["a", "b"], ["b", "a"], "match"],
  ["overlapping label sets", ["a", "b"], ["b", "c"], "partial"],
  ["answer is a subset", ["a", "b"], ["a"], "partial"],
  ["disjoint label sets", ["a"], ["b"], "mismatch"],
  ["both empty", [], [], "match"],
  ["old string reference vs label list", "spam", ["spam"], "match"],
  ["old string reference vs wider list", "spam", ["spam", "scam"], "partial"],
  ["no answer", "spam", undefined, ""],
  ["no reference", undefined, ["spam"], ""],
  ["null reference", null, "spam", ""],
];

for (const [name, reference, answer, expected] of MATCHES) {
  test(`setMatch: ${name}`, () => assert.equal(setMatch(reference, answer), expected));
}

const TEXTS = [
  ["missing", undefined, "—"],
  ["null", null, "—"],
  ["empty label list", [], "none"],
  ["label list", ["billing", "meeting"], "billing, meeting"],
  ["option id", "today", "today"],
];

for (const [name, answer, expected] of TEXTS) {
  test(`answerText: ${name}`, () => assert.equal(answerText(answer), expected));
}

test("asLabels wraps a single answer and passes arrays through", () => {
  assert.deepEqual(asLabels("a"), ["a"]);
  assert.deepEqual(asLabels(["a", "b"]), ["a", "b"]);
  assert.deepEqual(asLabels(undefined), []);
});

const SCORES = [
  ["rounded score", 72.4, "today · 72"],
  ["zero", 0, "today · 0"],
  ["missing", undefined, "today"],
  ["not finite", NaN, "today"],
];

for (const [name, score, expected] of SCORES) {
  test(`withScore: ${name}`, () => assert.equal(withScore("today", score), expected));
}

const LIMITS = [
  ["three quarters of the top", { billing: 0.5, meeting: 0.375, travel: 0.125 }, 0.75, 0.375],
  ["threshold one keeps only the top", { a: 0.6, b: 0.4 }, 1, 0.6],
  ["empty distribution applies nothing", {}, 0.8, Infinity],
  ["all-zero distribution applies nothing", { a: 0, b: 0 }, 0.8, Infinity],
];

for (const [name, distribution, threshold, expected] of LIMITS) {
  test(`appliedLimit: ${name}`, () => assert.equal(appliedLimit(distribution, threshold), expected));
}
