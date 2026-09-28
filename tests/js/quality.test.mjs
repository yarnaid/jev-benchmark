// Run with `node --test tests/js/`. Exercises the pure helpers of src/jev_bench/web/static/js/quality.js.
import assert from "node:assert/strict";
import test from "node:test";

const { cardQuality, kappaBand, qualityTargets, targetHeading } = await import("../../src/jev_bench/web/static/js/quality.js");
const { describe } = await import("../../src/jev_bench/web/static/js/glossary.js");

const BANDS = [
  ["negative", -0.1, { label: "poor", tone: "danger" }],
  ["zero", 0, { label: "slight", tone: "danger" }],
  ["0.20 is still slight", 0.2, { label: "slight", tone: "danger" }],
  ["just above 0.20", 0.21, { label: "fair", tone: "warning" }],
  ["0.40 is still fair", 0.4, { label: "fair", tone: "warning" }],
  ["just above 0.40", 0.41, { label: "moderate", tone: "warning" }],
  ["0.60 is still moderate", 0.6, { label: "moderate", tone: "warning" }],
  ["just above 0.60", 0.61, { label: "substantial", tone: "success" }],
  ["0.80 is still substantial", 0.8, { label: "substantial", tone: "success" }],
  ["just above 0.80", 0.81, { label: "almost perfect", tone: "success" }],
  ["perfect", 1, { label: "almost perfect", tone: "success" }],
  ["null", null, null],
  ["undefined", undefined, null],
  ["NaN", NaN, null],
  ["a string", "0.9", null],
];

for (const [name, kappa, expected] of BANDS) {
  test(`kappaBand: ${name}`, () => {
    assert.deepEqual(kappaBand(kappa), expected);
  });
}

const score = (rater, target = "reference") => ({ rater, target, kappa: 0.7, agreement: 0.8, n_questions: 3, n_emails: 4 });
const REPORT = { raters: [{ id: "r1" }, { id: "r2" }, { id: "reference" }], quality: [score("r1"), score("r1", "human"), score("other")] };

const CARDS = [
  ["no run", undefined, REPORT, { state: "none" }],
  ["running without a report", { id: "r1", status: "running" }, null, { state: "running" }],
  ["running run in the report", { id: "r1", status: "running" }, REPORT, { state: "running" }],
  ["no report yet", { id: "r1", status: "completed" }, null, { state: "none" }],
  ["run not compared", { id: "r9", status: "completed" }, REPORT, { state: "absent" }],
  ["compared without scores", { id: "r2", status: "completed" }, REPORT, { state: "unscored" }],
  ["only the run's own scores", { id: "r1", status: "failed" }, REPORT, { state: "scored", scores: [score("r1"), score("r1", "human")] }],
];

for (const [name, run, report, expected] of CARDS) {
  test(`cardQuality: ${name}`, () => {
    assert.deepEqual(cardQuality(run, report), expected);
  });
}

test("qualityTargets lists the scored targets in a fixed order", () => {
  assert.deepEqual(qualityTargets([]), []);
  assert.deepEqual(qualityTargets([score("r1", "human"), score("r2")]), ["reference", "human"]);
});

test("every target heading has a glossary entry", () => {
  for (const target of ["reference", "human"]) {
    const [, key] = targetHeading(target);
    assert.ok(describe(key).text.length > 0, target);
  }
});
