// Run with `node --test tests/js/`. Exercises the pure thresholdRange of src/jev_bench/web/static/js/widgets.js.
import assert from "node:assert/strict";
import test from "node:test";

const { thresholdRange } = await import("../../src/jev_bench/web/static/js/widgets.js");

const RANGES = [
  ["default 80 %", 0.8, { min: 50, max: 100, step: 5, value: 80 }],
  ["one hundred", 1, { min: 50, max: 100, step: 5, value: 100 }],
  ["configured below 50 %", 0.4, { min: 40, max: 100, step: 5, value: 40 }],
  ["not a multiple of 5 below 50 %", 0.37, { min: 35, max: 100, step: 1, value: 37 }],
  ["not a multiple of 5 above 50 %", 0.73, { min: 50, max: 100, step: 1, value: 73 }],
  ["tiny threshold never allows 0", 0.01, { min: 1, max: 100, step: 1, value: 1 }],
];

for (const [name, threshold, expected] of RANGES) {
  test(`thresholdRange: ${name}`, () => {
    const range = thresholdRange(threshold);
    assert.deepEqual(range, expected);
    assert.equal((range.value - range.min) % range.step, 0, "the configured value is a valid slider position");
  });
}
