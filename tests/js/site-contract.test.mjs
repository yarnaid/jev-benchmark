// Run with `node --test tests/js/`. Checks the UI rules that jev_bench.site mirrors in Python against the
// shared cases in tests/fixtures.
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

const { thresholdRange } = await import("../../src/jev_bench/web/static/js/widgets.js");
const fixture = (name) => JSON.parse(readFileSync(new URL(`../fixtures/${name}`, import.meta.url), "utf-8"));

for (const { id, default: value, min, step } of fixture("threshold_steps.json")) {
  test(`thresholdRange matches the shared case ${id}`, () => {
    const range = thresholdRange(value);
    assert.deepEqual({ min: range.min, max: range.max, step: range.step }, { min, max: 100, step });
  });
}
