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

const { latestCompletedPerColumn } = await import("../../src/jev_bench/web/static/js/selection.js");
const { hiddenColumnIds, slotGroups } = await import("../../src/jev_bench/web/static/js/slots.js");
const views = fixture("default_views.json");

for (const { id, generation_ids, hidden, expected } of views.latest) {
  test(`latestCompletedPerColumn matches the shared case ${id}`, () => {
    const shown = views.runs.filter((run) => !hidden.includes(run.column));
    assert.deepEqual(latestCompletedPerColumn(shown, generation_ids).sort(), expected);
  });
}

test("every combination of slot picks hides one of the shared hidden sets", () => {
  const combos = slotGroups(views.catalog).reduce((all, { slot, columns }) => all.flatMap((picks) => columns.map((column) => ({ ...picks, [slot]: column.id }))), [{}]);
  assert.deepEqual(combos.map((picks) => [...hiddenColumnIds(views.catalog, picks)].sort()), views.hidden_sets);
});
