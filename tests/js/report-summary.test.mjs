// Run with `node --test tests/js/`. Exercises the "×Jev" baseline of src/jev_bench/web/static/js/report-summary.js.
import assert from "node:assert/strict";
import test from "node:test";

const { jevBaseline } = await import("../../src/jev_bench/web/static/js/report-summary.js");

const rater = (column, kind = "decisions") => ({ id: column, run: { column, kind } });
const REFERENCE = { id: "reference", run: null };

const CASES = [
  ["the jev column's run", [rater("anthropic", "chat"), rater("jev")], "jev"],
  ["jev even when Kev (another decisions column) comes first", [rater("kev"), rater("jev")], "jev"],
  ["no baseline without a jev run, even with Kev", [rater("kev"), REFERENCE], null],
  ["no baseline without runs", [REFERENCE], null],
];

for (const [name, raters, expected] of CASES) {
  test(`jevBaseline: ${name}`, () => assert.equal(jevBaseline(raters)?.column ?? null, expected));
}
