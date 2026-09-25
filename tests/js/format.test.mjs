// Manual gate (controller ruling, not part of the pytest suite): run with `node --test tests/js/`.
// Exercises src/jev_bench/web/static/js/format.js.
import assert from "node:assert/strict";
import test from "node:test";

const { money, duration, pct, fixed, num, when, perMillion, shortModel } = await import(
  "../../src/jev_bench/web/static/js/format.js"
);

const NON_FINITE = [NaN, Infinity, -Infinity];
const NUMERIC_FORMATTERS = { money, duration, pct, fixed, num, perMillion };

for (const [name, formatter] of Object.entries(NUMERIC_FORMATTERS)) {
  for (const value of NON_FINITE) {
    test(`${name}(${String(value)}) is treated as missing`, () => {
      assert.equal(formatter(value), "—");
    });
  }
}

test("null and undefined are missing", () => {
  assert.equal(money(null), "—");
  assert.equal(money(undefined), "—");
  assert.equal(when(null), "—");
  assert.equal(shortModel(undefined), "—");
});

test("finite numbers still format normally", () => {
  assert.equal(money(1), "$1.0000");
  assert.equal(pct(0.5), "50.0%");
  assert.equal(num(1000), "1,000");
  assert.equal(fixed(1.23456), "1.235");
});

const NON_STRING_MODELS = [42, {}, [], true, NaN];

for (const value of NON_STRING_MODELS) {
  test(`shortModel(${String(value)}) of a non-string is missing`, () => {
    assert.equal(shortModel(value), "—");
  });
}

test("shortModel keeps the last path segment of a real model id", () => {
  assert.equal(shortModel("openai/gpt-5.6-terra"), "gpt-5.6-terra");
});

const { times } = await import("../../src/jev_bench/web/static/js/format.js");

const TIMES = [
  ["four times", 12, 3, "×4"],
  ["one decimal below ten", 57, 5.83, "×9.8"],
  ["whole number from ten", 0.635, 0.00808, "×79"],
  ["the base itself", 3, 3, "×1"],
  ["faster than the base", 1.5, 3, "×0.5"],
  ["much cheaper than the base", 0.1, 3, "×0.03"],
  ["zero base", 5, 0, "—"],
  ["missing value", null, 3, "—"],
  ["missing base", 3, undefined, "—"],
  ["non-finite value", Infinity, 3, "—"],
];

for (const [name, value, base, expected] of TIMES) {
  test(`times: ${name}`, () => assert.equal(times(value, base), expected));
}
