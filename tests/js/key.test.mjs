// Run with `node --test tests/js/`. Exercises the Hugging Face token check of src/jev_bench/web/static/js/key.js.
import assert from "node:assert/strict";
import test from "node:test";

const { isHfToken } = await import("../../src/jev_bench/web/static/js/key.js");

const CASES = [
  ["a read token", "hf_AbCdEf0123456789", true],
  ["an OpenRouter key pasted into the wrong field", "sk-or-v1-0123456789abcdef", false],
  ["an empty value", "", false],
  ["the bare prefix", "hf_", false],
  ["inner whitespace", "hf_abc def", false],
  ["non-ASCII characters", "hf_abcé", false],
];

for (const [name, value, expected] of CASES) {
  test(`isHfToken: ${name}`, () => assert.equal(isHfToken(value), expected));
}
