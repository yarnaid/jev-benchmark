// Run with `node --test tests/js/`. Exercises src/jev_bench/web/static/js/palette.js (pure, no DOM).
import assert from "node:assert/strict";
import test from "node:test";

const { raterColors } = await import("../../src/jev_bench/web/static/js/palette.js");

const run = (id, column) => ({ id, run: { column } });
const REFERENCE = { id: "reference", kind: "reference" };
const HUMAN = { id: "human", kind: "human" };

test("each column keeps its own validated color in light and dark mode", () => {
  const raters = [run("a", "jev"), run("b", "anthropic"), run("c", "openai"), run("d", "embeddings"), run("e", "kev"), REFERENCE, HUMAN];
  assert.deepEqual([...raterColors(raters, "light").values()], ["#1baf7a", "#eb6834", "#2a78d6", "#e87ba4", "#008300", "#eda100", "#4a3aa7"]);
  assert.deepEqual([...raterColors(raters, "dark").values()], ["#199e70", "#d95926", "#3987e5", "#d55181", "#008300", "#c98500", "#9085e9"]);
});

test("a color follows the entity, not its position", () => {
  const colors = raterColors([run("b", "anthropic"), REFERENCE], "light");
  assert.equal(colors.get("b"), "#eb6834");
  assert.equal(colors.get("reference"), "#eda100");
});

test("a second run of the same column and unknown columns take the spare slots, then gray", () => {
  const colors = raterColors([run("a", "jev"), run("a2", "jev"), run("x", "custom"), run("y", "other")], "light");
  assert.deepEqual([...colors.values()], ["#1baf7a", "#e34948", "#8a8f98", "#8a8f98"]);
});
