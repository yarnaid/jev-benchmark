// Run with `node --test tests/js/`. Exercises src/jev_bench/web/static/js/slots.js (pure, no DOM).
import assert from "node:assert/strict";
import test from "node:test";

const { hiddenColumnIds, slotGroups, slotPicks, slotPrefKey, visibleColumns } = await import("../../src/jev_bench/web/static/js/slots.js");

const CATALOG = [
  { id: "jev", title: "Jev", slot: "jev" },
  { id: "embeddings", title: "Embeddings", slot: "embeddings" },
  { id: "anthropic", title: "Anthropic", slot: "anthropic" },
  { id: "kev", title: "Kev", slot: "embeddings" },
];

test("slotGroups groups by slot, positioned at the slot's first column", () => {
  assert.deepEqual(slotGroups(CATALOG).map(({ slot, columns }) => [slot, columns.map((column) => column.id)]), [["jev", ["jev"]], ["embeddings", ["embeddings", "kev"]], ["anthropic", ["anthropic"]]]);
});

test("a column without a slot is its own slot", () => {
  assert.deepEqual(slotGroups([{ id: "x" }]).map((group) => group.slot), ["x"]);
});

const PICK_CASES = [
  ["no stored pick shows the slot's first column", {}, ["jev", "embeddings", "anthropic"], ["kev"]],
  ["a stored pick swaps the column in", { embeddings: "kev" }, ["jev", "kev", "anthropic"], ["embeddings"]],
  ["a stale pick (column removed from config) falls back", { embeddings: "gone" }, ["jev", "embeddings", "anthropic"], ["kev"]],
  ["a pick naming another slot's column is ignored", { embeddings: "jev" }, ["jev", "embeddings", "anthropic"], ["kev"]],
];

for (const [name, picks, visible, hidden] of PICK_CASES) {
  test(`visibleColumns and hiddenColumnIds: ${name}`, () => {
    assert.deepEqual(visibleColumns(CATALOG, picks).map((column) => column.id), visible);
    assert.deepEqual([...hiddenColumnIds(CATALOG, picks)], hidden);
  });
}

test("slotPicks reads one stored pick per slot", () => {
  const stored = { "benchmark.slot.embeddings": "kev" };
  assert.equal(slotPrefKey("embeddings"), "benchmark.slot.embeddings");
  assert.deepEqual(slotPicks(CATALOG, (key) => stored[key] ?? null), { jev: null, embeddings: "kev", anthropic: null });
});
