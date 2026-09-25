// Run with `node --test tests/js/`. Exercises src/jev_bench/web/static/js/selection.js (pure, no DOM).
import assert from "node:assert/strict";
import test from "node:test";

const { sameSet, latestCompletedPerColumn } = await import("../../src/jev_bench/web/static/js/selection.js");

const run = (id, column, status = "completed", generation_ids = ["g1"]) => ({ id, column, status, generation_ids });

test("sameSet compares membership and size", () => {
  assert.equal(sameSet(["a", "b"], new Set(["b", "a"])), true);
  assert.equal(sameSet(["a"], new Set(["a", "b"])), false);
});

test("latestCompletedPerColumn picks the newest completed run of each column on exactly those generations", () => {
  const runs = [
    run("20260925-080000-jev-a", "jev"),
    run("20260925-090000-jev-b", "jev"),
    run("20260925-100000-jev-c", "jev", "running"),
    run("20260925-090000-openai-a", "openai", "completed", ["g1", "g2"]),
    run("20260925-080000-openai-b", "openai"),
    run("20260925-110000-anthropic-a", "anthropic", "failed"),
  ];
  assert.deepEqual(latestCompletedPerColumn(runs, ["g1"]), ["20260925-090000-jev-b", "20260925-080000-openai-b"]);
  assert.deepEqual(latestCompletedPerColumn(runs, ["g1", "g2"]), ["20260925-090000-openai-a"]);
  assert.deepEqual(latestCompletedPerColumn([], ["g1"]), []);
});
