// Run with `node --test tests/js/`. Exercises src/jev_bench/web/static/js/selection.js (pure, no DOM).
import assert from "node:assert/strict";
import test from "node:test";

const { defaultRunIds, generationChoices, initialGenerations, latestCompletedPerColumn, orderByColumn, runChoices, sameSet } = await import(
  "../../src/jev_bench/web/static/js/selection.js"
);

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

const CATALOG = [
  { id: "jev", title: "Jev" },
  { id: "anthropic", title: "Anthropic" },
  { id: "openai", title: "OpenAI" },
];

test("orderByColumn sorts run ids by catalog column, unknown columns then unknown runs last", () => {
  const runs = [run("a", "openai"), run("b", "jev"), run("c", "anthropic"), run("d", "retired")];
  assert.deepEqual(orderByColumn(["a", "d", "x", "c", "b"], runs, CATALOG), ["b", "c", "a", "d", "x"]);
  assert.deepEqual(orderByColumn([], runs, CATALOG), []);
});

test("defaultRunIds is the latest completed run per column, in column order", () => {
  const runs = [run("20260925-090000-openai-a", "openai"), run("20260925-080000-jev-a", "jev"), run("20260925-100000-jev-b", "jev", "failed")];
  assert.deepEqual(defaultRunIds(runs, ["g1"], CATALOG), ["20260925-080000-jev-a", "20260925-090000-openai-a"]);
});

const GENERATIONS = [{ id: "g2", name: "new", done: 5, status: "completed" }, { id: "g1", name: "old", done: 3, status: "interrupted" }];

const INITIAL_CASES = [
  ["a requested run pins its generations", { requested: ["r"], stored: ["g1"] }, ["g1", "g2"]],
  ["stored generations that still exist", { requested: [], stored: ["gone", "g1"] }, ["g1"]],
  ["otherwise the newest generation", { requested: [], stored: null }, ["g2"]],
  ["unknown requested runs fall back", { requested: ["nope"], stored: [] }, ["g2"]],
];

for (const [name, { requested, stored }, expected] of INITIAL_CASES) {
  test(`initialGenerations: ${name}`, () => {
    const runs = [run("r", "jev", "completed", ["g1", "g2"])];
    assert.deepEqual(initialGenerations(GENERATIONS, runs, requested, stored), expected);
  });
}

test("initialGenerations of no generations is empty", () => assert.deepEqual(initialGenerations([], [], [], null), []));

test("generationChoices and runChoices describe the checklist items", () => {
  assert.deepEqual(generationChoices(GENERATIONS)[1], { value: "g1", text: "old · 3 emails", hint: "g1 · interrupted" });
  const runs = [
    { ...run("r1", "anthropic"), model: "anthropic/claude-sonnet-5", mode: "all_in_one", created_at: "2026-09-25T10:00:00Z", n_done: 4 },
    { ...run("r2", "retired"), model: "x/y", mode: "per_email", created_at: "2026-09-25T11:00:00Z", n_done: 2 },
    { ...run("r3", "jev", "failed"), model: "typesafe/jev-1.13", mode: "per_email", created_at: "2026-09-25T11:00:00Z", n_done: 0 },
  ];
  const choices = runChoices(runs, ["g1"], CATALOG);
  assert.deepEqual(choices.map((choice) => [choice.value, choice.text]), [["r1", "Anthropic · claude-sonnet-5 · all in one"], ["r2", "retired · y"]]);
  assert.match(choices[0].hint, / · 4 emails$/);
});
