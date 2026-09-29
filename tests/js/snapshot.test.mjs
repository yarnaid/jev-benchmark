// Run with `node --test tests/js/`. Exercises src/jev_bench/web/static/js/snapshot.js (pure).
import assert from "node:assert/strict";
import test from "node:test";

const { snapshotInfo } = await import("../../src/jev_bench/web/static/js/snapshot.js");
const REPO = "https://github.com/yarnaid/jev-benchmark";

const CASES = [
  ["with a commit", { built_at: "2026-09-29T12:00:00Z", commit: "9c3c2956489458b3" }, { text: "Snapshot · 9c3c295 · 2026-09-29", href: `${REPO}/commit/9c3c2956489458b3` }],
  ["without a commit", { built_at: "2026-09-29T12:00:00Z", commit: null }, { text: "Snapshot · 2026-09-29", href: null }],
];

for (const [name, manifest, expected] of CASES) {
  test(`snapshotInfo ${name}`, () => {
    const { text, href } = snapshotInfo(manifest, REPO);
    assert.deepEqual({ text, href }, expected);
  });
}
