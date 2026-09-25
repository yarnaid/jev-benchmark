// Run with `node --test tests/js/`. Exercises src/jev_bench/web/static/js/analysis-links.js (pure).
import assert from "node:assert/strict";
import test from "node:test";

const { explorerHref } = await import("../../src/jev_bench/web/static/js/analysis-links.js");

const META = { generation_ids: ["g1", "g2"], run_ids: ["r1", "r2"], email_refs: { e001: "g1.0001", e017: "g2.0007" } };

test("explorerHref opens the email with the analysed generations and runs", () => {
  const url = new URL(explorerHref(META, "e017"), "http://host.test");
  assert.equal(url.pathname, "/explorer.html");
  assert.deepEqual(Object.fromEntries(url.searchParams), { generations: "g1,g2", runs: "r1,r2", email: "g2.0007" });
});

test("explorerHref is null for an unknown ref or a record without refs", () => {
  assert.equal(explorerHref(META, "e999"), null);
  assert.equal(explorerHref({ ...META, email_refs: undefined }, "e001"), null);
  assert.equal(explorerHref(META, "toString"), null);
});
