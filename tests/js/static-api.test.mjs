// Run with `node --test tests/js/`. Exercises src/jev_bench/web/static/js/static-api.js (pure) against the
// shared cases in tests/fixtures/site_paths.json.
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

const { NotPublished, staticFile } = await import("../../src/jev_bench/web/static/js/static-api.js");
const paths = JSON.parse(readFileSync(new URL("../fixtures/site_paths.json", import.meta.url), "utf-8"));

for (const { id, request, file } of paths.published) {
  test(`staticFile maps ${id}`, () => assert.equal(staticFile("GET", request, paths.manifest), file));
}

for (const { id, method, request } of paths.unpublished) {
  test(`staticFile refuses ${id}`, () => assert.throws(() => staticFile(method, request, paths.manifest), NotPublished));
}
