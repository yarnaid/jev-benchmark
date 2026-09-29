// Run with `node --test tests/js/`. Exercises the static-snapshot path of src/jev_bench/web/static/js/api.js.
import assert from "node:assert/strict";
import test from "node:test";

globalThis.localStorage = { getItem: () => null, setItem: () => {}, removeItem: () => {} };
const MANIFEST = { built_at: "2026-09-29T12:00:00Z", commit: null, views: [{ id: "v1", generation_ids: ["g1"], run_ids: ["r1"] }] };
const FILES = new Map([["api/site.json", MANIFEST], ["api/compare/v1/t-default.json", { raters: [] }], ["api/status.json", { server_key: false }]]);
const fetched = [];
const reply = (status, body) => ({ ok: status === 200, status, headers: new Map([["content-type", status === 200 ? "application/json" : "text/html"]]), json: async () => body, text: async () => "<html>404</html>" });
globalThis.fetch = async (url, init = {}) => {
  fetched.push({ url, init });
  return FILES.has(url) ? reply(200, FILES.get(url)) : reply(404, null);
};

const { ApiError, NOT_PUBLISHED, notPublished, staticRequest } = await import("../../src/jev_bench/web/static/js/api.js");

test("a published GET is answered by its file", async () => {
  assert.deepEqual(await staticRequest("GET", "/compare?runs=r1"), { raters: [] });
});

test("the manifest is fetched once and revalidated", async () => {
  await staticRequest("GET", "/status");
  const manifests = fetched.filter(({ url }) => url === "api/site.json");
  assert.equal(manifests.length, 1);
  assert.equal(manifests[0].init.cache, "no-cache");
});

const REFUSED = [
  ["an unpublished selection", "GET", "/compare?runs=r2"],
  ["a write", "POST", "/runs"],
  ["a mapped file missing from the snapshot", "GET", "/runs/r9"],
];

for (const [name, method, path] of REFUSED) {
  test(`${name} is reported as not published`, async () => {
    await assert.rejects(staticRequest(method, path), (error) => error instanceof ApiError && error.status === 404 && error.message === NOT_PUBLISHED && notPublished(error));
  });
}
