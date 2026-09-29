// Run with `node --test tests/js/`. Exercises the static-snapshot path of src/jev_bench/web/static/js/api.js.
import assert from "node:assert/strict";
import test from "node:test";

globalThis.localStorage = { getItem: () => null, setItem: () => {}, removeItem: () => {} };
const MANIFEST = { built_at: "2026-09-29T12:00:00Z", commit: null, views: [{ id: "v1", generation_ids: ["g1"], run_ids: ["r1"] }] };
const FILES = new Map([["api/site.json", MANIFEST], ["api/compare/v1/t-default.json", { raters: [] }], ["api/compare/v1/t50.json", { raters: ["at 50"] }], ["api/status.json", { server_key: false }]]);
const fetched = [];
const reply = (status, body) => ({ ok: status === 200, status, headers: new Map([["content-type", status === 200 ? "application/json" : "text/html"]]), json: async () => body, text: async () => "<html>404</html>" });
globalThis.fetch = async (url, init = {}) => {
  fetched.push({ url, init });
  return FILES.has(url) ? reply(200, FILES.get(url)) : reply(404, null);
};

const { ApiError, NOT_PUBLISHED, notPublished, publishedThreshold, staticRequest } = await import("../../src/jev_bench/web/static/js/api.js");

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

test("every published file is revalidated, so a deploy is never read from stale lists", async () => {
  await staticRequest("GET", "/status");
  const status = fetched.filter(({ url }) => url === "api/status.json").at(-1);
  assert.equal(status.init.cache, "no-cache");
});

const compareAt = (threshold) => staticRequest("GET", threshold == null ? "/compare?runs=r1" : `/compare?runs=r1&threshold=${threshold}`);

const THRESHOLDS = [
  ["a published threshold is kept", 0.5, { threshold: 0.5, value: { raters: ["at 50"] } }],
  ["an unpublished stored threshold falls back to the default", 0.35, { threshold: null, value: { raters: [] } }],
  ["no stored threshold loads the default", null, { threshold: null, value: { raters: [] } }],
];

for (const [name, threshold, expected] of THRESHOLDS) {
  test(`publishedThreshold: ${name}`, async () => {
    assert.deepEqual(await publishedThreshold(compareAt, threshold), expected);
  });
}

const FAILURES = [
  ["an unpublished selection still fails without a threshold", (t) => staticRequest("GET", t == null ? "/compare?runs=r2" : `/compare?runs=r2&threshold=${t}`), 0.5, NOT_PUBLISHED],
  ["other errors are not swallowed", async () => { throw new ApiError(500, "boom"); }, 0.5, "boom"],
];

for (const [name, load, threshold, message] of FAILURES) {
  test(`publishedThreshold: ${name}`, async () => {
    await assert.rejects(publishedThreshold(load, threshold), (error) => error instanceof ApiError && error.message === message);
  });
}
