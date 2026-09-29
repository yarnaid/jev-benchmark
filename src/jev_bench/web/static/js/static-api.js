/**
 * Pure mapping of a JSON API request to its file in the static snapshot: the JS half of the contract whose
 * Python half is jev_bench.site.paths (shared cases: tests/fixtures/site_paths.json). A view-dependent
 * request is matched to the manifest view with the same run set (and the same generation set, or the
 * email's generation for a detail), in any order. Anything else with a query, and any non-GET, throws.
 * Exports: NotPublished, staticFile.
 */
import { sameSet } from "./selection.js";

export class NotPublished extends Error {}

const ids = (params, name) => (params.get(name) ?? "").split(",").filter(Boolean);
const percentFile = (params) => (params.has("threshold") ? `t${Math.round(Number(params.get("threshold")) * 100)}` : "t-default");

function findView(manifest, runIds, matchesGenerations) {
  const view = manifest.views.find((item) => sameSet(runIds, new Set(item.run_ids)) && matchesGenerations(item.generation_ids));
  if (!view) throw new NotPublished("selection");
  return view.id;
}

function thresholdFile(endpoint, manifest, params, generationIds = null) {
  const matches = (candidate) => generationIds === null || sameSet(generationIds, new Set(candidate));
  return `api/${endpoint}/${findView(manifest, ids(params, "runs"), matches)}/${percentFile(params)}.json`;
}

function emailFile(emailId, manifest, params) {
  const generation = emailId.slice(0, emailId.lastIndexOf("."));
  return `api/email/${findView(manifest, ids(params, "runs"), (candidate) => candidate.includes(generation))}/${emailId}.json`;
}

export function staticFile(method, path, manifest) {
  if (method !== "GET") throw new NotPublished(`${method} ${path}`);
  const url = new URL(path, "http://snapshot.invalid");
  if (url.pathname === "/compare") return thresholdFile("compare", manifest, url.searchParams);
  if (url.pathname === "/emails") return thresholdFile("emails", manifest, url.searchParams, ids(url.searchParams, "generations"));
  if (url.pathname.startsWith("/emails/")) return emailFile(url.pathname.slice("/emails/".length), manifest, url.searchParams);
  if (url.search) throw new NotPublished(path);
  return `api${url.pathname}.json`;
}
