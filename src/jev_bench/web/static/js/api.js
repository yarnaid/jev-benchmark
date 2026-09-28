/**
 * JSON client for the /api endpoints; the browser-stored OpenRouter key is attached only to job-starting
 * calls, and the Hugging Face token only to run creation.
 * Query parameters that are undefined, null or "" are omitted (an unset label threshold is never sent).
 * Exports: api, ApiError, needsKey.
 */
import { getHfToken, getKey } from "./key.js";

export class ApiError extends Error {
  constructor(status, detail) {
    super(typeof detail === "string" ? detail : JSON.stringify(detail));
    this.status = status;
  }
}

async function request(method, path, { body, withKey = false, withHfToken = false } = {}) {
  const headers = { Accept: "application/json" };
  if (body !== undefined) headers["Content-Type"] = "application/json";
  const key = withKey ? getKey() : null;
  if (key) headers["X-OpenRouter-Key"] = key;
  const hfToken = withHfToken ? getHfToken() : null;
  if (hfToken) headers["X-HF-Token"] = hfToken;
  const response = await fetch(`/api${path}`, { method, headers, body: body === undefined ? undefined : JSON.stringify(body) });
  const isJson = (response.headers.get("content-type") ?? "").includes("json");
  const payload = isJson ? await response.json() : await response.text();
  if (!response.ok) throw new ApiError(response.status, payload?.detail ?? payload);
  return payload;
}

function query(params) {
  const entries = Object.entries(params).filter(([, value]) => value !== undefined && value !== null && value !== "");
  const text = new URLSearchParams(entries).toString();
  return text ? `?${text}` : "";
}

const segment = (value) => encodeURIComponent(value);

export const api = {
  status: () => request("GET", "/status"),
  questions: () => request("GET", "/questions"),
  estimateRun: (body) => request("POST", "/runs/estimate", { body }),
  catalog: () => request("GET", "/catalog"),
  generations: () => request("GET", "/generations"),
  generation: (id) => request("GET", `/generations/${segment(id)}`),
  createGeneration: (body) => request("POST", "/generations", { body, withKey: true }),
  cancelGeneration: (id) => request("POST", `/generations/${segment(id)}/cancel`),
  runs: (generationIds = []) => request("GET", `/runs${query({ generations: generationIds.join(",") })}`),
  run: (id) => request("GET", `/runs/${segment(id)}`),
  createRun: (body) => request("POST", "/runs", { body, withKey: true, withHfToken: true }),
  cancelRun: (id) => request("POST", `/runs/${segment(id)}/cancel`),
  compare: (runIds, threshold) => request("GET", `/compare${query({ runs: runIds.join(","), threshold })}`),
  emails: (generationIds, runIds = [], threshold) =>
    request("GET", `/emails${query({ generations: generationIds.join(","), runs: runIds.join(","), threshold })}`),
  email: (id, runIds = []) => request("GET", `/emails/${segment(id)}${query({ runs: runIds.join(",") })}`),
  putLabel: (id, answers) => request("PUT", `/labels/${segment(id)}`, { body: { answers } }),
  analysisDefaults: () => request("GET", "/analysis/defaults"),
  estimateAnalysis: (body) => request("POST", "/analyses/estimate", { body }),
  createAnalysis: (body) => request("POST", "/analyses", { body, withKey: true }),
  analyses: () => request("GET", "/analyses"),
  analysis: (id) => request("GET", `/analyses/${segment(id)}`),
  analysisPrompts: (id) => request("GET", `/analyses/${segment(id)}/prompts`),
  cancelAnalysis: (id) => request("POST", `/analyses/${segment(id)}/cancel`),
};

export const needsKey = (error) => error instanceof ApiError && error.status === 400 && error.message.includes("API key");
