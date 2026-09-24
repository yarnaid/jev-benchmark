/**
 * JSON client for the /api endpoints; the browser-stored key is attached only to job-starting calls.
 * Exports: api, ApiError, needsKey.
 */
import { getKey } from "./key.js";

export class ApiError extends Error {
  constructor(status, detail) {
    super(typeof detail === "string" ? detail : JSON.stringify(detail));
    this.status = status;
  }
}

async function request(method, path, { body, withKey = false } = {}) {
  const headers = { Accept: "application/json" };
  if (body !== undefined) headers["Content-Type"] = "application/json";
  const key = withKey ? getKey() : null;
  if (key) headers["X-OpenRouter-Key"] = key;
  const response = await fetch(`/api${path}`, { method, headers, body: body === undefined ? undefined : JSON.stringify(body) });
  const isJson = (response.headers.get("content-type") ?? "").includes("json");
  const payload = isJson ? await response.json() : await response.text();
  if (!response.ok) throw new ApiError(response.status, payload?.detail ?? payload);
  return payload;
}

function query(params) {
  const entries = Object.entries(params).filter(([, value]) => value !== undefined && value !== "");
  const text = new URLSearchParams(entries).toString();
  return text ? `?${text}` : "";
}

const segment = (value) => encodeURIComponent(value);

export const api = {
  status: () => request("GET", "/status"),
  catalog: () => request("GET", "/catalog"),
  generations: () => request("GET", "/generations"),
  generation: (id) => request("GET", `/generations/${segment(id)}`),
  createGeneration: (body) => request("POST", "/generations", { body, withKey: true }),
  cancelGeneration: (id) => request("POST", `/generations/${segment(id)}/cancel`),
  runs: (generationIds = []) => request("GET", `/runs${query({ generations: generationIds.join(",") })}`),
  run: (id) => request("GET", `/runs/${segment(id)}`),
  createRun: (body) => request("POST", "/runs", { body, withKey: true }),
  cancelRun: (id) => request("POST", `/runs/${segment(id)}/cancel`),
  compare: (runIds) => request("GET", `/compare${query({ runs: runIds.join(",") })}`),
  emails: (generationIds, runIds = []) =>
    request("GET", `/emails${query({ generations: generationIds.join(","), runs: runIds.join(",") })}`),
  email: (id, runIds = []) => request("GET", `/emails/${segment(id)}${query({ runs: runIds.join(",") })}`),
  putLabel: (id, answers) => request("PUT", `/labels/${segment(id)}`, { body: { answers } }),
};

export const needsKey = (error) => error instanceof ApiError && error.status === 400 && error.message.includes("API key");
