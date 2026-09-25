/**
 * Display formatting helpers; every function returns "—" for null/undefined and for non-finite
 * numbers (NaN, ±Infinity); shortModel additionally returns "—" for any non-string input.
 * times(value, base) renders a ratio against a base as "×4", "×9.8" or "×0.03" ("—" without a positive base).
 * Exports: money, duration, pct, fixed, num, when, perMillion, shortModel, times.
 */

const missing = (value) =>
  value === null || value === undefined || (typeof value === "number" && !Number.isFinite(value));

export function money(usd) {
  if (missing(usd)) return "—";
  return `$${usd < 0.01 ? usd.toFixed(5) : usd.toFixed(4)}`;
}

export function duration(seconds) {
  if (missing(seconds)) return "—";
  if (seconds < 60) return `${seconds.toFixed(1)} s`;
  return `${Math.floor(seconds / 60)} min ${Math.round(seconds % 60)} s`;
}

export const pct = (value) => (missing(value) ? "—" : `${(value * 100).toFixed(1)}%`);
export const fixed = (value, digits = 3) => (missing(value) ? "—" : Number(value).toFixed(digits));
export const num = (value) => (missing(value) ? "—" : Number(value).toLocaleString("en-US"));
export const when = (iso) => (missing(iso) ? "—" : new Date(iso).toLocaleString());
export const perMillion = (usd) => (missing(usd) ? "—" : `$${Number(usd).toFixed(2)}`);
export const shortModel = (id) =>
  missing(id) || typeof id !== "string" ? "—" : id.split("/").pop();

export function times(value, base) {
  if (missing(value) || missing(base) || base <= 0) return "—";
  const ratio = value / base;
  const shown = ratio < 0.1 ? Number(ratio.toPrecision(1)) : ratio < 10 ? Number(ratio.toFixed(1)) : Math.round(ratio);
  return `×${shown}`;
}
