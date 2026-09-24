/**
 * Safe localStorage access: JSON values under the "jev-bench." prefix; private windows or blocked
 * storage fall back to defaults and never throw.
 * Exports: readPref, writePref, removePref.
 */

const PREFIX = "jev-bench.";

export function readPref(name, fallback = null) {
  try {
    const raw = localStorage.getItem(PREFIX + name);
    return raw === null ? fallback : JSON.parse(raw);
  } catch {
    return fallback;
  }
}

export function writePref(name, value) {
  try {
    localStorage.setItem(PREFIX + name, JSON.stringify(value));
  } catch (error) {
    console.warn("localStorage unavailable", error);
  }
}

export function removePref(name) {
  try {
    localStorage.removeItem(PREFIX + name);
  } catch (error) {
    console.warn("localStorage unavailable", error);
  }
}
