/**
 * Browser-side OpenRouter key kept in localStorage, with a change event for the navbar badge.
 * Exports: KEY_EVENT, getKey, setKey, forgetKey.
 */
import { readPref, removePref, writePref } from "./storage.js";

const KEY_PREF = "openrouter-key";
export const KEY_EVENT = "jev-bench:key-changed";

export const getKey = () => readPref(KEY_PREF);

export function setKey(value) {
  writePref(KEY_PREF, value);
  window.dispatchEvent(new Event(KEY_EVENT));
}

export function forgetKey() {
  removePref(KEY_PREF);
  window.dispatchEvent(new Event(KEY_EVENT));
}
