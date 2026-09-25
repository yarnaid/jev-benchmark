/**
 * Browser-side OpenRouter key kept in localStorage, with a change event for the navbar badge, and
 * the steps to get a key (shown in the key dialog and on the Help page).
 * Exports: KEY_EVENT, KEY_STEPS, getKey, setKey, forgetKey.
 */
import { readPref, removePref, writePref } from "./storage.js";

const KEY_PREF = "openrouter-key";
export const KEY_EVENT = "jev-bench:key-changed";
export const KEY_STEPS = [
  ["Create a key", "https://openrouter.ai/keys", "Sign in to OpenRouter, open Keys and create a key (it starts with sk-or-v1-)."],
  ["Add credits", "https://openrouter.ai/settings/credits", "Runs, generations and analyses are paid per use from your OpenRouter credits."],
  ["Optional: your provider keys", "https://openrouter.ai/workspaces/default/byok", "Bring your own Anthropic or OpenAI key (BYOK) to be billed by the provider directly."],
];

export const getKey = () => readPref(KEY_PREF);

export function setKey(value) {
  writePref(KEY_PREF, value);
  window.dispatchEvent(new Event(KEY_EVENT));
}

export function forgetKey() {
  removePref(KEY_PREF);
  window.dispatchEvent(new Event(KEY_EVENT));
}
