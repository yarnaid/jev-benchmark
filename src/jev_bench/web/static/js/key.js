/**
 * Browser-side OpenRouter key and optional Hugging Face token (for Kev) kept in localStorage, with a
 * change event for the navbar badge and the Kev card, and the steps to get each (shown in the key dialog
 * and on the Help page).
 * isHfToken guards against pasting another secret (the OpenRouter key) into the Hugging Face field.
 * Exports: KEY_EVENT, KEY_STEPS, HF_STEPS, getKey, setKey, forgetKey, getHfToken, setHfToken, forgetHfToken,
 * isHfToken.
 */
import { readPref, removePref, writePref } from "./storage.js";

const KEY_PREF = "openrouter-key";
const HF_PREF = "hf-token";
export const KEY_EVENT = "jev-bench:key-changed";
export const KEY_STEPS = [
  ["Create a key", "https://openrouter.ai/keys", "Sign in to OpenRouter, open Keys and create a key (it starts with sk-or-v1-)."],
  ["Add credits", "https://openrouter.ai/settings/credits", "Runs, generations and analyses are paid per use from your OpenRouter credits."],
  ["Optional: your provider keys", "https://openrouter.ai/workspaces/default/byok", "Bring your own Anthropic or OpenAI key (BYOK) to be billed by the provider directly."],
];

export const HF_STEPS = [
  ["Optional: a Hugging Face token", "https://huggingface.co/settings/tokens", "For the Kev column only: sign in to Hugging Face, open Access Tokens and create a Read token (it starts with hf_). Without one Kev gets 2 minutes of free GPU a day."],
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

export const isHfToken = (value) => /^hf_[A-Za-z0-9]+$/.test(value);

export const getHfToken = () => readPref(HF_PREF);

export function setHfToken(value) {
  writePref(HF_PREF, value);
  window.dispatchEvent(new Event(KEY_EVENT));
}

export function forgetHfToken() {
  removePref(HF_PREF);
  window.dispatchEvent(new Event(KEY_EVENT));
}
