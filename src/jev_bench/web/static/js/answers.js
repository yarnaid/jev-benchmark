/**
 * Pure helpers for comparing and displaying answers. A multi-label answer is an array of labels; any
 * other answer is one option id. A string compared with an array is treated as a one-label array, so
 * an old single-label reference still matches a multi-label answer. A multi-label question applies a
 * label when its probability is at least `appliedLimit` (threshold x the top probability), the same
 * rule as the server's metrics.multilabel.relative_labels.
 * Exports: asLabels, setMatch, answerText, withScore, appliedLimit.
 */

export const asLabels = (answer) => (Array.isArray(answer) ? answer : answer === undefined || answer === null ? [] : [answer]);

export function setMatch(reference, answer) {
  if (reference === undefined || reference === null || answer === undefined || answer === null) return "";
  const expected = new Set(asLabels(reference));
  const actual = asLabels(answer);
  const shared = actual.filter((label) => expected.has(label)).length;
  if (shared === expected.size && shared === actual.length) return "match";
  return shared > 0 ? "partial" : "mismatch";
}

export function answerText(answer) {
  if (answer === undefined || answer === null) return "—";
  if (Array.isArray(answer)) return answer.length ? answer.join(", ") : "none";
  return String(answer);
}

export function withScore(text, score) {
  return typeof score === "number" && Number.isFinite(score) ? `${text} · ${Math.round(score)}` : text;
}

export function appliedLimit(distribution, threshold) {
  const top = Math.max(0, ...Object.values(distribution ?? {}));
  return top > 0 ? threshold * top : Infinity;
}
