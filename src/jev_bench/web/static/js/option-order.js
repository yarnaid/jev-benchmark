/**
 * Orders a question's options for the Explorer detail panel: the reference labels first (in option
 * order), then the other options by the highest probability any run gave them (ties keep option
 * order). Options that are not in the reference and stay below `minShown` in every run are returned
 * separately, so the panel can fold them away. Runs without an answer are ignored.
 * Exports: orderOptions.
 */

export function orderOptions(optionIds, reference, distributions, minShown = 0.05) {
  const expected = new Set(reference);
  const answered = distributions.filter(Boolean);
  const top = (option) => Math.max(0, ...answered.map((distribution) => distribution[option] ?? 0));
  const others = optionIds.filter((option) => !expected.has(option)).sort((a, b) => top(b) - top(a));
  return {
    shown: [...optionIds.filter((option) => expected.has(option)), ...others.filter((option) => top(option) >= minShown)],
    folded: others.filter((option) => top(option) < minShown).sort((a, b) => optionIds.indexOf(a) - optionIds.indexOf(b)),
  };
}
