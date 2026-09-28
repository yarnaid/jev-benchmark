/**
 * Validated categorical colors for raters: one fixed hue per benchmark column (Jev, Anthropic, OpenAI,
 * Embeddings, Kev) plus the generator reference and human labels, stepped separately for the light and the
 * dark surface (checked with the dataviz palette validator: lightness, chroma, CVD and normal-vision
 * separation of adjacent slots). Kev takes the former green spare (validated in both modes, both with Kev
 * beside Embeddings and swapped in for it; its dark step sits at CVD ΔE 6.9 next to the reference, legal
 * because every chart has a legend and tooltips). A color follows the entity, never its position; a second run of the same
 * column and unknown columns take the spare slots, then a neutral gray.
 * Exports: raterColors.
 */

const SERIES = {
  jev: ["#1baf7a", "#199e70"],
  anthropic: ["#eb6834", "#d95926"],
  openai: ["#2a78d6", "#3987e5"],
  embeddings: ["#e87ba4", "#d55181"],
  kev: ["#008300", "#008300"],
  reference: ["#eda100", "#c98500"],
  human: ["#4a3aa7", "#9085e9"],
};
const SPARE = [["#e34948", "#e66767"]];
const GRAY = ["#8a8f98", "#8a8f98"];

const entityKey = (rater) => rater.run?.column ?? rater.kind;

export function raterColors(raters, theme) {
  const shade = theme === "dark" ? 1 : 0;
  const taken = new Set();
  let spare = 0;
  const colors = new Map();
  for (const rater of raters) {
    const key = entityKey(rater);
    const own = SERIES[key] && !taken.has(key) ? SERIES[key] : null;
    taken.add(key);
    colors.set(rater.id, (own ?? SPARE[spare++] ?? GRAY)[shade]);
  }
  return colors;
}
