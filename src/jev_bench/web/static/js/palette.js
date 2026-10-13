/**
 * Validated categorical colors for raters: one fixed hue per benchmark column (Jev, Anthropic, OpenAI,
 * Embeddings, Kev, Microsoft) plus the generator reference and human labels, stepped separately for the light
 * and the dark surface (checked with the dataviz palette validator: lightness, chroma, CVD and normal-vision
 * separation of adjacent slots). Kev takes the former green spare (validated in both modes, both with Kev
 * beside Embeddings and swapped in for it; its dark step sits at CVD ΔE 6.9 next to the reference, legal
 * because every chart has a legend and tooltips). Microsoft takes the last hue, red, at deeper steps than the
 * documented ones: those sat at normal-vision ΔE 7.1 (both modes) from Anthropic's orange, and the dark one at
 * 13.0 from the reference's amber. The light step keeps the documented hue at OKLCH L 0.43 (ΔE >= 24.5 from
 * every other color, CVD >= 9.1); the dark step is a crimson at L 0.49 (ΔE 17.0 from Anthropic, CVD 7.6 next
 * to Kev, contrast 2.67:1 on the dark surface, relieved by the legends, tooltips and tables). Every default
 * view (Latest per slot pick, all columns) passes in both modes. A hand-picked dark comparison with Embeddings
 * and Microsoft but no Kev puts them side by side at ΔE 13.3, under the floor of 15; the legends name them.
 * All eight hues are taken, so a second run of the same column and unknown columns take a neutral gray. A color
 * follows the entity, never its position.
 * Exports: raterColors.
 */

const SERIES = {
  jev: ["#1baf7a", "#199e70"],
  anthropic: ["#eb6834", "#d95926"],
  openai: ["#2a78d6", "#3987e5"],
  embeddings: ["#e87ba4", "#d55181"],
  kev: ["#008300", "#008300"],
  microsoft: ["#94151d", "#ac1b53"],
  reference: ["#eda100", "#c98500"],
  human: ["#4a3aa7", "#9085e9"],
};
const GRAY = ["#8a8f98", "#8a8f98"];

const entityKey = (rater) => rater.run?.column ?? rater.kind;

export function raterColors(raters, theme) {
  const shade = theme === "dark" ? 1 : 0;
  const taken = new Set();
  const colors = new Map();
  for (const rater of raters) {
    const key = entityKey(rater);
    const own = SERIES[key] && !taken.has(key) ? SERIES[key] : null;
    taken.add(key);
    colors.set(rater.id, (own ?? GRAY)[shade]);
  }
  return colors;
}
