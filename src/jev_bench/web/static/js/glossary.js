/**
 * Plain-language descriptions of every metric and figure in the UI, written for a non-technical reader:
 * what it is and how to read it. The single source for the (?) tooltips and the Help page glossary.
 * hideTooltips(root) disposes the tooltips of (?) icons inside `root` before it is re-rendered, so a
 * tooltip that is open while its icon is removed never stays behind.
 * Exports: GLOSSARY (groups of [key, label, text]), describe, helpIcon, hideTooltips, withHelp.
 */
import { h, icon } from "./dom.js";

export const GLOSSARY = [
  {
    title: "Quality",
    entries: [
      ["quality_reference", "κ vs reference", "How closely the run matches the generator's intended answers: κ per question (1 = perfect, 0 = chance), averaged with every question counting equally. Above 0.8 reads as almost perfect, above 0.6 substantial, above 0.4 moderate, above 0.2 fair."],
      ["quality_human", "κ vs human", "The same score against your own labels from the Explorer, over the questions and emails you labeled. With only a few labels it is a rough guide."],
    ],
  },
  {
    title: "Speed and cost",
    entries: [
      ["status", "Status", "Where the run is: running, completed, cancelled, failed, or interrupted (the server stopped while it ran)."],
      ["elapsed", "Elapsed", "Time since the run started (or how long it took once finished)."],
      ["emails", "Emails", "How many emails this model returned a usable answer for."],
      ["errors", "Errors", "Emails whose answer could not be used, for example a network problem or a malformed reply. Lower is better."],
      ["duration", "Duration", "Wall-clock time for the whole run. Shorter is faster."],
      ["time_vs_jev", "Time vs Jev", "How many times longer the run took than the Jev run in this comparison. ×1 = as fast as Jev, ×4 = four times slower, ×0.5 = twice as fast."],
      ["cost", "Cost", "What the run cost on OpenRouter, in US dollars."],
      ["cost_vs_jev", "Cost vs Jev", "How many times more the run cost than the Jev run in this comparison. ×1 = Jev's cost, ×80 = eighty times more."],
      ["cost_per_email", "Cost per email", "The run's cost divided by the emails it answered: what classifying one email costs."],
      ["cold_cost", "Cold cost", "Embeddings only: what the run would cost if nothing were cached and every text had to be embedded again."],
      ["requests", "Requests", "How many API calls were made. \"+N split\" means a batch was cut into smaller calls to fit the model's limits."],
      ["tokens", "Tokens in / out", "Volume of text sent to and received from the model, in tokens (about ¾ of a word each). Tokens drive the cost."],
      ["latency", "Latency p50 / p95", "Time one API call took: p50 is the typical call, p95 the slow case (95% of calls were faster). Lower is faster."],
      ["cache_hits", "Cache hits", "Embeddings only: emails answered from the local cache instead of a new, paid API call."],
      ["estimate", "Estimated cost", "Expected cost before you start. \"Past runs\" scales what earlier runs of the same model really cost; \"tokens\" is an upper bound from the text size and list prices."],
    ],
  },
  {
    title: "Per-question statistics",
    entries: [
      ["n", "n", "How many emails this rater answered for this question."],
      ["entropy", "Entropy", "How spread out the model's probabilities are. 0 = completely sure of one answer; higher = more undecided between answers."],
      ["confidence", "Confidence", "The probability the model gave its top answer, averaged over emails. 100% = always fully sure; 50% = often torn between options."],
      ["mean_score", "Mean score (0–100)", "The average position on the question's scale, where 0 is the lowest level and 100 the highest (for urgency: 0 = no action needed, 100 = act immediately)."],
      ["mean_yes", "Mean P(yes)", "The average probability of \"yes\" across emails: 0.3 means the model leans towards \"yes\" for about 30% of the emails."],
      ["fleiss", "Fleiss κ", "Agreement among all compared runs at once. 1 = identical answers, 0 = no better than chance, below 0 = systematic disagreement. Above 0.6 is usually called substantial agreement."],
      ["answer_counts", "Answer counts", "How often each option was the answer (for categories: how often each was applied). Bars side by side compare the raters."],
    ],
  },
  {
    title: "Agreement between two raters",
    entries: [
      ["agreement", "Agreement", "Share of emails where the two raters gave the same top answer. The range in brackets is the 95% confidence interval: the true value most likely lies in it."],
      ["kappa", "κ (kappa)", "Agreement corrected for luck (Cohen's κ): 1 = perfect, 0 = what chance alone would give. On ordered scales a near miss (\"today\" vs \"this week\") counts less than a far one."],
      ["jsd", "JSD", "How different the two sets of probabilities are (Jensen–Shannon divergence): 0 = identical, 1 = completely different. It also sees disagreement in confidence, not only in the top answer."],
      ["pearson", "r", "How closely the two raters' scores or yes-probabilities move together (Pearson correlation): 1 = in step, 0 = unrelated, −1 = opposite."],
      ["brier", "Brier", "How far a model's probabilities are from the reference answer: 0 = perfect, lower is better. It rewards being confident and right, and penalizes being confident and wrong."],
    ],
  },
  {
    title: "Multi-label categories",
    entries: [
      ["threshold", "Label threshold", "A category counts as applied when its probability is at least this share of the most likely category. At 80%, a category at 0.45 next to a top of 0.5 is applied (0.45 ≥ 0.8 × 0.5); the top category always is."],
      ["labels_per_email", "Labels / email", "The average number of categories applied to one email at the current threshold."],
      ["exact_match", "Exact match", "Share of emails where the two raters applied exactly the same set of categories. The strictest measure."],
      ["jaccard", "Jaccard", "Overlap of the two category sets (shared categories ÷ all categories named), averaged over emails. 1 = identical sets, 0.5 = half overlap, 0 = nothing in common."],
      ["f1", "F1", "Balance of precision (applied categories that are right) and recall (right categories that were applied), over all category decisions. 1 = perfect."],
      ["kappa_macro", "κ (macro)", "Chance-corrected agreement computed for each category separately, then averaged."],
      ["fleiss_multi", "Fleiss κ (mean over labels)", "Fleiss' κ for each category (applied or not), averaged over the categories."],
    ],
  },
  {
    title: "Explorer",
    entries: [
      ["reference", "Reference", "The answer the email's generator intended when it wrote the email. It is the benchmark's ground truth, but it is itself a model's judgment."],
      ["disagreement", "Disagreement", "How much the selected runs disagree about this email, averaged over questions: 0 = all agree, 1 = completely different. Sort by it to find hard or ambiguous emails."],
      ["score", "Score (0–100)", "The chosen level converted to 0–100: 0 = the lowest level of the scale, 100 = the highest."],
      ["human", "Human label", "Your own answer. Once saved, it is compared as a separate rater in the Benchmark report."],
    ],
  },
  {
    title: "Analyze",
    entries: [
      ["analyst_model", "Analyst model", "The AI model that reads all the results and writes the assessment. It does not classify emails itself. Any OpenRouter chat model works; type its id or pick a suggestion."],
      ["disputed_emails", "Disputed emails", "How many of the emails the runs disagree on most are sent to the analyst in full, so it can explain the disagreements. More emails give more evidence but cost more."],
      ["prompts", "Instructions", "What the analyst is told to do. Placeholders such as $report are replaced with the data before sending. Your edits are kept in this browser; \"Reset to default\" restores the text from config/analysis.toml."],
      ["analysis_estimate", "Estimated cost", "An upper bound: the instructions and data at the model's input price, plus the full output budget at its output price. The real cost is usually lower, because the answer is shorter than the budget."],
      ["context_window", "Context window", "The most text the model can read and write in one request. The instructions and data plus the output budget must fit in it."],
      ["run_names", "R1, R2, …", "The analyst refers to the runs by these names; the legend shows which model each one is. \"ref\" is the generator's intended answer and \"human\" your labels."],
    ],
  },
];

const BY_KEY = new Map(GLOSSARY.flatMap((group) => group.entries.map(([key, label, text]) => [key, { label, text }])));

export function describe(key) {
  const entry = BY_KEY.get(key);
  if (!entry) throw new Error(`glossary has no entry ${key}`);
  return entry;
}

export function helpIcon(key) {
  const { label, text } = describe(key);
  return h("span", { class: "help-tip", tabindex: 0, role: "button", "aria-label": `${label}: ${text}`, "data-bs-toggle": "tooltip", "data-bs-title": text }, icon("question-circle"));
}

export function withHelp(text, key) {
  return [text, " ", helpIcon(key)];
}

export function hideTooltips(root) {
  for (const element of root.querySelectorAll('[data-bs-toggle="tooltip"]')) bootstrap.Tooltip.getInstance(element)?.dispose();
}
