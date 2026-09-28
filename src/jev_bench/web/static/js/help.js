/**
 * Help page: what the benchmark is, how to read each page, every metric (from glossary.js, the same
 * texts as the tooltips), the methodology, the live question set and how to get an API key.
 * Exports: none (page entry point).
 */
import { api } from "./api.js";
import { clear, h, icon } from "./dom.js";
import { GLOSSARY } from "./glossary.js";
import { initLayout, keySteps, toastError } from "./layout.js";

const TYPE_TEXT = {
  multi: "Multi-label: one or more categories can apply",
  choice: "Single choice",
  score: "Ordered scale (lowest level first) with a 0–100 score",
  noul: "Yes / no",
};

const SECTIONS = [
  ["about", "info-circle", "About this benchmark", [
    "This benchmark compares Jev, TypeSafe's decision model, with a Claude model, a GPT model and an embeddings baseline on the same task: triaging emails. Every model answers the same questions about the same emails, so speed, cost and quality can be compared side by side. All calls go through OpenRouter.",
    "The emails are synthetic. An email-writing model (the generator) writes each one to a recipe, for example \"a phishing email that needs action today\", and records the answers it intended. Those intended answers are the reference: the benchmark's ground truth.",
    "The workflow: create a generation (Generations page); run each model on it (Benchmark page); read the comparison; open individual emails (Explorer); and optionally let an AI analyst write an assessment of all results (Analyze).",
  ]],
  ["benchmark", "speedometer2", "Reading the Benchmark page", [
    "Each card is one column: pick the model, and for chat models whether emails are sent one per request or all in one request. The card shows the estimated cost before you start, and live progress, cost and speed while the run is going.",
    "The large number at the top of a card is the run's quality: its κ against the reference, averaged over all questions, colored from red (chance level) to green (almost perfect), with a second line against your own labels once you have some. The summary table repeats it for every compared run.",
    "Pick which generations to compare and which runs to show. By default the latest completed run of every column is shown. The label threshold slider changes how many categories count as applied; the report recomputes instantly, and nothing is re-run.",
    "The summary table shows each run's time and cost, also relative to Jev (×4 = four times the time or cost of Jev). Each question card shows how often each answer was given, how sure each model was, and how well every pair of raters agrees. Hover over any (?) for an explanation.",
  ]],
  ["explorer", "search", "Reading the Explorer", [
    "The table lists every email with the reference answer and each run's answer for the selected question. Green = same as the reference, amber = partly the same (some categories match), red = different. Sort by disagreement to find the emails the models disagree about most.",
    "Click an email to see its full text and every model's probabilities. Options are ordered with the reference answer first, then by how likely the models found them; unlikely options are folded away. A ✓ marks the reference, bold marks each model's answer, and the small tick on a category bar marks the threshold for that model. You can save your own labels, which then appear as a \"Human\" rater in the report.",
  ]],
  ["analyze", "stars", "The Analyze tab", [
    "Choose the runs to analyse and an AI model, review or edit the instructions, check the estimated cost, and start. The analyst receives the full comparison report, a compact table of every run's answers per email, and the full text of the emails the models disagreed about most. It writes an assessment covering strengths, weaknesses, systematic differences and recommendations. Every analysis is saved with the exact instructions used, so it can be read again later.",
  ]],
];

const METHODOLOGY = [
  ["Synthetic data", "A seeded plan spreads the emails evenly over the categories and varies urgency, length and the share of emails containing hidden instructions aimed at AI systems (prompt injections). Several generator models write the emails, so no single writing style dominates. Each generator also answers the question set about its own email; that is the reference."],
  ["What the models see", "Only the email itself: when it was sent, sender, recipients, subject and body. Never the recipe, the reference answers, the generator model or internal ids. Email bodies are treated as untrusted text."],
  ["Probabilities", "Jev returns probabilities from its Decisions API. Chat models state their probabilities as numbers in a structured JSON answer; these \"verbalized\" probabilities are not necessarily calibrated. The embeddings baseline measures how similar the email is to each option's description and turns the similarities into probabilities with a fixed temperature (not tuned on the reference)."],
  ["Multi-label categories", "Every model gives one probability per category, summing to 1. A category is applied when its probability is at least the threshold times the top category's probability (80% by default), so the most likely category always applies and close runners-up join it. A reference or human label with several categories counts as an even split between them."],
  ["Scores", "Ordered questions (urgency, importance, sentiment, confidentiality) also get a 0–100 score: the probability-weighted average level, scaled so the lowest level is 0 and the highest 100."],
  ["Agreement", "Two raters agree on an email when their top answers match. κ corrects that for luck, and for ordered scales it counts near misses less than far ones. JSD compares the full probability distributions, r compares scores or yes-probabilities, and Brier measures how close a model's probabilities are to the reference. For multi-label categories, exact match, Jaccard, F1 and κ per category compare the applied category sets."],
  ["Confidence intervals", "Ranges in brackets come from bootstrapping: the emails are resampled 1,000 times and the metric recomputed; the range covers the middle 95% of the results. Few emails mean wide ranges, so be careful with small generations."],
  ["Group agreement and disagreement", "Fleiss' κ measures agreement among all compared runs at once (the reference is not included). An email's disagreement index is the average difference (JSD) between every pair of runs, averaged over the questions."],
  ["Cost figures", "Costs are what OpenRouter reports for each request, including the provider's charge when you use your own provider key. Estimates before a run come from earlier runs of the same model when available, otherwise from the text size and list prices (an upper bound)."],
  ["Caveats", "The emails are synthetic and the reference is itself a model's judgment, so \"agrees with the reference\" means \"agrees with the generator\". Chat-model probabilities are self-reported. Compare runs on the same generations, and prefer larger generations for firm conclusions."],
];

async function main() {
  await initLayout();
  const target = document.getElementById("help");
  clear(target, contents(), SECTIONS.map(proseSection), glossarySection(), methodologySection(), questionSection(), keySection());
  fillQuestions().catch(toastError);
}

function card(id, iconName, title, ...body) {
  return h("section", { class: "card shadow-sm mb-3 help-section", id }, h("div", { class: "card-header fw-semibold" }, icon(iconName), ` ${title}`), h("div", { class: "card-body" }, body));
}

function contents() {
  const links = [...SECTIONS.map(([id, , title]) => [id, title]), ["metrics", "Metrics"], ["methodology", "Methodology"], ["questions", "The questions"], ["keys", "API keys"]];
  return h("nav", { class: "d-flex flex-wrap gap-2 mb-3", "aria-label": "Help sections" }, links.map(([id, title]) => h("a", { class: "btn btn-sm btn-outline-primary", href: `#${id}` }, title)));
}

function proseSection([id, iconName, title, paragraphs]) {
  return card(id, iconName, title, paragraphs.map((text) => h("p", {}, text)));
}

function glossarySection() {
  const groups = GLOSSARY.map((group) => h("div", { class: "mb-3" }, h("h3", { class: "h6 text-body-secondary" }, group.title), h("dl", { class: "row mb-0" }, group.entries.flatMap(([, label, text]) => [h("dt", { class: "col-sm-3" }, label), h("dd", { class: "col-sm-9" }, text)]))));
  return card("metrics", "book", "Metrics", groups);
}

function methodologySection() {
  return card("methodology", "diagram-3", "Methodology", METHODOLOGY.map(([title, text]) => h("div", { class: "mb-2" }, h("h3", { class: "h6 mb-1" }, title), h("p", { class: "mb-0" }, text))));
}

function questionSection() {
  return card("questions", "list-check", "The questions", h("div", { id: "question-list" }, h("p", { class: "text-body-secondary small" }, "Loading…")));
}

function keySection() {
  const override = "Enter your key with the key button in the top bar. It is stored only in your browser, sent only when you start a run, a generation or an analysis, and it takes precedence over the server's key.";
  return card("keys", "key", "API keys", keySteps(), h("p", { class: "mb-0 mt-2" }, override));
}

async function fillQuestions() {
  const questionSet = await api.questions();
  const items = questionSet.questions.map(questionItem);
  clear(document.getElementById("question-list"), h("p", { class: "small text-body-secondary" }, `Question set ${questionSet.name}: ${items.length} questions.`), items);
}

function questionItem(question) {
  const kind = question.type === "multi" ? `${TYPE_TEXT.multi} (threshold ${Math.round(question.threshold * 100)}% of the top)` : TYPE_TEXT[question.type];
  const options = Object.entries(question.options).map(([option, text]) => h("li", {}, h("code", {}, option), ` — ${text}`));
  return h("details", { class: "mb-2" }, h("summary", {}, h("code", { class: "fw-semibold" }, question.id), " ", h("span", { class: "badge text-bg-light border" }, question.type), ` ${question.instructions}`), h("p", { class: "small text-body-secondary mb-1 mt-2" }, kind), h(question.type === "score" ? "ol" : "ul", { class: "small mb-0" }, options));
}

main().catch(toastError);
