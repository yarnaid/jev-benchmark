/**
 * Pure run-selection helpers shared by the Benchmark and Analyze pages: set equality of generation ids,
 * the initial generations (pinned by a requested run, else the stored choice, else the newest), the
 * default runs (the latest completed run per column not hidden in a card slot, in catalog column order),
 * and the checklist items
 * of the generation and run pickers.
 * Exports: sameSet, latestCompletedPerColumn, orderByColumn, defaultRunIds, initialGenerations,
 * generationChoices, runChoices, runText.
 */
import { shortModel, when } from "./format.js";

export const sameSet = (values, set) => values.length === set.size && values.every((value) => set.has(value));

export function latestCompletedPerColumn(runs, generationIds) {
  const wanted = new Set(generationIds);
  const chosen = new Map();
  const newestFirst = [...runs].sort((a, b) => b.id.localeCompare(a.id));
  for (const run of newestFirst) {
    if (run.status === "completed" && sameSet(run.generation_ids, wanted) && !chosen.has(run.column)) chosen.set(run.column, run.id);
  }
  return [...chosen.values()];
}

export function orderByColumn(runIds, runs, catalog) {
  const byId = new Map(runs.map((run) => [run.id, run]));
  const rank = (id) => {
    const index = catalog.findIndex((column) => column.id === byId.get(id)?.column);
    return index === -1 ? catalog.length + (byId.has(id) ? 0 : 1) : index;
  };
  return [...runIds].sort((a, b) => rank(a) - rank(b));
}

export function defaultRunIds(runs, generationIds, catalog, hidden = new Set()) {
  const shown = runs.filter((run) => !hidden.has(run.column));
  return orderByColumn(latestCompletedPerColumn(shown, generationIds), runs, catalog);
}

export function initialGenerations(generations, runs, requested, stored) {
  const pinned = runs.find((run) => requested.includes(run.id));
  if (pinned) return [...pinned.generation_ids];
  const known = new Set(generations.map((generation) => generation.id));
  const kept = (stored ?? []).filter((id) => known.has(id));
  return kept.length ? kept : generations.slice(0, 1).map((generation) => generation.id);
}

export const generationChoices = (generations) => generations.map((generation) => ({ value: generation.id, text: `${generation.name} · ${generation.done} emails`, hint: `${generation.id} · ${generation.status}` }));

export function runText(run, catalog) {
  const title = catalog.find((column) => column.id === run.column)?.title ?? run.column;
  return `${title} · ${shortModel(run.model)}${run.mode === "all_in_one" ? " · all in one" : ""}`;
}

export function runChoices(runs, generationIds, catalog) {
  const wanted = new Set(generationIds);
  return runs
    .filter((run) => run.status === "completed" && sameSet(run.generation_ids, wanted))
    .map((run) => ({ value: run.id, text: runText(run, catalog), hint: `${when(run.created_at)} · ${run.n_done} emails` }));
}
