/**
 * Pure run-selection helpers: set equality of generation ids and the default run per column.
 * Exports: sameSet, latestCompletedPerColumn.
 */

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
