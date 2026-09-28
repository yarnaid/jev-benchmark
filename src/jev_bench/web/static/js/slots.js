/**
 * Pure card-slot helpers. Columns sharing a `slot` in config/benchmark.toml share one Benchmark card
 * position and are swapped with a toggle in the card header (Kev swaps with Embeddings). A slot sits where
 * its first column appears in catalog order; its visible column is the stored pick when that is still a
 * member of the slot, else the slot's first column. Picks are read through a caller-supplied reader so this
 * module stays free of storage and DOM.
 * Exports: slotGroups, slotView, visibleColumns, hiddenColumnIds, slotPrefKey, slotPicks.
 */

export function slotGroups(catalog) {
  const groups = new Map();
  for (const column of catalog) {
    const slot = column.slot ?? column.id;
    if (!groups.has(slot)) groups.set(slot, []);
    groups.get(slot).push(column);
  }
  return [...groups].map(([slot, columns]) => ({ slot, columns }));
}

export const slotView = (catalog, picks) =>
  slotGroups(catalog).map(({ slot, columns }) => ({ slot, columns, shown: columns.find((column) => column.id === picks[slot]) ?? columns[0] }));

export const visibleColumns = (catalog, picks) => slotView(catalog, picks).map((view) => view.shown);

export function hiddenColumnIds(catalog, picks) {
  const shown = new Set(visibleColumns(catalog, picks).map((column) => column.id));
  return new Set(catalog.map((column) => column.id).filter((id) => !shown.has(id)));
}

export const slotPrefKey = (slot) => `benchmark.slot.${slot}`;

export const slotPicks = (catalog, read) => Object.fromEntries(slotGroups(catalog).map(({ slot }) => [slot, read(slotPrefKey(slot))]));
