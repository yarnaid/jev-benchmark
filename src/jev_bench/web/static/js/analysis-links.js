/**
 * Pure helper of the Analyze page: the Explorer link behind an e001… ref in an analysis (the email with
 * the analysed generations and runs).
 * Exports: explorerHref.
 */

export function explorerHref(meta, ref) {
  const refs = meta.email_refs ?? {};
  if (!Object.hasOwn(refs, ref)) return null;
  const query = new URLSearchParams({ generations: meta.generation_ids.join(","), runs: meta.run_ids.join(","), email: refs[ref] });
  return `/explorer.html?${query}`;
}
