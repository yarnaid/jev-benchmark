/**
 * Pure text of the navbar badge on the static snapshot: the build date and short commit, linked to it.
 * Exports: snapshotInfo.
 */
const TITLE = "A read-only snapshot of the committed results; run jev-bench locally to start runs";

export function snapshotInfo(manifest, repoUrl) {
  const date = String(manifest.built_at ?? "").slice(0, 10);
  if (!manifest.commit) return { text: `Snapshot · ${date}`, href: null, title: TITLE };
  return { text: `Snapshot · ${manifest.commit.slice(0, 7)} · ${date}`, href: `${repoUrl}/commit/${encodeURIComponent(manifest.commit)}`, title: TITLE };
}
