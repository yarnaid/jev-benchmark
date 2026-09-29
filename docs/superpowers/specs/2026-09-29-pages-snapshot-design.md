# Read-only GitHub Pages snapshot

Publishes the committed results (`data/`) as a static, read-only site on GitHub Pages at
`https://yarnaid.github.io/jev-benchmark/`, rebuilt by GitHub Actions on every push to `main`. Visitors
browse the Benchmark comparison, the Explorer, the Runs and Generations lists, saved analyses and Help
for a fixed set of run selections at every label threshold. The site never starts a job, never holds an
API key and never edits labels.

## Revision (2026-09-29, while planning)

Found while reading the code for the implementation plan; they supersede the sections below:

- **Offline catalog.** `GET /catalog` fetches OpenRouter's model list (`services.catalog.for_column`), so a
  network-refusing export would crash on it. The exporter replaces `services.catalog` with an
  `OfflineCatalog` that lists no models; the route then offers each column's default model only (the
  pickers are hidden on the site anyway). Every other exported route was probed offline on the real
  `data/`: all answer 200 and `data/` stays unchanged.
- **Content-addressed view ids.** GitHub Pages serves every file with `Cache-Control: max-age=600`. With
  `v1`, `v2`, … a browser holding the previous deploy's `site.json` could read a new deploy's files under an
  id that now means another selection, and show the wrong runs without any error. A view id is therefore
  `v` + the first 12 hex digits of the SHA-256 of its sorted generation ids and sorted run ids, so an id
  always means the same selection; the UI fetches `api/site.json` with `cache: "no-cache"`. Stale ES
  modules for up to 10 minutes after a deploy remain a known limitation (a reload fixes them).
- **One base-file rule.** Every request that does not depend on a view maps to `api<path>.json`, so
  `/analysis/defaults` becomes `api/analysis/defaults.json` (not `analysis-defaults.json`).
- **Run order.** A view lists its run ids in catalog column order, the order the Benchmark requests them
  (`orderByColumn`); the compare report lists raters in request order.
- **Analyses referencing missing data.** An analysis whose generations or runs are not all in `data/` gives
  no view (a warning is logged) instead of failing the export.
- **More root-absolute URLs** than listed in §2: `generations.js` (Explore link) and `report-summary.js`
  (Explorer link of the summary table). Both become relative.
- **Module split.** `site/` also gets `errors.py` (`ExportError`), `offline.py` (offline settings,
  refusing client, `OfflineCatalog`) and `writer.py` (fetch one planned request, write its file, refuse
  paths outside `OUT`). `export_site` takes the UI source directory as a parameter (default `STATIC_DIR`),
  so its test copies a two-file fake UI. The CLI body lives in `cli_export.py`: an export is not a job,
  and `cli_jobs.py` is about jobs.

## 1. Decisions

| Topic | Decision |
|---|---|
| Interactivity | Fixed views, all precomputed by the existing Python. Arbitrary run subsets are out: 9 runs are 512 subsets, doubling per run, and Fleiss' κ and the disagreement index depend on the whole set. Running the compare in the browser (Pyodide) was rejected: a 15–25 MB first load, and Python 3.14 + pydantic-core support is unverified. |
| Static mapping | A manifest (`api/site.json`) plus an adapter in `api.js` that turns a GET into a file path. A service worker (first load not intercepted, stale caches across deploys) and hash-named files (same canonicalisation on both sides, opaque names) were rejected. |
| Source of the JSON | The real FastAPI routes, called in-process through `httpx2.ASGITransport`. Bytes are written unchanged, so the site shows exactly what the local server returns. |
| Views | Derived from `data/`, never configured, so a newly committed run appears without edits. |
| Paths | The UI switches to relative URLs everywhere (both modes), instead of rewriting paths at export time. |
| Trigger | `push` to `main` only (plus `workflow_dispatch` for the first deploy); every run builds and deploys. Other branches do not build. |
| CSP | Pages cannot set headers, so every exported HTML file gets a `<meta>` CSP built from `web.security.CSP` (still the one source), minus `frame-ancestors` (ignored in a meta tag), plus `<meta name="referrer" content="no-referrer">`. |

## 2. Facts measured on the current data (2026-09-29)

- `data/` holds 1 generation (`20260925-152233-generation-b0fc`, 100 emails), 9 runs and 1 analysis. 4 runs
  belong to generation `20260925-111142-generation-67ba`, which is not committed; the UI never shows them
  (their generation cannot be selected), and the export leaves them out the same way.
- Through `httpx2.ASGITransport` with `app.state.services` set by hand (no lifespan): `/api/compare` for the
  4 latest runs takes about 350 ms and returns 104 KB; `/api/emails` for the same set takes about 100 ms and
  returns 350 KB; `/api/runs` takes 4 ms. `data/` is unchanged afterwards.
- The Benchmark and Explorer sliders take their default from the first `multi` question of the default
  response; `widgets.thresholdRange(0.8)` gives 50–100 in steps of 5 (11 values).
- Every UI request goes through `request()` in `js/api.js`, which fetches `/api${path}`. All HTML assets
  (`/js/…`, `/css/app.css`), the navbar (`layout.js` `PAGES`, brand `href: "/"`), `runs.js` `go(...)` and
  `analysis-links.js` use root-absolute URLs; a Pages project site lives under `/jev-benchmark/`.

## 3. What gets published

### 3.1 Views

A view is a `(generation_ids, run_ids)` pair; `run_ids` may be empty. The exporter derives, then merges
views with equal generation and run sets (sets, not lists):

- for each committed generation with at least one completed run on exactly that generation:
  - **Benchmark "Latest"**, once per slot state: the latest completed run per column not hidden by that
    state (`selection.latestCompletedPerColumn` over `slots.hiddenColumnIds` for every combination of
    slot picks). Today: Jev + Claude + GPT + Embeddings, and Jev + Claude + GPT + Kev;
  - **Explorer default**: the latest completed run per column over all columns (Explorer ignores slots).
    Today: all five;
- for each committed generation: the **generation alone** with no runs (Explorer with nothing selected, or
  a generation without runs);
- for each completed analysis: its `generation_ids` and `run_ids` (its `e001` links open the Explorer on
  them). Today it equals Latest with Embeddings.

Today this gives 4 views (Latest · Embeddings = the analysis, Latest · Kev, All columns, the generation
alone), about 90 threshold files and 400 email details. A view whose generation is not committed is
skipped. View ids are content hashes (see the revision above), listed in derivation order; the
label names the rule (for example "Latest without kev", "All columns", "Analysis <id>").

### 3.2 Thresholds

For each view and each threshold-dependent endpoint (`/compare`, `/emails`) the exporter first fetches the
response without `threshold` (`t-default`), reads the `threshold` of the first `multi` question in it, and
then fetches every step of `threshold_steps(default)`, the Python mirror of `widgets.thresholdRange`. A
view whose response has no `multi` question gets `t-default` only.

### 3.3 File layout

| Request | File |
|---|---|
| `GET /status`, `/catalog`, `/questions`, `/generations`, `/runs`, `/analyses`, `/analysis/defaults` | `api/<name>.json` (`analysis/defaults` → `api/analysis/defaults.json`) |
| `GET /generations/{id}`, `/runs/{id}`, `/analyses/{id}` | `api/generations/<id>.json`, `api/runs/<id>.json`, `api/analyses/<id>.json` |
| `GET /analyses/{id}/prompts` | `api/analyses/<id>/prompts.json` |
| `GET /compare?runs=R` / `&threshold=t` | `api/compare/<view>/t-default.json` / `t<round(100·t)>.json`, only for views with at least one run (the Benchmark never compares zero runs) |
| `GET /emails?generations=G&runs=R` / `&threshold=t` | `api/emails/<view>/t-default.json` / `t<round(100·t)>.json` |
| `GET /emails/{id}?runs=R` | `api/email/<view>/<email_id>.json`, for every email of the view's generations |

`/runs?generations=…` is not published (no page sends it).

### 3.4 Manifest

`api/site.json`:

```json
{
  "built_at": "2026-09-29T12:00:00Z",
  "commit": "9c3c2956489458b38926ecb4b2f90af55a593a0e",
  "views": [{"id": "v1", "label": "Latest · Embeddings", "generation_ids": ["…"], "run_ids": ["…"]}]
}
```

`commit` is `null` for a local export without `--commit`.

### 3.5 Refusals

The export fails, writing nothing further, when:
- any generation, run or analysis in `data/` has status `running`, since the page would poll it forever;
- any planned request answers other than 200;
- `OUT` exists and is not empty.

## 4. Components

### 4.1 Python: `src/jev_bench/site/`

| Module | Responsibility |
|---|---|
| `__init__.py` | Empty. |
| `views.py` | `View` (pydantic: `id`, `label`, `generation_ids`, `run_ids`); `derive_views(catalog, generations, runs, analyses) -> list[View]` per §3.1. The Python mirror of `latestCompletedPerColumn` + `hiddenColumnIds`. |
| `thresholds.py` | `threshold_steps(default: float) -> list[int]`, the mirror of `widgets.thresholdRange`. |
| `paths.py` | The Python half of the file contract: endpoint + view + threshold → relative file path (§3.3). |
| `plan.py` | Views → the ordered list of `(url, file)` pairs, including the default-first threshold rule of §3.2. |
| `export.py` | `export_site(settings, out, commit) -> ExportSummary` (files, bytes, seconds): guards (§3.5), fetches every planned URL, writes the bytes, writes `api/site.json`. |
| `static_copy.py` | Copies `web/static/` into `OUT`, overwrites `js/deployment.js` with `export const STATIC = true;`, and inserts the CSP and referrer meta tags after `<head>` in every HTML file. |

`export_site`:
- builds `create_app(settings, http=<refusing client>)`, where the client's `httpx2.MockTransport` raises on
  every request, so an export can never reach the network;
- sets `app.state.services = Services(settings, client)` directly: the lifespan (and so
  `sweep_interrupted`) never runs and `data/` is only read;
- receives `settings` with `openrouter_api_key=None` (the CLI clears it), so no key is in memory and
  `status.json` always reports no server key;
- fetches sequentially (about 20 s today); parallelism waits for a measured need.

### 4.2 CLI

`jev-bench export-site OUT [--commit SHA]` in `cli.py`, with lazy imports so `--help` stays under 500 ms.
It calls `configure_logging()` first like every entry point, prints a green `N files · X MB · Ys` summary
with `rich`, and on a refusal or a failed request prints the reason (and the URL) in red and exits 1.

### 4.3 UI

1. **Relative URLs, both modes.** HTML: `js/…`, `css/app.css`. `layout.js`: `PAGES` hrefs `./`,
   `generations.html`, …; the brand links to `./`; the active page is matched on the last path segment
   (empty → `index.html`). `runs.js` `go("./")` / `go("explorer.html")`; `analysis-links.js` returns
   `explorer.html?…`; `api.js` fetches `` `api${path}` ``.
2. **`js/deployment.js`**: `export const STATIC = false;`, nothing else. The exporter overwrites it.
3. **`js/static-api.js`** (pure): `staticFile(method, path, manifest)` returns the relative file path per
   §3.3, matching the view by set equality of generation and run ids (order-insensitive), or throws for a
   non-GET or an unpublished request. The JS half of the file contract.
4. **`api.js`**: when `STATIC`, `request()` loads `api/site.json` once (a cached promise), maps the request
   with `staticFile` and fetches that file; a throw becomes `ApiError(404, "Not published in this
   snapshot")`. Live mode is unchanged.
5. **Static chrome.** `layout.js` sets `data-static` on `<html>`; `app.css` gets
   `[data-static] .write-only { display: none !important; }`. Tagged `write-only`: the key badge and
   dialog, the estimate blocks and the start/cancel buttons on column cards, the new-generation form, the
   analysis setup card, the label editor in the email detail, and the Runs page's compare/explore
   controls. Pages skip the calls behind them when `STATIC` (run and analysis estimates), so nothing fails
   in the background. The navbar shows a "Snapshot · `<short sha>` · `<date>`" badge linking to the
   commit (plain "Snapshot · `<date>`" when `commit` is null).
6. **Pickers.** `widgets.checklist` gets a `disabled` option; the Benchmark and Explorer run checklists are
   disabled in static mode and still show the runs of the view. The slot toggle and the threshold slider
   work unchanged: every state they reach is published. A selection that is not published (for example two
   generations at once) shows an empty state "Not in this snapshot" instead of an error toast.

### 4.4 Workflow: `.github/workflows/pages.yml`

```yaml
on:
  push: { branches: [main] }
  workflow_dispatch:
permissions: { contents: read, pages: write, id-token: write }
concurrency: { group: pages, cancel-in-progress: false }
jobs:
  build:    # ubuntu-latest, timeout-minutes: 10
    # checkout → setup-uv (cache on) → uv sync --locked --no-dev
    # → uv run --no-sync jev-bench export-site _site --commit "$GITHUB_SHA"
    # → upload-pages-artifact (path: _site)
  deploy:   # needs: build; environment github-pages (url from the deploy step); deploy-pages
```

Every action is pinned to a full commit SHA with a `# vX.Y.Z` comment; the versions are looked up from the
actions' current releases at implementation time. uv installs the Python 3.14 that `requires-python`
asks for.

**One-time repository setting**: Pages source "GitHub Actions" (Settings → Pages, or
`gh api -X POST repos/yarnaid/jev-benchmark/pages -f build_type=workflow`). It changes a public repository
setting, so it is run only on the owner's explicit go-ahead. Until it is set, the deploy job fails.

## 5. Error handling

| Situation | Behavior |
|---|---|
| Running job in `data/` | Export exits 1 naming the job; the deploy does not happen and the old site stays up. |
| A planned request answers non-200 | Export exits 1 with the URL and status. |
| `OUT` not empty | Export exits 1 without touching it. |
| Any outbound HTTP during export | The refusing transport raises; export exits 1. |
| Static page asks for an unpublished combination or a write | `ApiError(404, "Not published in this snapshot")`; pages show an empty state, write controls are hidden anyway. |

## 6. Invariants kept

- Keys: the exporter never has one (cleared settings), and a test greps `OUT` for a sentinel key.
- Models' view of emails is untouched (the export only calls GET routes).
- UI: no `innerHTML` family, no inline scripts; the meta tags are not scripts. `test_web_static.py`'s SRI and
  pinned-CDN checks cover the exported copy's sources as well (same files).
- `data/` stays read-only during an export (no lifespan, no sweep).
- The question set remains the single source of truth: every exported JSON comes from the routes.

## 7. Testing

- **pytest**, one file per module:
  - `test_site_views.py`: slot states, the all-columns view, the generation-alone view, analysis views,
    set-merging, orphan runs skipped, non-completed runs ignored;
  - `test_site_thresholds.py` and `test_site_paths.py`, both reading the shared fixtures below;
  - `test_site_plan.py`: default-first ordering, views without a `multi` question;
  - `test_site_static_copy.py`: `deployment.js` flipped, meta tags present once per page, no
    `frame-ancestors`;
  - `test_site_export.py` on a mini data dir from `tests/factories.py`: files equal the route bytes; a
    sentinel key appears nowhere in `OUT`; the refusing transport is never hit; a `running` job and a
    non-empty `OUT` abort;
  - a new row in the CLI test table.
- **Shared contract fixtures** in `tests/fixtures/`, read by both pytest and `node --test`:
  `site_paths.json` (request → file), `default_views.json` (catalog + runs + slot picks → run ids),
  `threshold_steps.json` (default → steps). They keep the Python and JS copies of the three rules equal.
- **Node**: `tests/js/static-api.test.js`, and contract tests running `selection.js`, `slots.js` and
  `widgets.thresholdRange` on the same fixtures.
- **`test_web_static.py`**: a new check that no HTML or JS uses a root-absolute URL.
- **Done when**:
  1. a local `jev-bench export-site`, served under a `/jev-benchmark/` prefix, passes a Playwright walk of
     all six pages, both slot states and both threshold extremes with 0 console errors and 0 404s;
  2. after the first deploy, the same walk passes on `https://yarnaid.github.io/jev-benchmark/`;
  3. the default suite stays under 5 s and coverage at 95% or more.

## 8. Docs

`CLAUDE.md`: an architecture bullet for `site/` and the static mode, and the `export-site` command.
`README.md`: a "Published snapshot" section with the URL and how it is built.

## 9. Out of scope

- Arbitrary run subsets and custom thresholds beyond the slider steps.
- Human-label editing, new generations, runs or analyses on the site.
- Preview deploys of other branches.
- CI checks (lint, types, tests) in Actions.
- A custom domain.
