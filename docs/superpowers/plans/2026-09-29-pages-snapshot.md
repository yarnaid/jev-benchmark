# Read-only GitHub Pages Snapshot Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Publish the committed results in `data/` as a static, read-only site at
`https://yarnaid.github.io/jev-benchmark/`, rebuilt by GitHub Actions on every push to `main`.

**Architecture:** A new `jev_bench.site` package derives the published views from `data/`, runs every GET
the UI needs against the real FastAPI app in-process (`httpx2.ASGITransport`, no lifespan, network
refused, no key) and writes each response body to a file, next to a copy of the UI switched to static
mode. In static mode `js/api.js` maps each GET to its file through the manifest `api/site.json` (pure
`js/static-api.js`); write controls are hidden. The UI uses relative URLs so it works under the local `/`
mount and under `/jev-benchmark/`.

**Tech Stack:** Python 3.14, FastAPI, httpx2, pydantic, typer + rich, loguru; build-free ES modules
(Bootstrap 5.3); pytest, `node --test`; GitHub Actions (`setup-uv`, `upload-pages-artifact`,
`deploy-pages`).

**Spec:** `docs/superpowers/specs/2026-09-29-pages-snapshot-design.md` (read its "Revision (2026-09-29,
while planning)" section first: it supersedes parts of §3–§4).

## Global Constraints

- Python `>= 3.14`, deps through `uv`; no new dependencies. `httpx2`, never `httpx`.
- All repository content in English.
- Every Python file starts with a docstring listing its classes and functions; no inline comments; full
  type annotations; pydantic models (or `NamedTuple`/`Protocol`) for known shapes; `loguru` for logs,
  `rich` for CLI output.
- UI: never `innerHTML` / `outerHTML` / `insertAdjacentHTML` / `document.write` (build DOM with `h()`);
  no inline scripts; CDN assets pinned with SRI (`tests/test_web_static.py`).
- Keys are never persisted or logged; the exporter never holds one (`openrouter_api_key=None`).
- `data/` is only read by an export (no lifespan, no `sweep_interrupted`).
- Tests: one file per module; tables via `pytest.param(..., id=...)`; OpenRouter always mocked; each test
  < 50 ms (global `timeout = 1`); default suite < 5 s; `uv run pytest --cov --cov-fail-under=95` passes.
- After every task: `uv run ruff check --fix && uv run ruff format && uv run pyright` (0 errors),
  `uv run pytest`, `node --check src/jev_bench/web/static/js/*.js`, `node --test tests/js/`.
- `jev-bench --help` < 500 ms: nothing heavy imported at the top of `cli.py`.
- One commit per task, message `type(scope): subject`, ending with
  `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- Branch: `feat/pages-snapshot`.

## Review Focus

1. **A redeploy while a visitor has cached files** (Pages: `max-age=600`) must never show another
   selection's data: view ids are content hashes (Task 3, `test_view_id_is_an_order_insensitive_content_hash`)
   and the manifest is revalidated (Task 11, "the manifest is fetched once and revalidated").
2. **The selection a page makes on first load must be published**, for example Explorer's first generation
   with no runs, and both Benchmark slot states: Task 3 table rows `slot-states-and-all-columns` and
   `generation-without-runs`; Task 15 checks every page's first load in a browser.
3. **Float noise in thresholds** (`0.29 * 100 = 28.999…`) must map to the same file on both sides:
   Task 2 rows `float-noise` and `half-rounds-up`, Task 4 row `compare-float-noise`.
4. **Any root-absolute URL left behind** 404s under `/jev-benchmark/`: Task 1's HTML and JS checks, and
   Task 15's browser walk counts 404s.
5. **An analysis that references a run or generation missing from `data/`** must not break the export:
   Task 3 rows `analysis-missing-run`, `analysis-missing-generation` and `analysis-not-completed`.

---

## File Structure

| Path | Responsibility |
|---|---|
| `src/jev_bench/site/__init__.py` | Empty package marker. |
| `src/jev_bench/site/thresholds.py` | `threshold_steps`: slider steps, the mirror of `widgets.thresholdRange`. |
| `src/jev_bench/site/views.py` | `View` and `derive_views`: the published selections (mirror of `selection.js` + `slots.js`). |
| `src/jev_bench/site/paths.py` | Python half of the request → file contract. |
| `src/jev_bench/site/plan.py` | `SiteRequest` and the request builders; `multi_threshold`. |
| `src/jev_bench/site/errors.py` | `ExportError`. |
| `src/jev_bench/site/offline.py` | Offline settings, refusing HTTP client, `OfflineCatalog`, `offline_services`. |
| `src/jev_bench/site/static_copy.py` | Copy the UI, flip `deployment.js`, insert CSP/referrer meta tags. |
| `src/jev_bench/site/writer.py` | `SiteWriter`: fetch one planned GET, write its file, refuse escapes. |
| `src/jev_bench/site/export.py` | `export_site`, `ExportSummary`, `SiteManifest`: the orchestration. |
| `src/jev_bench/cli_export.py` | Async body of `jev-bench export-site`. |
| `src/jev_bench/cli.py` | New `export-site` command (modify). |
| `src/jev_bench/web/static/js/deployment.js` | `STATIC` flag (false here; true in the snapshot). |
| `src/jev_bench/web/static/js/static-api.js` | JS half of the request → file contract. |
| `src/jev_bench/web/static/js/snapshot.js` | Navbar badge text of the snapshot. |
| `src/jev_bench/web/static/js/api.js` | Relative fetch; static branch (modify). |
| other `web/static/**` | Relative URLs; `write-only` tags; disabled run pickers; not-published empty states (modify). |
| `tests/fixtures/threshold_steps.json`, `default_views.json`, `site_paths.json` | Shared Python/JS contract cases. |
| `.github/workflows/pages.yml` | Build and deploy on push to `main`. |

---

### Task 1: Relative URLs in the UI

**Files:**
- Modify: `src/jev_bench/web/static/{index,generations,explorer,runs,analyze,help}.html` (+ any other `*.html`)
- Modify: `src/jev_bench/web/static/js/layout.js:17-24,36,48`
- Modify: `src/jev_bench/web/static/js/runs.js:19-27`
- Modify: `src/jev_bench/web/static/js/analysis-links.js:11`
- Modify: `src/jev_bench/web/static/js/generations.js:98`
- Modify: `src/jev_bench/web/static/js/report-summary.js:38`
- Modify: `src/jev_bench/web/static/js/api.js:20`
- Test: `tests/test_web_static.py`, `tests/js/api.test.mjs`

**Interfaces:**
- Produces: every page, script and API URL is relative to the page (`js/…`, `css/…`, `explorer.html`,
  `./`, `api/…`). Later tasks rely on `fetch(\`api${path}\`)` in `api.js`.

- [ ] **Step 1: Write the failing tests**

In `tests/test_web_static.py`, add the pattern next to `FORBIDDEN_SINKS`:

```python
ROOT_ABSOLUTE = re.compile(r"""["'`]/(?:api\b|[\w-]+\.html)|href:\s*["'`]/["'`]|go\(\s*["'`]/""")
```

Replace the last `if` block of `test_pages_follow_the_csp_contract` with:

```python
        if url and "://" not in url and not url.startswith("data:"):
            assert not url.startswith("/"), f"root-absolute URL {url}"
            assert (STATIC_DIR / url).exists(), f"missing local asset {url}"
```

Add after `test_module_imports_resolve`:

```python
def test_javascript_uses_relative_urls() -> None:
    offenders = [
        f"{path.name}: {match.group(0)}"
        for path in sorted(STATIC_DIR.glob("js/*.js"))
        for match in ROOT_ABSOLUTE.finditer(path.read_text(encoding="utf-8"))
    ]
    assert offenders == []
```

In `tests/js/api.test.mjs`, append:

```js
test("requests are relative to the page, so the UI works under a sub-path", async () => {
  sent.length = 0;
  await api.catalog();
  assert.equal(sent[0].url, "api/catalog");
});
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_web_static.py -q` and `node --test tests/js/api.test.mjs`
Expected: FAIL. The HTML check reports `root-absolute URL /css/app.css`; the JS check lists `layout.js`,
`runs.js`, `analysis-links.js`, `generations.js`, `report-summary.js` and `api.js`; the node test gets
`/api/catalog`.

- [ ] **Step 3: Make every URL relative**

HTML (all pages):

```bash
sed -i '' -e 's#href="/css/#href="css/#' -e 's#src="/js/#src="js/#' src/jev_bench/web/static/*.html
```

`layout.js`:

```js
const PAGES = [
  ["./", "Benchmark", "speedometer2"],
  ["generations.html", "Generations", "envelope-paper"],
  ["explorer.html", "Explorer", "search"],
  ["runs.html", "Runs", "clock-history"],
  ["analyze.html", "Analyze", "stars"],
  ["help.html", "Help", "question-circle"],
];
```

In `navbar()`, replace the `here` line and the brand `href`:

```js
  const page = location.pathname.split("/").pop() || "index.html";
  const here = page === "index.html" ? "./" : page;
```

```js
      h("a", { class: "navbar-brand fw-semibold", href: "./" }, icon("bar-chart-steps"), " jev-bench"),
```

`runs.js`:

```js
  document.getElementById("compare").addEventListener("click", () => go("./"));
  document.getElementById("explore").addEventListener("click", () => go("explorer.html"));
```

```js
  if (page !== "./") params.set("generations", [...new Set(chosen.flatMap((run) => run.generation_ids))].join(","));
```

`analysis-links.js`: `` return `explorer.html?${query}`; ``

`generations.js:98`: `` href: `explorer.html?generations=${encodeURIComponent(meta.id)}` ``

`report-summary.js:38`: `` const explore = `explorer.html?${new URLSearchParams({ … })}`; `` (only the leading
`/` goes).

`api.js:20`: `` const response = await fetch(`api${path}`, { … }); ``

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_web_static.py -q && node --check src/jev_bench/web/static/js/*.js && node --test tests/js/`
Expected: PASS (the existing `analysis-links` test still resolves `explorer.html?…` to `/explorer.html`
against `http://host.test`).

- [ ] **Step 5: Check the live server by hand**

Run `uv run jev-bench serve --port 8123` in the background (stop it afterwards). With the Playwright MCP
tools, open `http://127.0.0.1:8123/`, `…/explorer.html` and `…/runs.html`. Expected: the pages render,
the active nav link is highlighted, and there are no console errors and no 404s
(`browser_network_requests`).

- [ ] **Step 6: Commit**

```bash
git add src/jev_bench/web/static tests/test_web_static.py tests/js/api.test.mjs
git commit -m "refactor(ui): relative URLs so the UI also works under a sub-path"
```

---

### Task 2: Threshold steps (Python mirror of `thresholdRange`)

**Files:**
- Create: `src/jev_bench/site/__init__.py` (empty)
- Create: `src/jev_bench/site/thresholds.py`
- Create: `tests/fixtures/threshold_steps.json`
- Create: `tests/js/site-contract.test.mjs`
- Test: `tests/test_site_thresholds.py`

**Interfaces:**
- Produces: `threshold_steps(default: float) -> list[int]`, every percent the slider can produce, ascending.

- [ ] **Step 1: Write the shared cases and both failing tests**

`tests/fixtures/threshold_steps.json`:

```json
[
  {"id": "default-80", "default": 0.8, "min": 50, "step": 5},
  {"id": "one-hundred", "default": 1, "min": 50, "step": 5},
  {"id": "below-50", "default": 0.4, "min": 40, "step": 5},
  {"id": "odd-below-50", "default": 0.37, "min": 35, "step": 1},
  {"id": "odd-above-50", "default": 0.73, "min": 50, "step": 1},
  {"id": "tiny", "default": 0.01, "min": 1, "step": 1},
  {"id": "float-noise", "default": 0.29, "min": 25, "step": 1},
  {"id": "half-rounds-up", "default": 0.125, "min": 10, "step": 1}
]
```

`tests/test_site_thresholds.py`:

```python
"""Tests for jev_bench.site.thresholds against the shared cases in tests/fixtures."""

import json
from pathlib import Path

import pytest

from jev_bench.site.thresholds import threshold_steps

CASES = json.loads(
    (Path(__file__).parent / "fixtures" / "threshold_steps.json").read_text(encoding="utf-8")
)


@pytest.mark.parametrize(
    ("default", "low", "step"),
    [pytest.param(case["default"], case["min"], case["step"], id=case["id"]) for case in CASES],
)
def test_threshold_steps_match_the_slider(default: float, low: int, step: int) -> None:
    assert threshold_steps(default) == list(range(low, 101, step))
```

`tests/js/site-contract.test.mjs`:

```js
// Run with `node --test tests/js/`. Checks the UI rules that jev_bench.site mirrors in Python against the
// shared cases in tests/fixtures.
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

const { thresholdRange } = await import("../../src/jev_bench/web/static/js/widgets.js");
const fixture = (name) => JSON.parse(readFileSync(new URL(`../fixtures/${name}`, import.meta.url), "utf-8"));

for (const { id, default: value, min, step } of fixture("threshold_steps.json")) {
  test(`thresholdRange matches the shared case ${id}`, () => {
    const range = thresholdRange(value);
    assert.deepEqual({ min: range.min, max: range.max, step: range.step }, { min, max: 100, step });
  });
}
```

- [ ] **Step 2: Run them**

Run: `node --test tests/js/site-contract.test.mjs`. Expected: PASS (the JS rule already exists; this pins
the fixture).
Run: `uv run pytest tests/test_site_thresholds.py -q`. Expected: FAIL with
`ModuleNotFoundError: No module named 'jev_bench.site'`.

- [ ] **Step 3: Implement**

`src/jev_bench/site/__init__.py`: empty file.

`src/jev_bench/site/thresholds.py`:

```python
"""Label-threshold slider steps of the static snapshot: the Python mirror of `widgets.thresholdRange`.

Functions:
    threshold_steps: every percent the slider offers for a default threshold in (0, 1], ascending.
        The percent is rounded half up, like JavaScript's Math.round.
"""

import math

__all__ = [
    "threshold_steps",
]


def threshold_steps(default: float) -> list[int]:
    percent = math.floor(default * 100 + 0.5)
    low = max(1, min(50, percent - percent % 5))
    step = 5 if percent % 5 == 0 else 1
    return list(range(low, 101, step))
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/test_site_thresholds.py -q --durations=3`. Expected: PASS, every test < 50 ms.

- [ ] **Step 5: Commit**

```bash
git add src/jev_bench/site tests/test_site_thresholds.py tests/fixtures/threshold_steps.json tests/js/site-contract.test.mjs
git commit -m "feat(site): slider threshold steps, shared with the UI through a fixture"
```

---

### Task 3: Published views

**Files:**
- Create: `src/jev_bench/site/views.py`
- Create: `tests/fixtures/default_views.json`
- Modify: `tests/js/site-contract.test.mjs` (append)
- Test: `tests/test_site_views.py`

**Interfaces:**
- Produces:
  - `RunLike`, `ColumnLike`, `AnalysisLike`: read-only `Protocol`s. `RunMeta`, `ColumnConfig` and
    `AnalysisMeta` satisfy them.
  - `View` (pydantic): `id: str`, `label: str`, `generation_ids: tuple[str, ...]`, `run_ids: tuple[str, ...]`;
    `View.of(label, generation_ids, run_ids) -> View`.
  - `latest_per_column(runs, generation_ids, hidden=frozenset()) -> tuple[str, ...]`.
  - `hidden_sets(columns) -> list[frozenset[str]]`.
  - `derive_views(columns, generation_ids, runs, analyses) -> list[View]`.

- [ ] **Step 1: Write the shared cases**

`tests/fixtures/default_views.json`:

```json
{
  "catalog": [
    {"id": "jev", "slot": "jev"},
    {"id": "anthropic", "slot": "anthropic"},
    {"id": "embeddings", "slot": "embeddings"},
    {"id": "kev", "slot": "embeddings"}
  ],
  "hidden_sets": [["kev"], ["embeddings"]],
  "runs": [
    {"id": "20260925-080000-jev-a", "column": "jev", "status": "completed", "generation_ids": ["g1"]},
    {"id": "20260925-090000-jev-b", "column": "jev", "status": "completed", "generation_ids": ["g1"]},
    {"id": "20260925-100000-jev-c", "column": "jev", "status": "running", "generation_ids": ["g1"]},
    {"id": "20260925-090000-embeddings-a", "column": "embeddings", "status": "completed", "generation_ids": ["g1"]},
    {"id": "20260925-095000-kev-a", "column": "kev", "status": "completed", "generation_ids": ["g1"]},
    {"id": "20260925-110000-anthropic-a", "column": "anthropic", "status": "failed", "generation_ids": ["g1"]},
    {"id": "20260925-120000-anthropic-b", "column": "anthropic", "status": "completed", "generation_ids": ["g1", "g2"]}
  ],
  "latest": [
    {"id": "hide-kev", "generation_ids": ["g1"], "hidden": ["kev"], "expected": ["20260925-090000-embeddings-a", "20260925-090000-jev-b"]},
    {"id": "hide-embeddings", "generation_ids": ["g1"], "hidden": ["embeddings"], "expected": ["20260925-090000-jev-b", "20260925-095000-kev-a"]},
    {"id": "all-columns", "generation_ids": ["g1"], "hidden": [], "expected": ["20260925-090000-embeddings-a", "20260925-090000-jev-b", "20260925-095000-kev-a"]},
    {"id": "exactly-two-generations", "generation_ids": ["g2", "g1"], "hidden": [], "expected": ["20260925-120000-anthropic-b"]},
    {"id": "unknown-generation", "generation_ids": ["g9"], "hidden": [], "expected": []}
  ]
}
```

- [ ] **Step 2: Append the JS contract tests**

Append to `tests/js/site-contract.test.mjs`:

```js
const { latestCompletedPerColumn } = await import("../../src/jev_bench/web/static/js/selection.js");
const { hiddenColumnIds, slotGroups } = await import("../../src/jev_bench/web/static/js/slots.js");
const views = fixture("default_views.json");

for (const { id, generation_ids, hidden, expected } of views.latest) {
  test(`latestCompletedPerColumn matches the shared case ${id}`, () => {
    const shown = views.runs.filter((run) => !hidden.includes(run.column));
    assert.deepEqual(latestCompletedPerColumn(shown, generation_ids).sort(), expected);
  });
}

test("every combination of slot picks hides one of the shared hidden sets", () => {
  const combos = slotGroups(views.catalog).reduce((all, { slot, columns }) => all.flatMap((picks) => columns.map((column) => ({ ...picks, [slot]: column.id }))), [{}]);
  assert.deepEqual(combos.map((picks) => [...hiddenColumnIds(views.catalog, picks)].sort()), views.hidden_sets);
});
```

Run: `node --test tests/js/site-contract.test.mjs`. Expected: PASS (it pins the existing JS rules).

- [ ] **Step 3: Write the failing Python tests**

`tests/test_site_views.py`:

```python
"""Tests for jev_bench.site.views: the shared JS cases, then the views derived from committed data."""

import json
import re
from pathlib import Path
from typing import NamedTuple

import pytest

from jev_bench.site.views import View, derive_views, hidden_sets, latest_per_column

FIXTURE = json.loads(
    (Path(__file__).parent / "fixtures" / "default_views.json").read_text(encoding="utf-8")
)


class Run(NamedTuple):
    id: str
    column: str
    status: str
    generation_ids: tuple[str, ...]


class Column(NamedTuple):
    id: str
    effective_slot: str


class Analysis(NamedTuple):
    id: str
    status: str
    generation_ids: tuple[str, ...]
    run_ids: tuple[str, ...]


SHARED_RUNS = [
    Run(run["id"], run["column"], run["status"], tuple(run["generation_ids"]))
    for run in FIXTURE["runs"]
]
CATALOG = [Column(column["id"], column["slot"]) for column in FIXTURE["catalog"]]


@pytest.mark.parametrize(
    ("generation_ids", "hidden", "expected"),
    [
        pytest.param(case["generation_ids"], frozenset(case["hidden"]), case["expected"], id=case["id"])
        for case in FIXTURE["latest"]
    ],
)
def test_latest_per_column_matches_the_shared_cases(
    generation_ids: list[str], hidden: frozenset[str], expected: list[str]
) -> None:
    assert sorted(latest_per_column(SHARED_RUNS, generation_ids, hidden)) == expected


def test_hidden_sets_match_the_shared_cases() -> None:
    assert [sorted(hidden) for hidden in hidden_sets(CATALOG)] == FIXTURE["hidden_sets"]


JEV = Run("20260925-090000-jev-a", "jev", "completed", ("g1",))
OLD_JEV = Run("20260925-080000-jev-b", "jev", "completed", ("g1",))
CLAUDE = Run("20260925-090100-anthropic-a", "anthropic", "completed", ("g1",))
EMBEDDINGS = Run("20260925-090200-embeddings-a", "embeddings", "completed", ("g1",))
KEV = Run("20260925-090300-kev-a", "kev", "completed", ("g1",))
ORPHAN = Run("20260925-070000-jev-c", "jev", "completed", ("g9",))
FULL = [KEV, EMBEDDINGS, CLAUDE, JEV, OLD_JEV, ORPHAN]


def _analysis(run_ids: tuple[str, ...], generation_ids: tuple[str, ...] = ("g1",), status: str = "completed") -> Analysis:
    return Analysis("20260928-070000-analysis-a", status, generation_ids, run_ids)


type Expected = list[tuple[str, tuple[str, ...], tuple[str, ...]]]

SLOT_VIEWS: Expected = [
    ("Latest without kev", ("g1",), (JEV.id, CLAUDE.id, EMBEDDINGS.id)),
    ("Latest without embeddings", ("g1",), (JEV.id, CLAUDE.id, KEV.id)),
    ("All columns", ("g1",), (JEV.id, CLAUDE.id, EMBEDDINGS.id, KEV.id)),
    ("Generation only", ("g1",), ()),
]


@pytest.mark.parametrize(
    ("generation_ids", "runs", "analyses", "expected"),
    [
        pytest.param(["g1"], FULL, [], SLOT_VIEWS, id="slot-states-and-all-columns"),
        pytest.param(
            ["g1"], [JEV], [],
            [("Latest without kev", ("g1",), (JEV.id,)), ("Generation only", ("g1",), ())],
            id="equal-selections-merge",
        ),
        pytest.param(["g2"], FULL, [], [("Generation only", ("g2",), ())], id="generation-without-runs"),
        pytest.param(
            ["g1"], FULL, [_analysis((OLD_JEV.id, CLAUDE.id))],
            [*SLOT_VIEWS, (f"Analysis {_analysis(()).id}", ("g1",), (OLD_JEV.id, CLAUDE.id))],
            id="analysis-view",
        ),
        pytest.param(["g1"], FULL, [_analysis((JEV.id, CLAUDE.id, EMBEDDINGS.id))], SLOT_VIEWS, id="analysis-equal-to-latest"),
        pytest.param(["g1"], FULL, [_analysis(("20260925-000000-jev-gone",))], SLOT_VIEWS, id="analysis-missing-run"),
        pytest.param(["g1"], FULL, [_analysis((JEV.id,), ("g1", "g9"))], SLOT_VIEWS, id="analysis-missing-generation"),
        pytest.param(["g1"], FULL, [_analysis((OLD_JEV.id,), status="failed")], SLOT_VIEWS, id="analysis-not-completed"),
    ],
)
def test_derive_views(
    generation_ids: list[str], runs: list[Run], analyses: list[Analysis], expected: Expected
) -> None:
    views = derive_views(CATALOG, generation_ids, runs, analyses)
    assert [(view.label, view.generation_ids, view.run_ids) for view in views] == expected


def test_view_id_is_an_order_insensitive_content_hash() -> None:
    first = View.of("a", ["g1"], ["r2", "r1"])
    assert first.id == View.of("b", ["g1"], ["r1", "r2"]).id
    assert first.id != View.of("a", ["g2"], ["r1", "r2"]).id
    assert re.fullmatch(r"v[0-9a-f]{12}", first.id)
```

Run: `uv run pytest tests/test_site_views.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'jev_bench.site.views'`.

- [ ] **Step 4: Implement `views.py`**

```python
"""Published selections of the static snapshot: the (generation set, run set) pairs it precomputes.

A view mirrors a selection the UI makes by itself:
- the Benchmark's "Latest" for every card-slot state (`selection.defaultRunIds` over
  `slots.hiddenColumnIds`);
- the Explorer's default over all columns (`selection.latestCompletedPerColumn`);
- a generation alone;
- every completed analysis whose generations and runs are all committed.

`tests/fixtures/default_views.json` holds the cases the Python and JS rules must both pass. Runs are listed
in catalog column order, the order the Benchmark requests them in.

Classes:
    RunLike, ColumnLike, AnalysisLike: the fields read from runs, catalog columns and analyses.
    View: one published selection; its id is a hash of its sorted generation and run ids, so an id
        always names the same selection across deploys.
Functions:
    latest_per_column: the newest completed run id per column on exactly these generations.
    hidden_sets: the hidden column ids of every combination of card-slot picks.
    derive_views: every view of the committed data, the first of equal selections kept.
"""

import hashlib
from collections.abc import Iterable, Sequence
from itertools import product
from typing import Protocol

from loguru import logger
from pydantic import BaseModel

__all__ = [
    "AnalysisLike",
    "ColumnLike",
    "RunLike",
    "View",
    "derive_views",
    "hidden_sets",
    "latest_per_column",
]

ALL_COLUMNS = "All columns"
GENERATION_ONLY = "Generation only"


class RunLike(Protocol):
    @property
    def id(self) -> str: ...
    @property
    def column(self) -> str: ...
    @property
    def status(self) -> str: ...
    @property
    def generation_ids(self) -> tuple[str, ...]: ...


class ColumnLike(Protocol):
    @property
    def id(self) -> str: ...
    @property
    def effective_slot(self) -> str: ...


class AnalysisLike(Protocol):
    @property
    def id(self) -> str: ...
    @property
    def status(self) -> str: ...
    @property
    def generation_ids(self) -> tuple[str, ...]: ...
    @property
    def run_ids(self) -> tuple[str, ...]: ...


class View(BaseModel):
    id: str
    label: str
    generation_ids: tuple[str, ...]
    run_ids: tuple[str, ...]

    @classmethod
    def of(cls, label: str, generation_ids: Iterable[str], run_ids: Iterable[str]) -> "View":
        generations, runs = tuple(generation_ids), tuple(run_ids)
        return cls(id=_view_id(generations, runs), label=label, generation_ids=generations, run_ids=runs)


def _view_id(generation_ids: tuple[str, ...], run_ids: tuple[str, ...]) -> str:
    text = ",".join(sorted(generation_ids)) + "|" + ",".join(sorted(run_ids))
    return "v" + hashlib.sha256(text.encode()).hexdigest()[:12]


def latest_per_column(
    runs: Iterable[RunLike], generation_ids: Iterable[str], hidden: frozenset[str] = frozenset()
) -> tuple[str, ...]:
    wanted = set(generation_ids)
    chosen: dict[str, str] = {}
    for run in sorted(runs, key=lambda run: run.id, reverse=True):
        if run.status == "completed" and set(run.generation_ids) == wanted and run.column not in hidden:
            chosen.setdefault(run.column, run.id)
    return tuple(chosen.values())


def hidden_sets(columns: Sequence[ColumnLike]) -> list[frozenset[str]]:
    slots: dict[str, list[str]] = {}
    for column in columns:
        slots.setdefault(column.effective_slot, []).append(column.id)
    everything = frozenset(column.id for column in columns)
    return [everything - set(shown) for shown in product(*slots.values())]


def derive_views(
    columns: Sequence[ColumnLike],
    generation_ids: Sequence[str],
    runs: Sequence[RunLike],
    analyses: Iterable[AnalysisLike],
) -> list[View]:
    rank = _column_rank(columns, runs)
    per_generation = [view for gid in generation_ids for view in _generation_views(columns, gid, runs, rank)]
    known = {run.id for run in runs}
    return _unique([*per_generation, *_analysis_views(analyses, set(generation_ids), known)])


def _column_rank(columns: Sequence[ColumnLike], runs: Sequence[RunLike]) -> dict[str, int]:
    index = {column.id: position for position, column in enumerate(columns)}
    return {run.id: index.get(run.column, len(index)) for run in runs}


def _latest_label(hidden: frozenset[str]) -> str:
    return f"Latest without {', '.join(sorted(hidden))}" if hidden else "Latest"


def _generation_views(
    columns: Sequence[ColumnLike], generation_id: str, runs: Sequence[RunLike], rank: dict[str, int]
) -> list[View]:
    latest = [(_latest_label(hidden), latest_per_column(runs, [generation_id], hidden)) for hidden in hidden_sets(columns)]
    pairs = [*latest, (ALL_COLUMNS, latest_per_column(runs, [generation_id]))]
    views = [View.of(label, [generation_id], sorted(ids, key=rank.__getitem__)) for label, ids in pairs if ids]
    return [*views, View.of(GENERATION_ONLY, [generation_id], [])]


def _analysis_views(analyses: Iterable[AnalysisLike], generations: set[str], runs: set[str]) -> list[View]:
    return [
        View.of(f"Analysis {analysis.id}", analysis.generation_ids, analysis.run_ids)
        for analysis in analyses
        if _publishable(analysis, generations, runs)
    ]


def _publishable(analysis: AnalysisLike, generations: set[str], runs: set[str]) -> bool:
    if analysis.status != "completed":
        return False
    complete = set(analysis.generation_ids) <= generations and set(analysis.run_ids) <= runs
    if not complete:
        logger.warning("Analysis {} is not published: some of its generations or runs are not committed", analysis.id)
    return complete


def _unique(views: Iterable[View]) -> list[View]:
    kept: dict[str, View] = {}
    for view in views:
        kept.setdefault(view.id, view)
    return list(kept.values())
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest tests/test_site_views.py -q --durations=3 && uv run pyright src/jev_bench/site`
Expected: PASS, and pyright reports 0 errors. Task 9 passes real `RunMeta`, `ColumnConfig` and
`AnalysisMeta` objects where these Protocols are expected; the read-only properties accept their fields
(a `Literal` status is a `str`). If pyright disagrees, fix the Protocols, never the store models.

- [ ] **Step 6: Commit**

```bash
git add src/jev_bench/site/views.py tests/test_site_views.py tests/fixtures/default_views.json tests/js/site-contract.test.mjs
git commit -m "feat(site): derive the published views from committed runs, slots and analyses"
```

---

### Task 4: Request → file contract (Python and JS halves)

**Files:**
- Create: `src/jev_bench/site/paths.py`
- Create: `src/jev_bench/web/static/js/static-api.js`
- Create: `tests/fixtures/site_paths.json`
- Create: `tests/js/static-api.test.mjs`
- Modify: `tests/test_web_static.py` (`PAGE_MODULES` += `"static-api.js"`)
- Test: `tests/test_site_paths.py`

**Interfaces:**
- Produces (Python): `type ThresholdEndpoint = Literal["compare", "emails"]`;
  `base_file(path: str) -> str`; `threshold_file(endpoint, view_id: str, percent: int | None) -> str`;
  `email_file(view_id: str, email_id: str) -> str`.
- Produces (JS): `class NotPublished extends Error`; `staticFile(method, path, manifest) -> string`
  (throws `NotPublished`). `manifest` is `{ views: [{ id, generation_ids, run_ids }] }`.

- [ ] **Step 1: Write the shared cases**

`tests/fixtures/site_paths.json` (requests are exactly what `api.js` builds; `URLSearchParams` encodes `,`
as `%2C`):

```json
{
  "manifest": {
    "views": [
      {"id": "v1", "generation_ids": ["g1"], "run_ids": ["r1", "r2"]},
      {"id": "v2", "generation_ids": ["g1"], "run_ids": []},
      {"id": "v3", "generation_ids": ["g2"], "run_ids": []}
    ]
  },
  "published": [
    {"id": "status", "request": "/status", "base": "/status", "file": "api/status.json"},
    {"id": "runs-list", "request": "/runs", "base": "/runs", "file": "api/runs.json"},
    {"id": "run-detail", "request": "/runs/r1", "base": "/runs/r1", "file": "api/runs/r1.json"},
    {"id": "analysis-prompts", "request": "/analyses/a1/prompts", "base": "/analyses/a1/prompts", "file": "api/analyses/a1/prompts.json"},
    {"id": "analysis-defaults", "request": "/analysis/defaults", "base": "/analysis/defaults", "file": "api/analysis/defaults.json"},
    {"id": "compare-default", "request": "/compare?runs=r1%2Cr2", "endpoint": "compare", "view": "v1", "percent": null, "file": "api/compare/v1/t-default.json"},
    {"id": "compare-any-order", "request": "/compare?runs=r2%2Cr1&threshold=0.85", "endpoint": "compare", "view": "v1", "percent": 85, "file": "api/compare/v1/t85.json"},
    {"id": "compare-float-noise", "request": "/compare?runs=r1%2Cr2&threshold=0.29", "endpoint": "compare", "view": "v1", "percent": 29, "file": "api/compare/v1/t29.json"},
    {"id": "emails-with-runs", "request": "/emails?generations=g1&runs=r1%2Cr2&threshold=0.5", "endpoint": "emails", "view": "v1", "percent": 50, "file": "api/emails/v1/t50.json"},
    {"id": "emails-generation-only", "request": "/emails?generations=g1", "endpoint": "emails", "view": "v2", "percent": null, "file": "api/emails/v2/t-default.json"},
    {"id": "email-detail", "request": "/emails/g1.0001?runs=r1%2Cr2", "email": "g1.0001", "view": "v1", "file": "api/email/v1/g1.0001.json"},
    {"id": "email-detail-by-generation", "request": "/emails/g2.0003", "email": "g2.0003", "view": "v3", "file": "api/email/v3/g2.0003.json"}
  ],
  "unpublished": [
    {"id": "run-subset", "method": "GET", "request": "/compare?runs=r1"},
    {"id": "run-superset", "method": "GET", "request": "/compare?runs=r1%2Cr2%2Cr3"},
    {"id": "two-generations", "method": "GET", "request": "/emails?generations=g1%2Cg2"},
    {"id": "filtered-run-list", "method": "GET", "request": "/runs?generations=g1"},
    {"id": "email-of-unpublished-generation", "method": "GET", "request": "/emails/g9.0001"},
    {"id": "start-run", "method": "POST", "request": "/runs"},
    {"id": "estimate", "method": "POST", "request": "/runs/estimate"},
    {"id": "label", "method": "PUT", "request": "/labels/g1.0001"}
  ]
}
```

- [ ] **Step 2: Write the failing tests**

`tests/test_site_paths.py`:

```python
"""Tests for jev_bench.site.paths against the shared cases in tests/fixtures/site_paths.json."""

import json
from pathlib import Path
from typing import Any

import pytest

from jev_bench.site.paths import base_file, email_file, threshold_file

FIXTURE = json.loads(
    (Path(__file__).parent / "fixtures" / "site_paths.json").read_text(encoding="utf-8")
)


def _python_file(case: dict[str, Any]) -> str:
    if "base" in case:
        return base_file(case["base"])
    if "email" in case:
        return email_file(case["view"], case["email"])
    return threshold_file(case["endpoint"], case["view"], case["percent"])


@pytest.mark.parametrize("case", [pytest.param(case, id=case["id"]) for case in FIXTURE["published"]])
def test_files_match_the_shared_cases(case: dict[str, Any]) -> None:
    assert _python_file(case) == case["file"]
```

`tests/js/static-api.test.mjs`:

```js
// Run with `node --test tests/js/`. Exercises src/jev_bench/web/static/js/static-api.js (pure) against the
// shared cases in tests/fixtures/site_paths.json.
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

const { NotPublished, staticFile } = await import("../../src/jev_bench/web/static/js/static-api.js");
const paths = JSON.parse(readFileSync(new URL("../fixtures/site_paths.json", import.meta.url), "utf-8"));

for (const { id, request, file } of paths.published) {
  test(`staticFile maps ${id}`, () => assert.equal(staticFile("GET", request, paths.manifest), file));
}

for (const { id, method, request } of paths.unpublished) {
  test(`staticFile refuses ${id}`, () => assert.throws(() => staticFile(method, request, paths.manifest), NotPublished));
}
```

Run: `uv run pytest tests/test_site_paths.py -q` and `node --test tests/js/static-api.test.mjs`
Expected: both FAIL, because neither module exists.

- [ ] **Step 3: Implement both halves**

`src/jev_bench/site/paths.py`:

```python
"""File layout of the static snapshot: the file that answers each GET of the JSON API.

The JS half of this contract is `js/static-api.js`; both are checked against the shared cases in
`tests/fixtures/site_paths.json`.

Types:
    ThresholdEndpoint: the view-dependent endpoints that take a label threshold.
Functions:
    base_file: `api<path>.json`, for a request that does not depend on a view.
    threshold_file: the compare or email-list file of a view at a threshold percent (None: default).
    email_file: the email-detail file of a view.
"""

from typing import Literal

__all__ = [
    "ThresholdEndpoint",
    "base_file",
    "email_file",
    "threshold_file",
]

type ThresholdEndpoint = Literal["compare", "emails"]


def base_file(path: str) -> str:
    return f"api{path}.json"


def threshold_file(endpoint: ThresholdEndpoint, view_id: str, percent: int | None) -> str:
    name = "t-default" if percent is None else f"t{percent}"
    return f"api/{endpoint}/{view_id}/{name}.json"


def email_file(view_id: str, email_id: str) -> str:
    return f"api/email/{view_id}/{email_id}.json"
```

`src/jev_bench/web/static/js/static-api.js`:

```js
/**
 * Pure mapping of a JSON API request to its file in the static snapshot: the JS half of the contract whose
 * Python half is jev_bench.site.paths (shared cases: tests/fixtures/site_paths.json). A view-dependent
 * request is matched to the manifest view with the same run set (and the same generation set, or the
 * email's generation for a detail), in any order. Anything else with a query, and any non-GET, throws.
 * Exports: NotPublished, staticFile.
 */
import { sameSet } from "./selection.js";

export class NotPublished extends Error {}

const ids = (params, name) => (params.get(name) ?? "").split(",").filter(Boolean);
const percentFile = (params) => (params.has("threshold") ? `t${Math.round(Number(params.get("threshold")) * 100)}` : "t-default");

function findView(manifest, runIds, matchesGenerations) {
  const view = manifest.views.find((item) => sameSet(runIds, new Set(item.run_ids)) && matchesGenerations(item.generation_ids));
  if (!view) throw new NotPublished("selection");
  return view.id;
}

function thresholdFile(endpoint, manifest, params, generationIds = null) {
  const matches = (candidate) => generationIds === null || sameSet(generationIds, new Set(candidate));
  return `api/${endpoint}/${findView(manifest, ids(params, "runs"), matches)}/${percentFile(params)}.json`;
}

function emailFile(emailId, manifest, params) {
  const generation = emailId.slice(0, emailId.lastIndexOf("."));
  return `api/email/${findView(manifest, ids(params, "runs"), (candidate) => candidate.includes(generation))}/${emailId}.json`;
}

export function staticFile(method, path, manifest) {
  if (method !== "GET") throw new NotPublished(`${method} ${path}`);
  const url = new URL(path, "http://snapshot.invalid");
  if (url.pathname === "/compare") return thresholdFile("compare", manifest, url.searchParams);
  if (url.pathname === "/emails") return thresholdFile("emails", manifest, url.searchParams, ids(url.searchParams, "generations"));
  if (url.pathname.startsWith("/emails/")) return emailFile(url.pathname.slice("/emails/".length), manifest, url.searchParams);
  if (url.search) throw new NotPublished(path);
  return `api${url.pathname}.json`;
}
```

In `tests/test_web_static.py`, add `"static-api.js",` to `PAGE_MODULES`.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_site_paths.py tests/test_web_static.py -q && node --check src/jev_bench/web/static/js/static-api.js && node --test tests/js/`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/jev_bench/site/paths.py src/jev_bench/web/static/js/static-api.js tests/test_site_paths.py tests/js/static-api.test.mjs tests/fixtures/site_paths.json tests/test_web_static.py
git commit -m "feat(site): request-to-file contract of the static snapshot, in Python and JS"
```

---

### Task 5: Request plan

**Files:**
- Create: `src/jev_bench/site/plan.py`
- Test: `tests/test_site_plan.py`

**Interfaces:**
- Consumes: `View` (Task 3); `base_file`, `threshold_file`, `email_file`, `ThresholdEndpoint` (Task 4).
- Produces:
  - `SiteRequest(path: str, params: dict[str, str] = {}, file: str)` (pydantic);
  - `base_requests(generation_ids, run_ids, analysis_ids) -> list[SiteRequest]`;
  - `threshold_requests(endpoint, view, percents: Sequence[int | None]) -> list[SiteRequest]`;
  - `email_requests(view, email_ids) -> list[SiteRequest]`;
  - `multi_threshold(endpoint, body: Mapping[str, Any]) -> float | None`.

- [ ] **Step 1: Write the failing tests**

`tests/test_site_plan.py`:

```python
"""Tests for jev_bench.site.plan."""

from typing import Any

import pytest

from jev_bench.site.paths import ThresholdEndpoint
from jev_bench.site.plan import (
    base_requests,
    email_requests,
    multi_threshold,
    threshold_requests,
)
from jev_bench.site.views import View

RUNS = View.of("Latest", ["g1"], ["r1", "r2"])
ALONE = View.of("Generation only", ["g1"], [])


def test_base_requests_cover_lists_and_every_detail() -> None:
    requests = base_requests(["g1"], ["r1"], ["a1"])
    assert [(request.path, request.params, request.file) for request in requests] == [
        ("/status", {}, "api/status.json"),
        ("/catalog", {}, "api/catalog.json"),
        ("/questions", {}, "api/questions.json"),
        ("/generations", {}, "api/generations.json"),
        ("/runs", {}, "api/runs.json"),
        ("/analyses", {}, "api/analyses.json"),
        ("/analysis/defaults", {}, "api/analysis/defaults.json"),
        ("/generations/g1", {}, "api/generations/g1.json"),
        ("/runs/r1", {}, "api/runs/r1.json"),
        ("/analyses/a1", {}, "api/analyses/a1.json"),
        ("/analyses/a1/prompts", {}, "api/analyses/a1/prompts.json"),
    ]


@pytest.mark.parametrize(
    ("endpoint", "view", "percent", "params", "file"),
    [
        pytest.param("compare", RUNS, None, {"runs": "r1,r2"}, f"api/compare/{RUNS.id}/t-default.json", id="compare-default"),
        pytest.param("compare", RUNS, 55, {"runs": "r1,r2", "threshold": "0.55"}, f"api/compare/{RUNS.id}/t55.json", id="compare-step"),
        pytest.param("emails", RUNS, 100, {"generations": "g1", "runs": "r1,r2", "threshold": "1.0"}, f"api/emails/{RUNS.id}/t100.json", id="emails-step"),
        pytest.param("emails", ALONE, None, {"generations": "g1"}, f"api/emails/{ALONE.id}/t-default.json", id="emails-without-runs"),
    ],
)
def test_threshold_requests(
    endpoint: ThresholdEndpoint, view: View, percent: int | None, params: dict[str, str], file: str
) -> None:
    (request,) = threshold_requests(endpoint, view, [percent])
    assert (request.path, request.params, request.file) == (f"/{endpoint}", params, file)


@pytest.mark.parametrize(
    ("view", "params"),
    [
        pytest.param(RUNS, {"runs": "r1,r2"}, id="with-runs"),
        pytest.param(ALONE, {}, id="without-runs"),
    ],
)
def test_email_requests(view: View, params: dict[str, str]) -> None:
    (request,) = email_requests(view, ["g1.0001"])
    assert (request.path, request.params, request.file) == ("/emails/g1.0001", params, f"api/email/{view.id}/g1.0001.json")


MULTI = {"type": "multi", "threshold": 0.75}
CHOICE = {"type": "choice"}


@pytest.mark.parametrize(
    ("endpoint", "body", "expected"),
    [
        pytest.param("compare", {"questions": [CHOICE, MULTI]}, 0.75, id="compare-report"),
        pytest.param("emails", {"questions": {"questions": [CHOICE, MULTI]}}, 0.75, id="email-list"),
        pytest.param("compare", {"questions": [CHOICE]}, None, id="no-multi-question"),
    ],
)
def test_multi_threshold(endpoint: ThresholdEndpoint, body: dict[str, Any], expected: float | None) -> None:
    assert multi_threshold(endpoint, body) == expected
```

Run: `uv run pytest tests/test_site_plan.py -q`. Expected: FAIL (`No module named 'jev_bench.site.plan'`).

- [ ] **Step 2: Implement `plan.py`**

```python
"""Request plan of the snapshot export: the GETs the static site needs, each with the file answering it.

Classes:
    SiteRequest: one GET of the JSON API (path under /api, query) and its file in the snapshot.
Functions:
    base_requests: status, catalog, questions, the lists, and every generation, run and analysis
        detail (analyses with their sent prompts).
    threshold_requests: compare or the email list of a view at the given threshold percents (None:
        without ?threshold=, i.e. each question's own default).
    email_requests: one email detail per email id, with the view's runs.
    multi_threshold: the threshold of the first multi-label question in a compare report or an email
        list (None without one); the Benchmark and Explorer sliders start from it.
"""

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from pydantic import BaseModel, Field

from jev_bench.site.paths import ThresholdEndpoint, base_file, email_file, threshold_file
from jev_bench.site.views import View

__all__ = [
    "LISTS",
    "SiteRequest",
    "base_requests",
    "email_requests",
    "multi_threshold",
    "threshold_requests",
]

LISTS = ("/status", "/catalog", "/questions", "/generations", "/runs", "/analyses", "/analysis/defaults")


class SiteRequest(BaseModel):
    path: str
    params: dict[str, str] = Field(default_factory=dict)
    file: str


def base_requests(
    generation_ids: Iterable[str], run_ids: Iterable[str], analysis_ids: Sequence[str]
) -> list[SiteRequest]:
    details = [
        *(f"/generations/{generation_id}" for generation_id in generation_ids),
        *(f"/runs/{run_id}" for run_id in run_ids),
        *(f"/analyses/{analysis_id}" for analysis_id in analysis_ids),
        *(f"/analyses/{analysis_id}/prompts" for analysis_id in analysis_ids),
    ]
    return [SiteRequest(path=path, file=base_file(path)) for path in [*LISTS, *details]]


def _runs_param(view: View) -> dict[str, str]:
    return {"runs": ",".join(view.run_ids)} if view.run_ids else {}


def _view_params(endpoint: ThresholdEndpoint, view: View) -> dict[str, str]:
    generations = {"generations": ",".join(view.generation_ids)} if endpoint == "emails" else {}
    return {**generations, **_runs_param(view)}


def threshold_requests(
    endpoint: ThresholdEndpoint, view: View, percents: Sequence[int | None]
) -> list[SiteRequest]:
    base = _view_params(endpoint, view)
    return [
        SiteRequest(
            path=f"/{endpoint}",
            params=base if percent is None else {**base, "threshold": str(percent / 100)},
            file=threshold_file(endpoint, view.id, percent),
        )
        for percent in percents
    ]


def email_requests(view: View, email_ids: Iterable[str]) -> list[SiteRequest]:
    params = _runs_param(view)
    return [
        SiteRequest(path=f"/emails/{email_id}", params=params, file=email_file(view.id, email_id))
        for email_id in email_ids
    ]


def multi_threshold(endpoint: ThresholdEndpoint, body: Mapping[str, Any]) -> float | None:
    questions = body["questions"] if endpoint == "compare" else body["questions"]["questions"]
    return next((question["threshold"] for question in questions if question["type"] == "multi"), None)
```

- [ ] **Step 3: Run the tests to verify they pass**

Run: `uv run pytest tests/test_site_plan.py -q --durations=3`. Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add src/jev_bench/site/plan.py tests/test_site_plan.py
git commit -m "feat(site): plan every GET of the static snapshot with its file"
```

---

### Task 6: Offline services

**Files:**
- Create: `src/jev_bench/site/errors.py`
- Create: `src/jev_bench/site/offline.py`
- Test: `tests/test_site_offline.py`

**Interfaces:**
- Produces:
  - `ExportError(Exception)`;
  - `OfflineCatalog(Catalog)` whose `for_column` returns `[]`;
  - `refusing_client() -> httpx2.AsyncClient` (any request raises `ExportError`);
  - `offline_settings(settings) -> Settings` (no key, `max_retries=0`);
  - `offline_services(settings, http) -> Services`.

- [ ] **Step 1: Write the failing tests**

`tests/test_site_offline.py`:

```python
"""Tests for jev_bench.site.offline: no network, no key, no OpenRouter model list."""

from pathlib import Path

import pytest
from tests.factories import mini_settings

from jev_bench.site.errors import ExportError
from jev_bench.site.offline import OfflineCatalog, offline_services, refusing_client


async def test_offline_services_hold_no_key_and_list_no_models(tmp_path: Path) -> None:
    async with refusing_client() as http:
        services = offline_services(mini_settings(tmp_path, api_key="sk-or-v1-SENTINEL"), http)
        column = services.benchmark_config().columns[0]
        assert isinstance(services.catalog, OfflineCatalog)
        assert await services.catalog.for_column(column) == []
    assert services.settings.server_api_key() is None
    assert services.settings.max_retries == 0


async def test_refusing_client_raises_on_any_request() -> None:
    async with refusing_client() as http:
        with pytest.raises(ExportError, match="must not reach the network"):
            await http.get("https://openrouter.ai/api/v1/models")
```

Run: `uv run pytest tests/test_site_offline.py -q`. Expected: FAIL (modules missing).

- [ ] **Step 2: Implement**

`src/jev_bench/site/errors.py`:

```python
"""Errors of the static snapshot export.

Classes:
    ExportError: the export cannot produce a correct snapshot; the message says why.
"""

__all__ = [
    "ExportError",
]


class ExportError(Exception):
    pass
```

`src/jev_bench/site/offline.py`:

```python
"""Offline services for the snapshot export: nothing reaches the network and no key is held.

Classes:
    OfflineCatalog: a Catalog listing no OpenRouter models; the catalog route then offers each
        column's default model only.
Functions:
    refusing_client: an httpx2 client whose transport raises ExportError on any request.
    offline_settings: the given Settings without an OpenRouter key and without retries.
    offline_services: Services over offline_settings and the given client, with OfflineCatalog.
"""

import httpx2

from jev_bench.benchmark_config import ColumnConfig
from jev_bench.catalog import Catalog, ModelInfo
from jev_bench.services import Services
from jev_bench.settings import Settings
from jev_bench.site.errors import ExportError

__all__ = [
    "OfflineCatalog",
    "offline_services",
    "offline_settings",
    "refusing_client",
]


class OfflineCatalog(Catalog):
    async def for_column(self, column: ColumnConfig) -> list[ModelInfo]:
        return []


def _refuse(request: httpx2.Request) -> httpx2.Response:
    raise ExportError(f"the export must not reach the network: {request.url}")


def refusing_client() -> httpx2.AsyncClient:
    return httpx2.AsyncClient(transport=httpx2.MockTransport(_refuse))


def offline_settings(settings: Settings) -> Settings:
    return settings.model_copy(update={"openrouter_api_key": None, "max_retries": 0})


def offline_services(settings: Settings, http: httpx2.AsyncClient) -> Services:
    services = Services(offline_settings(settings), http)
    services.catalog = OfflineCatalog(services.client)
    return services
```

- [ ] **Step 3: Run the tests to verify they pass**

Run: `uv run pytest tests/test_site_offline.py -q --durations=3 && uv run pyright src/jev_bench/site`
Expected: PASS; 0 errors.

- [ ] **Step 4: Commit**

```bash
git add src/jev_bench/site/errors.py src/jev_bench/site/offline.py tests/test_site_offline.py
git commit -m "feat(site): offline services for the export (no network, no key, no model list)"
```

---

### Task 7: Static copy of the UI

**Files:**
- Create: `src/jev_bench/site/static_copy.py`
- Test: `tests/test_site_static_copy.py`

**Interfaces:**
- Consumes: `ExportError` (Task 6); `jev_bench.web.security.CSP`.
- Produces: `STATIC_MODULE: str`; `meta_policy(policy: str = CSP) -> str`; `head_tags() -> str`;
  `copy_static(source: Path, out: Path) -> None`.

- [ ] **Step 1: Write the failing tests**

`tests/test_site_static_copy.py`:

```python
"""Tests for jev_bench.site.static_copy."""

from pathlib import Path

import pytest

from jev_bench.site.errors import ExportError
from jev_bench.site.static_copy import STATIC_MODULE, copy_static
from jev_bench.web.app import STATIC_DIR


def _source(root: Path, head: str = "<head>") -> Path:
    source = root / "ui"
    (source / "js").mkdir(parents=True)
    (source / "index.html").write_text(f"<!doctype html><html>{head}<title>x</title></head></html>", encoding="utf-8")
    (source / "js" / "deployment.js").write_text("export const STATIC = false;\n", encoding="utf-8")
    return source


def test_copy_static_switches_to_static_mode_and_adds_the_policy(tmp_path: Path) -> None:
    out = tmp_path / "out"
    copy_static(_source(tmp_path), out)
    assert (out / "js" / "deployment.js").read_text(encoding="utf-8") == STATIC_MODULE
    html = (out / "index.html").read_text(encoding="utf-8")
    assert html.count('http-equiv="Content-Security-Policy"') == 1
    assert '<meta name="referrer" content="no-referrer">' in html
    assert "default-src 'self'" in html
    assert "frame-ancestors" not in html


@pytest.mark.parametrize(
    "head",
    [
        pytest.param("", id="no-head"),
        pytest.param("<head></head><head>", id="two-heads"),
    ],
)
def test_copy_static_refuses_a_page_without_exactly_one_head(tmp_path: Path, head: str) -> None:
    with pytest.raises(ExportError, match="exactly one <head>"):
        copy_static(_source(tmp_path, head), tmp_path / "out")


@pytest.mark.parametrize("page", sorted(STATIC_DIR.glob("*.html")), ids=lambda path: path.name)
def test_every_real_page_has_exactly_one_head(page: Path) -> None:
    assert page.read_text(encoding="utf-8").count("<head>") == 1
```

Run: `uv run pytest tests/test_site_static_copy.py -q`. Expected: FAIL (module missing).

- [ ] **Step 2: Implement**

```python
"""Static half of the snapshot: the UI files, switched to static mode, with the CSP as meta tags.

GitHub Pages cannot send response headers, so the policy of `web.security.CSP` (still the one source)
goes into a <meta> tag, minus `frame-ancestors`, which browsers ignore there.

Constants:
    STATIC_MODULE: the `js/deployment.js` written into the snapshot.
Functions:
    meta_policy: the CSP for a <meta> tag.
    head_tags: the CSP and referrer meta tags inserted right after <head>.
    copy_static: copy the UI into `out`, write STATIC_MODULE, insert head_tags into every HTML page.
"""

import shutil
from pathlib import Path

from jev_bench.site.errors import ExportError
from jev_bench.web.security import CSP

__all__ = [
    "STATIC_MODULE",
    "copy_static",
    "head_tags",
    "meta_policy",
]

STATIC_MODULE = (
    "/** Static snapshot flag, written by `jev-bench export-site`. Exports: STATIC. */\n"
    "export const STATIC = true;\n"
)
HEAD = "<head>"


def meta_policy(policy: str = CSP) -> str:
    return "; ".join(part for part in policy.split("; ") if not part.startswith("frame-ancestors"))


def head_tags() -> str:
    return (
        f'\n  <meta http-equiv="Content-Security-Policy" content="{meta_policy()}">'
        '\n  <meta name="referrer" content="no-referrer">'
    )


def copy_static(source: Path, out: Path) -> None:
    shutil.copytree(source, out, dirs_exist_ok=True)
    (out / "js" / "deployment.js").write_text(STATIC_MODULE, encoding="utf-8")
    for page in sorted(out.glob("*.html")):
        _insert_head_tags(page)


def _insert_head_tags(page: Path) -> None:
    html = page.read_text(encoding="utf-8")
    if html.count(HEAD) != 1:
        raise ExportError(f"{page.name}: expected exactly one {HEAD}")
    page.write_text(html.replace(HEAD, HEAD + head_tags(), 1), encoding="utf-8")
```

- [ ] **Step 3: Run the tests to verify they pass**

Run: `uv run pytest tests/test_site_static_copy.py -q --durations=3`. Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add src/jev_bench/site/static_copy.py tests/test_site_static_copy.py
git commit -m "feat(site): copy the UI in static mode with the CSP as meta tags"
```

---

### Task 8: Snapshot writer

**Files:**
- Create: `src/jev_bench/site/writer.py`
- Test: `tests/test_site_writer.py`

**Interfaces:**
- Consumes: `SiteRequest` (Task 5), `ExportError` (Task 6).
- Produces: `SiteWriter(client: httpx2.AsyncClient, out: Path, *, timeout_s: float = 60.0)` with
  `async write(request: SiteRequest) -> httpx2.Response` and `put(file: str, content: bytes) -> None`.

- [ ] **Step 1: Write the failing tests**

`tests/test_site_writer.py`:

```python
"""Tests for jev_bench.site.writer."""

import asyncio
from pathlib import Path

import httpx2
import pytest

from jev_bench.site.errors import ExportError
from jev_bench.site.plan import SiteRequest
from jev_bench.site.writer import SiteWriter

BODY = b'{"ok": true}'


def _client(seen: list[str], status: int = 200, delay_s: float = 0.0) -> httpx2.AsyncClient:
    async def handler(request: httpx2.Request) -> httpx2.Response:
        seen.append(str(request.url))
        await asyncio.sleep(delay_s)
        return httpx2.Response(status, content=BODY)

    return httpx2.AsyncClient(transport=httpx2.MockTransport(handler), base_url="http://snapshot.invalid")


async def test_write_fetches_under_api_and_stores_the_body(tmp_path: Path) -> None:
    seen: list[str] = []
    request = SiteRequest(path="/compare", params={"runs": "r1,r2"}, file="api/compare/v1/t-default.json")
    async with _client(seen) as client:
        response = await SiteWriter(client, tmp_path).write(request)
    assert response.json() == {"ok": True}
    assert seen == ["http://snapshot.invalid/api/compare?runs=r1%2Cr2"]
    assert (tmp_path / "api" / "compare" / "v1" / "t-default.json").read_bytes() == BODY


@pytest.mark.parametrize(
    ("status", "delay_s", "file", "message"),
    [
        pytest.param(404, 0.0, "api/x.json", "answered 404", id="not-200"),
        pytest.param(200, 1.0, "api/x.json", "timed out", id="timeout"),
        pytest.param(200, 0.0, "../escape.json", "outside the snapshot", id="path-escape"),
    ],
)
async def test_write_refuses(tmp_path: Path, status: int, delay_s: float, file: str, message: str) -> None:
    async with _client([], status, delay_s) as client:
        writer = SiteWriter(client, tmp_path / "out", timeout_s=0.01)
        with pytest.raises(ExportError, match=message):
            await writer.write(SiteRequest(path="/x", file=file))
    assert not (tmp_path / "escape.json").exists()
```

Run: `uv run pytest tests/test_site_writer.py -q`. Expected: FAIL (module missing).

- [ ] **Step 2: Implement**

```python
"""Writer of the static snapshot: fetch one planned GET from the in-process app and store its body.

Classes:
    SiteWriter: `write` fetches a SiteRequest under /api within a timeout and stores the response
        body unchanged under the request's file (any status but 200 is an ExportError); `put` stores
        bytes, refusing any path outside the snapshot.
"""

import asyncio
from pathlib import Path

import httpx2

from jev_bench.site.errors import ExportError
from jev_bench.site.plan import SiteRequest

__all__ = [
    "SiteWriter",
]


class SiteWriter:
    def __init__(self, client: httpx2.AsyncClient, out: Path, *, timeout_s: float = 60.0) -> None:
        self._client = client
        self._out = out.resolve()
        self._timeout_s = timeout_s

    async def write(self, request: SiteRequest) -> httpx2.Response:
        response = await self._get(request)
        if response.status_code != 200:
            raise ExportError(f"GET /api{request.path} {request.params} answered {response.status_code}")
        self.put(request.file, response.content)
        return response

    async def _get(self, request: SiteRequest) -> httpx2.Response:
        try:
            async with asyncio.timeout(self._timeout_s):
                return await self._client.get(f"/api{request.path}", params=request.params)
        except TimeoutError as exc:
            raise ExportError(f"GET /api{request.path} timed out after {self._timeout_s} s") from exc

    def put(self, file: str, content: bytes) -> None:
        target = (self._out / file).resolve()
        if not target.is_relative_to(self._out):
            raise ExportError(f"refusing to write outside the snapshot: {file}")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
```

- [ ] **Step 3: Run the tests to verify they pass**

Run: `uv run pytest tests/test_site_writer.py -q --durations=3`. Expected: PASS, the timeout row ~10 ms.

- [ ] **Step 4: Commit**

```bash
git add src/jev_bench/site/writer.py tests/test_site_writer.py
git commit -m "feat(site): snapshot writer with timeout, status and path checks"
```

---

### Task 9: Export orchestration

**Files:**
- Create: `src/jev_bench/site/export.py`
- Test: `tests/test_site_export.py`

**Interfaces:**
- Consumes: Tasks 2–8.
- Produces:
  - `ExportSummary(files: int, bytes: int, seconds: float)`;
  - `SiteManifest(built_at: AwareDatetime, commit: str | None, views: list[View])`;
  - `async export_site(settings: Settings, out: Path, *, commit: str | None = None, static_dir: Path = STATIC_DIR) -> ExportSummary`,
    which raises `ExportError`.

- [ ] **Step 1: Write the failing tests**

`tests/test_site_export.py`:

```python
"""Tests for jev_bench.site.export on a mini data dir (two emails, a Jev and a Kev run)."""

import json
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import httpx2
import pytest
from tests.factories import ServicesFactory, seed_generation

from jev_bench.benchmark_config import JevParams
from jev_bench.emails import Email
from jev_bench.questions import Distribution, QuestionSet, hard_distribution
from jev_bench.services import Services
from jev_bench.site import export as export_module
from jev_bench.site.errors import ExportError
from jev_bench.site.export import export_site
from jev_bench.site.static_copy import STATIC_MODULE
from jev_bench.site.views import View
from jev_bench.store.runs import Prediction, RunMeta
from jev_bench.store.status import JobStatus

GENERATION = "20260924-100000-seed-abcd"
JEV = "20260924-110000-jev-0001"
KEV = "20260924-110500-kev-0002"
SENTINEL = "sk-or-v1-SENTINEL-export"


def _refuse(request: httpx2.Request) -> httpx2.Response:
    raise AssertionError(f"unexpected request {request.url}")


def _answers(questions: QuestionSet, email: Email) -> dict[str, Distribution]:
    pairs = ((q.id, hard_distribution(q, email.reference_answers.get(q.id))) for q in questions.questions)
    return {question_id: dist for question_id, dist in pairs if dist is not None}


def _seed_run(services: Services, run_id: str, column: str, status: JobStatus = "completed") -> None:
    questions = services.question_set()
    emails = services.generations.emails(GENERATION)
    meta = RunMeta(
        id=run_id, column=column, kind="decisions", model="m", generation_ids=(GENERATION,),
        mode="per_email", emails_per_request=1, question_set=questions, params=JevParams(),
        concurrency=1, created_at=datetime(2026, 9, 24, tzinfo=UTC), n_emails=len(emails),
        n_done=len(emails), status=status,
    )
    services.runs.save(meta)
    services.runs.append_predictions(run_id, [Prediction(email_id=e.id, answers=_answers(questions, e)) for e in emails])


def _seeded(services: Services) -> Services:
    seed_generation(services, emails=1, generation_id=GENERATION)
    _seed_run(services, JEV, "jev")
    _seed_run(services, KEV, "kev")
    return services


def _fake_ui(root: Path) -> Path:
    ui = root / "ui"
    (ui / "js").mkdir(parents=True)
    (ui / "index.html").write_text("<!doctype html><html><head><title>x</title></head></html>", encoding="utf-8")
    (ui / "js" / "deployment.js").write_text("export const STATIC = false;\n", encoding="utf-8")
    return ui


def _view_files(view: View, endpoints: tuple[str, ...]) -> set[str]:
    thresholds = {f"api/{endpoint}/{view.id}/{name}.json" for endpoint in endpoints for name in ("t-default", "t100")}
    return {*thresholds, f"api/email/{view.id}/{GENERATION}.0001.json"}


def _expected_files() -> set[str]:
    lists = {f"api/{name}.json" for name in ("status", "catalog", "questions", "generations", "runs", "analyses", "analysis/defaults")}
    details = {f"api/generations/{GENERATION}.json", f"api/runs/{JEV}.json", f"api/runs/{KEV}.json"}
    views = [View.of("", [GENERATION], [JEV]), View.of("", [GENERATION], [JEV, KEV])]
    per_view = set().union(*(_view_files(view, ("compare", "emails")) for view in views))
    alone = _view_files(View.of("", [GENERATION], []), ("emails",))
    return {"index.html", "js/deployment.js", "api/site.json", *lists, *details, *per_view, *alone}


@pytest.fixture(autouse=True)
def _one_threshold_step(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(export_module, "threshold_steps", lambda default: [100])


def _files(out: Path) -> set[str]:
    return {path.relative_to(out).as_posix() for path in out.rglob("*") if path.is_file()}


async def test_export_writes_every_planned_file_and_no_key(make_services: ServicesFactory, tmp_path: Path) -> None:
    services = _seeded(make_services(_refuse, api_key=SENTINEL))
    out = tmp_path / "site"
    summary = await export_site(services.settings, out, commit="abc1234def", static_dir=_fake_ui(tmp_path))
    assert _files(out) == _expected_files()
    assert summary.files == len(_expected_files())
    manifest = json.loads((out / "api" / "site.json").read_text(encoding="utf-8"))
    assert manifest["commit"] == "abc1234def"
    assert [view["label"] for view in manifest["views"]] == ["Latest without kev", "Latest without embeddings", "Generation only"]
    assert json.loads((out / "api" / "status.json").read_text(encoding="utf-8")) == {"server_key": False}
    assert (out / "js" / "deployment.js").read_text(encoding="utf-8") == STATIC_MODULE
    assert [path for path in out.rglob("*") if path.is_file() and SENTINEL.encode() in path.read_bytes()] == []


def _running_job(services: Services, out: Path) -> None:
    _seed_run(services, "20260924-120000-jev-0003", "jev", status="running")


def _stale_output(services: Services, out: Path) -> None:
    out.mkdir()
    (out / "old.txt").write_text("x", encoding="utf-8")


@pytest.mark.parametrize(
    ("prepare", "message"),
    [
        pytest.param(_running_job, "still running", id="running-job"),
        pytest.param(_stale_output, "not an empty directory", id="non-empty-output"),
    ],
)
async def test_export_refuses(
    make_services: ServicesFactory, tmp_path: Path, prepare: Callable[[Services, Path], None], message: str
) -> None:
    services = _seeded(make_services(_refuse))
    out = tmp_path / "site"
    prepare(services, out)
    with pytest.raises(ExportError, match=message):
        await export_site(services.settings, out, static_dir=_fake_ui(tmp_path))
```

Run: `uv run pytest tests/test_site_export.py -q`. Expected: FAIL (`No module named 'jev_bench.site.export'`).

- [ ] **Step 2: Implement `export.py`**

```python
"""Snapshot export: run every planned GET against the real app in-process and write the static site.

The app is built without its lifespan (so `sweep_interrupted` never runs and `data/` is only read), over
offline services (no network, no key, no OpenRouter model list); response bodies are written unchanged.

Classes:
    ExportSummary: files written, their total size and the duration.
    SiteManifest: api/site.json, the build time, the commit and the published views.
Functions:
    export_site: refuse a non-empty output or a running job, copy the UI in static mode, write every
        planned file and the manifest.
"""

import time
from datetime import UTC, datetime
from pathlib import Path

import httpx2
from pydantic import AwareDatetime, BaseModel

from jev_bench.services import Services
from jev_bench.settings import Settings
from jev_bench.site.errors import ExportError
from jev_bench.site.offline import offline_services, refusing_client
from jev_bench.site.paths import ThresholdEndpoint
from jev_bench.site.plan import base_requests, email_requests, multi_threshold, threshold_requests
from jev_bench.site.static_copy import copy_static
from jev_bench.site.thresholds import threshold_steps
from jev_bench.site.views import View, derive_views
from jev_bench.site.writer import SiteWriter
from jev_bench.web.app import STATIC_DIR, create_app

__all__ = [
    "ExportSummary",
    "SiteManifest",
    "export_site",
]

MANIFEST_FILE = "api/site.json"


class ExportSummary(BaseModel):
    files: int
    bytes: int
    seconds: float


class SiteManifest(BaseModel):
    built_at: AwareDatetime
    commit: str | None
    views: list[View]


async def export_site(
    settings: Settings, out: Path, *, commit: str | None = None, static_dir: Path = STATIC_DIR
) -> ExportSummary:
    started = time.perf_counter()
    _require_empty(out)
    async with refusing_client() as http:
        services = offline_services(settings, http)
        _refuse_running(services)
        views = _views(services)
        copy_static(static_dir, out)
        async with _app_client(services, http) as client:
            writer = SiteWriter(client, out)
            await _write_api(writer, services, views)
            _write_manifest(writer, views, commit)
    return _summary(out, time.perf_counter() - started)


def _require_empty(out: Path) -> None:
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise ExportError(f"{out} exists and is not an empty directory")
    out.mkdir(parents=True, exist_ok=True)


def _refuse_running(services: Services) -> None:
    metas = [*services.generations.list_metas(), *services.runs.list_metas(), *services.analyses.list_metas()]
    running = [meta.id for meta in metas if meta.status == "running"]
    if running:
        raise ExportError(f"jobs still running (finish or cancel them first): {', '.join(running)}")


def _views(services: Services) -> list[View]:
    generation_ids = [meta.id for meta in services.generations.list_metas()]
    return derive_views(
        services.benchmark_config().columns, generation_ids, services.runs.list_metas(), services.analyses.list_metas()
    )


def _app_client(services: Services, http: httpx2.AsyncClient) -> httpx2.AsyncClient:
    app = create_app(services.settings, http=http)
    app.state.services = services
    return httpx2.AsyncClient(transport=httpx2.ASGITransport(app=app), base_url="http://snapshot.invalid")


async def _write_api(writer: SiteWriter, services: Services, views: list[View]) -> None:
    generations = [meta.id for meta in services.generations.list_metas()]
    runs = [meta.id for meta in services.runs.list_metas()]
    analyses = [meta.id for meta in services.analyses.list_metas()]
    for request in base_requests(generations, runs, analyses):
        await writer.write(request)
    for view in views:
        await _write_view(writer, services, view)


async def _write_view(writer: SiteWriter, services: Services, view: View) -> None:
    endpoints: tuple[ThresholdEndpoint, ...] = ("compare", "emails") if view.run_ids else ("emails",)
    for endpoint in endpoints:
        await _write_thresholds(writer, endpoint, view)
    email_ids = [email.id for email in services.generations.emails_for(view.generation_ids)]
    for request in email_requests(view, email_ids):
        await writer.write(request)


async def _write_thresholds(writer: SiteWriter, endpoint: ThresholdEndpoint, view: View) -> None:
    (default,) = threshold_requests(endpoint, view, [None])
    threshold = multi_threshold(endpoint, (await writer.write(default)).json())
    steps: list[int | None] = [] if threshold is None else [*threshold_steps(threshold)]
    for request in threshold_requests(endpoint, view, steps):
        await writer.write(request)


def _write_manifest(writer: SiteWriter, views: list[View], commit: str | None) -> None:
    manifest = SiteManifest(built_at=datetime.now(UTC), commit=commit, views=views)
    writer.put(MANIFEST_FILE, manifest.model_dump_json().encode())


def _summary(out: Path, seconds: float) -> ExportSummary:
    files = [path for path in out.rglob("*") if path.is_file()]
    return ExportSummary(files=len(files), bytes=sum(path.stat().st_size for path in files), seconds=seconds)
```

- [ ] **Step 3: Run the tests and measure**

Run: `uv run pytest tests/test_site_export.py -q --durations=5`
Expected: PASS. If `test_export_writes_every_planned_file_and_no_key` takes more than 50 ms (it runs 4
bootstrap compares), monkeypatch the bootstrap size in that test file only:
`monkeypatch.setattr("jev_bench.web.routes.compare.compare", functools.partial(compare, resamples=20))`,
with `from jev_bench.compare import compare`. That is rule 2 of the fast-tests skill: patch the batch
constant. Re-measure. If it is still over 50 ms, list it in `tests/slow_tests.txt` and say so in the task
report. Don't raise the timeout.

- [ ] **Step 4: Run the real export once against `data/`**

Run: `uv run python -c "import asyncio; from pathlib import Path; from jev_bench.settings import load_settings; from jev_bench.site.export import export_site; print(asyncio.run(export_site(load_settings(), Path('/private/tmp/claude-501/-Users-yarnaid-projects-jev-benchmark/f2eb5943-c1f2-4355-9c94-dce033f767af/scratchpad/site-probe'))))" && git status --porcelain data`
Expected: about 550 files (19 base, 84 threshold files, 400 email details, the manifest, ~40 UI
files), about 15 MB, under 60 s; `git status` prints nothing (`data/` untouched).
Delete the probe directory afterwards.

- [ ] **Step 5: Full suite and coverage**

Run: `uv run pytest --cov --cov-fail-under=95 -q` and `uv run ruff check --fix && uv run ruff format && uv run pyright`
Expected: PASS, coverage ≥ 95%, 0 pyright errors.

- [ ] **Step 6: Commit**

```bash
git add src/jev_bench/site/export.py tests/test_site_export.py tests/slow_tests.txt
git commit -m "feat(site): export the static snapshot from the real routes, offline and keyless"
```

---

### Task 10: `jev-bench export-site`

**Files:**
- Create: `src/jev_bench/cli_export.py`
- Modify: `src/jev_bench/cli.py` (docstring, `__all__`, new command)
- Test: `tests/test_cli_export.py`, `tests/test_cli.py`

**Interfaces:**
- Consumes: `export_site`, `ExportSummary` (Task 9), `ExportError` (Task 6).
- Produces: `async export_and_report(out: Path, commit: str | None) -> int`; CLI
  `jev-bench export-site OUT [--commit SHA]`.

- [ ] **Step 1: Write the failing tests**

`tests/test_cli_export.py`:

```python
"""Tests for jev_bench.cli_export."""

from pathlib import Path

import pytest

from jev_bench import cli_export
from jev_bench.site.errors import ExportError
from jev_bench.site.export import ExportSummary


@pytest.mark.parametrize(
    ("outcome", "code", "text"),
    [
        pytest.param(ExportSummary(files=3, bytes=2_500_000, seconds=1.3), 0, "Exported 3 files · 2.5 MB · 1.3 s", id="success"),
        pytest.param(ExportError("jobs still running: r1"), 1, "Export failed: jobs still running: r1", id="failure"),
    ],
)
async def test_export_and_report(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path,
    outcome: ExportSummary | ExportError, code: int, text: str,
) -> None:
    seen: list[tuple[Path, str | None]] = []

    async def fake(settings: object, out: Path, *, commit: str | None = None) -> ExportSummary:
        seen.append((out, commit))
        if isinstance(outcome, ExportError):
            raise outcome
        return outcome

    monkeypatch.setattr(cli_export, "export_site", fake)
    assert await cli_export.export_and_report(tmp_path / "site", "abc") == code
    assert seen == [(tmp_path / "site", "abc")]
    assert text in capsys.readouterr().err
```

In `tests/test_cli.py`: add `"export-site"` to the tuple in `test_help_lists_commands`, and add:

```python
@pytest.mark.parametrize(
    ("args", "commit"),
    [
        pytest.param([], None, id="without-commit"),
        pytest.param(["--commit", "abc1234"], "abc1234", id="with-commit"),
    ],
)
def test_export_site_delegates(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, args: list[str], commit: str | None
) -> None:
    from jev_bench import cli_export

    seen: list[tuple[Path, str | None]] = []

    async def fake(out: Path, given: str | None) -> int:
        seen.append((out, given))
        return 0

    monkeypatch.setattr(cli_export, "export_and_report", fake)
    result = runner.invoke(app, ["export-site", str(tmp_path / "site"), *args])
    assert result.exit_code == 0
    assert seen == [(tmp_path / "site", commit)]
```

(Add `from pathlib import Path` to the imports of `tests/test_cli.py` if it is not there yet.)

Run: `uv run pytest tests/test_cli_export.py tests/test_cli.py -q`. Expected: FAIL (module/command missing).

- [ ] **Step 2: Implement**

`src/jev_bench/cli_export.py`:

```python
"""Async body of `jev-bench export-site`: export the static snapshot and report the outcome.

Functions:
    export_and_report: run export_site with the environment's Settings; print a green summary or a red
        reason on stderr; return the exit code.
"""

from pathlib import Path

from rich.console import Console
from rich.text import Text

from jev_bench.settings import load_settings
from jev_bench.site.errors import ExportError
from jev_bench.site.export import export_site

__all__ = [
    "console",
    "export_and_report",
]

console = Console(stderr=True)


async def export_and_report(out: Path, commit: str | None) -> int:
    try:
        summary = await export_site(load_settings(), out, commit=commit)
    except ExportError as exc:
        console.print(Text.assemble(("Export failed: ", "red"), str(exc)))
        return 1
    size = f"{summary.files} files · {summary.bytes / 1_000_000:.1f} MB · {summary.seconds:.1f} s"
    console.print(Text.assemble(("Exported ", "green"), f"{size} → {out}"))
    return 0
```

`src/jev_bench/cli.py`: update the module docstring's first line to
`"""Command-line entry point: \`serve\`, \`generate\`, \`run\`, \`export-site\`.` and add to its Functions
list `export_site: write the committed results as a static, read-only site.`; add `"export_site"` to
`__all__`; add `from pathlib import Path` to the top imports (stdlib, cheap); append:

```python
@app.command()
def export_site(
    out: Annotated[Path, typer.Argument(help="Missing or empty directory for the static site.")],
    commit: Annotated[
        str | None, typer.Option(help="Commit the snapshot is built from (shown on the site).")
    ] = None,
) -> None:
    """Export the committed results as a static, read-only site (GitHub Pages)."""
    import asyncio

    from jev_bench import cli_export
    from jev_bench.log_setup import configure_logging

    configure_logging()
    raise typer.Exit(asyncio.run(cli_export.export_and_report(out, commit)))
```

- [ ] **Step 3: Verify tests, help speed and a real run**

Run: `uv run pytest tests/test_cli_export.py tests/test_cli.py -q --durations=5`. Expected: PASS.
Run: `time uv run jev-bench --help`. Expected: `export-site` is listed; under 500 ms warm (run 3×, drop
the first).
Run: `uv run jev-bench export-site /private/tmp/claude-501/-Users-yarnaid-projects-jev-benchmark/f2eb5943-c1f2-4355-9c94-dce033f767af/scratchpad/site-cli --commit test`
Expected: a green `Exported … files` line. Running it again on the same directory prints a red
`not an empty directory` line and exits 1. Delete the directory afterwards.

- [ ] **Step 4: Commit**

```bash
git add src/jev_bench/cli.py src/jev_bench/cli_export.py tests/test_cli.py tests/test_cli_export.py
git commit -m "feat(cli): export-site writes the static read-only snapshot"
```

---

### Task 11: Static API client

**Files:**
- Create: `src/jev_bench/web/static/js/deployment.js`
- Create: `src/jev_bench/web/static/js/snapshot.js`
- Modify: `src/jev_bench/web/static/js/api.js`
- Modify: `tests/test_web_static.py` (`SHELL_MODULES` += `"deployment.js"`, `PAGE_MODULES` += `"snapshot.js"`)
- Test: `tests/js/api-static.test.mjs`, `tests/js/snapshot.test.mjs`

**Interfaces:**
- Consumes: `staticFile`, `NotPublished` (Task 4).
- Produces:
  - `deployment.js`: `STATIC`;
  - `api.js` (new exports): `NOT_PUBLISHED: string`, `notPublished(error) -> boolean`,
    `siteManifest() -> Promise<manifest>`, `staticRequest(method, path) -> Promise<payload>`;
  - `snapshot.js`: `snapshotInfo(manifest, repoUrl) -> { text, href, title }`.

- [ ] **Step 1: Write the failing tests**

`tests/js/api-static.test.mjs`:

```js
// Run with `node --test tests/js/`. Exercises the static-snapshot path of src/jev_bench/web/static/js/api.js.
import assert from "node:assert/strict";
import test from "node:test";

globalThis.localStorage = { getItem: () => null, setItem: () => {}, removeItem: () => {} };
const MANIFEST = { built_at: "2026-09-29T12:00:00Z", commit: null, views: [{ id: "v1", generation_ids: ["g1"], run_ids: ["r1"] }] };
const FILES = new Map([["api/site.json", MANIFEST], ["api/compare/v1/t-default.json", { raters: [] }], ["api/status.json", { server_key: false }]]);
const fetched = [];
const reply = (status, body) => ({ ok: status === 200, status, headers: new Map([["content-type", status === 200 ? "application/json" : "text/html"]]), json: async () => body, text: async () => "<html>404</html>" });
globalThis.fetch = async (url, init = {}) => {
  fetched.push({ url, init });
  return FILES.has(url) ? reply(200, FILES.get(url)) : reply(404, null);
};

const { ApiError, NOT_PUBLISHED, notPublished, staticRequest } = await import("../../src/jev_bench/web/static/js/api.js");

test("a published GET is answered by its file", async () => {
  assert.deepEqual(await staticRequest("GET", "/compare?runs=r1"), { raters: [] });
});

test("the manifest is fetched once and revalidated", async () => {
  await staticRequest("GET", "/status");
  const manifests = fetched.filter(({ url }) => url === "api/site.json");
  assert.equal(manifests.length, 1);
  assert.equal(manifests[0].init.cache, "no-cache");
});

const REFUSED = [
  ["an unpublished selection", "GET", "/compare?runs=r2"],
  ["a write", "POST", "/runs"],
  ["a mapped file missing from the snapshot", "GET", "/runs/r9"],
];

for (const [name, method, path] of REFUSED) {
  test(`${name} is reported as not published`, async () => {
    await assert.rejects(staticRequest(method, path), (error) => error instanceof ApiError && error.status === 404 && error.message === NOT_PUBLISHED && notPublished(error));
  });
}
```

`tests/js/snapshot.test.mjs`:

```js
// Run with `node --test tests/js/`. Exercises src/jev_bench/web/static/js/snapshot.js (pure).
import assert from "node:assert/strict";
import test from "node:test";

const { snapshotInfo } = await import("../../src/jev_bench/web/static/js/snapshot.js");
const REPO = "https://github.com/yarnaid/jev-benchmark";

const CASES = [
  ["with a commit", { built_at: "2026-09-29T12:00:00Z", commit: "9c3c2956489458b3" }, { text: "Snapshot · 9c3c295 · 2026-09-29", href: `${REPO}/commit/9c3c2956489458b3` }],
  ["without a commit", { built_at: "2026-09-29T12:00:00Z", commit: null }, { text: "Snapshot · 2026-09-29", href: null }],
];

for (const [name, manifest, expected] of CASES) {
  test(`snapshotInfo ${name}`, () => {
    const { text, href } = snapshotInfo(manifest, REPO);
    assert.deepEqual({ text, href }, expected);
  });
}
```

Run: `node --test tests/js/api-static.test.mjs tests/js/snapshot.test.mjs`. Expected: FAIL (exports and
module missing).

- [ ] **Step 2: Implement**

`src/jev_bench/web/static/js/deployment.js`:

```js
/**
 * Deployment flag: false when the FastAPI server serves this UI; `jev-bench export-site` overwrites this
 * file with `export const STATIC = true;` in the static snapshot.
 * Exports: STATIC.
 */
export const STATIC = false;
```

`src/jev_bench/web/static/js/snapshot.js`:

```js
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
```

`src/jev_bench/web/static/js/api.js`: replace the module docstring and `request()`, and add the static
path. The rest of the file (`query`, `segment`, `api`, `needsKey`) stays as it is.

```js
/**
 * JSON client for the /api endpoints; the browser-stored key is attached only to job-starting calls.
 * Query parameters that are undefined, null or "" are omitted (an unset label threshold is never sent).
 * In the static snapshot (deployment.js STATIC) every GET is answered by the file static-api.js maps it
 * to, through the manifest api/site.json (fetched once, revalidated); anything else, and a mapped file
 * the snapshot lacks, is ApiError(404, NOT_PUBLISHED).
 * Exports: api, ApiError, needsKey, NOT_PUBLISHED, notPublished, siteManifest, staticRequest.
 */
import { STATIC } from "./deployment.js";
import { getKey } from "./key.js";
import { NotPublished, staticFile } from "./static-api.js";

export const NOT_PUBLISHED = "Not in this snapshot: the published site holds the Latest selections only. Run jev-bench locally to explore any combination.";

export class ApiError extends Error {
  constructor(status, detail) {
    super(typeof detail === "string" ? detail : JSON.stringify(detail));
    this.status = status;
  }
}

async function parse(response) {
  const isJson = (response.headers.get("content-type") ?? "").includes("json");
  const payload = isJson ? await response.json() : await response.text();
  if (!response.ok) throw new ApiError(response.status, payload?.detail ?? payload);
  return payload;
}

async function request(method, path, { body, withKey = false } = {}) {
  if (STATIC) return staticRequest(method, path);
  const headers = { Accept: "application/json" };
  if (body !== undefined) headers["Content-Type"] = "application/json";
  const key = withKey ? getKey() : null;
  if (key) headers["X-OpenRouter-Key"] = key;
  return parse(await fetch(`api${path}`, { method, headers, body: body === undefined ? undefined : JSON.stringify(body) }));
}

let manifest = null;

export function siteManifest() {
  manifest ??= fetch("api/site.json", { cache: "no-cache", headers: { Accept: "application/json" } }).then(parse);
  return manifest;
}

function resolveFile(method, path, site) {
  try {
    return staticFile(method, path, site);
  } catch (error) {
    throw error instanceof NotPublished ? new ApiError(404, NOT_PUBLISHED) : error;
  }
}

export async function staticRequest(method, path) {
  const response = await fetch(resolveFile(method, path, await siteManifest()), { headers: { Accept: "application/json" } });
  if (response.status === 404) throw new ApiError(404, NOT_PUBLISHED);
  return parse(response);
}

export const notPublished = (error) => error instanceof ApiError && error.message === NOT_PUBLISHED;
```

Keep `query`, `segment`, `api` and `needsKey` below this, unchanged.

In `tests/test_web_static.py`: `SHELL_MODULES` += `"deployment.js"`, `PAGE_MODULES` += `"snapshot.js"`.

- [ ] **Step 3: Run the tests to verify they pass**

Run: `node --check src/jev_bench/web/static/js/*.js && node --test tests/js/ && uv run pytest tests/test_web_static.py -q`
Expected: PASS (the existing `api.test.mjs` still passes: live mode is unchanged).

- [ ] **Step 4: Commit**

```bash
git add src/jev_bench/web/static/js/deployment.js src/jev_bench/web/static/js/snapshot.js src/jev_bench/web/static/js/api.js tests/js/api-static.test.mjs tests/js/snapshot.test.mjs tests/test_web_static.py
git commit -m "feat(ui): static-snapshot API client answering GETs from published files"
```

---

### Task 12: Static mode on the pages

**Files:**
- Modify: `src/jev_bench/web/static/js/layout.js` (data-static, snapshot badge)
- Modify: `src/jev_bench/web/static/css/app.css` (append)
- Modify: `src/jev_bench/web/static/js/widgets.js` (`checklist` `disabled`)
- Modify: `src/jev_bench/web/static/js/column-card.js:20`
- Modify: `src/jev_bench/web/static/js/benchmark.js` (estimates, run picker, compare error)
- Modify: `src/jev_bench/web/static/js/explorer.js` (run picker, rows error)
- Modify: `src/jev_bench/web/static/js/email-detail.js` (label controls)
- Modify: `src/jev_bench/web/static/js/runs.js:80`, `runs.html`, `generations.html`
- Modify: `src/jev_bench/web/static/analyze.html`, `js/analyze.js` (estimates)
- Test: browser check (Step 3); node/pytest suites stay green.

**Interfaces:**
- Consumes: `STATIC` (Task 11), `siteManifest`, `notPublished` (Task 11), `snapshotInfo` (Task 11).
- Produces: the CSS contract `[data-static] .write-only { display: none }`. `checklist({ …, disabled })`.

- [ ] **Step 1: Shared chrome**

`app.css`, append:

```css
[data-static] .write-only {
  display: none !important;
}

[data-static] #result {
  width: 100%;
}
```

`widgets.js`: change the signature to `export function checklist({ label, items, selected, onChange, disabled = false })`,
add `disabled` to the checkbox attributes (`h("input", { …, checked: chosen.has(item.value), disabled, onchange: … })`),
and mention in the module docstring: "a checklist can be `disabled` (read-only, still showing the selection)".

`layout.js`: add the imports `import { STATIC } from "./deployment.js";`, `siteManifest` next to the existing
`api, needsKey` import, and `import { snapshotInfo } from "./snapshot.js";`. At the top of `initLayout()`:

```js
  if (STATIC) document.documentElement.dataset.static = "";
```

Replace `refreshKeyBadge()` with:

```js
async function refreshKeyBadge() {
  const target = document.getElementById("key-badge");
  if (STATIC) return clear(target, snapshotBadge(await siteManifest()));
  const serverKey = await api.status().then((status) => status.server_key, () => false);
  clear(target, keyButton(Boolean(getKey()), serverKey));
}

function snapshotBadge(manifest) {
  const { text, href, title } = snapshotInfo(manifest, REPO_URL);
  const attrs = { class: "btn btn-sm btn-outline-light", title };
  return href ? h("a", { ...attrs, href, target: "_blank", rel: "noopener noreferrer" }, icon("camera"), ` ${text}`) : h("span", attrs, icon("camera"), ` ${text}`);
}
```

Update the `layout.js` docstring: "…navbar (pages, API-key badge or, in the static snapshot, the snapshot
badge, GitHub link, theme toggle)…".

- [ ] **Step 2: Pages**

`column-card.js:20`: in `body`, add `write-only` to the model-picker wrapper, the estimate div and the button row:

```js
  const body = h("div", { class: "card-body d-flex flex-column gap-2" }, warning, h("div", { class: "quality-slot d-flex flex-column gap-2", id: `quality-${column.id}` }), h("div", { class: "write-only" }, h("label", { class: "form-label small mb-1" }, "Model"), select, extras(column, mode, onMode)), h("div", { class: "estimate small write-only", id: `estimate-${column.id}` }), h("div", { class: "d-flex gap-2 write-only" }, run, cancel), h("div", { id: `stats-${column.id}` }));
```

`benchmark.js`:
- imports: `import { api, notPublished } from "./api.js";` and `import { STATIC } from "./deployment.js";`;
- first line of `scheduleEstimate`: `if (STATIC) return;`;
- in `renderRunPicker`, pass `disabled: STATIC` to the Runs `checklist`;
- in `refreshComparison`, replace `state.report = await api.compare(runIds, state.threshold);` with:

```js
  try {
    state.report = await api.compare(runIds, state.threshold);
  } catch (error) {
    if (!notPublished(error)) throw error;
    return clear(container, emptyState(error.message, "camera"));
  }
```

`explorer.js`:
- imports: `notPublished` from `./api.js`, `STATIC` from `./deployment.js`;
- in `renderPickers`, pass `disabled: STATIC` to the Runs `checklist`;
- add a helper and use it in `loadRows` and `refreshRows`:

```js
async function fetchRows(target) {
  try {
    return await api.emails(state.selectedGenerations, activeRuns(), state.threshold);
  } catch (error) {
    if (!notPublished(error)) throw error;
    clear(document.getElementById("filters"));
    clear(target, emptyState(error.message, "camera"));
    return null;
  }
}
```

In `loadRows`: `const list = await fetchRows(target); if (!list) return;` in place of the direct call.
In `refreshRows`:

```js
async function refreshRows() {
  const list = await fetchRows(document.getElementById("email-table"));
  if (!list) return;
  state.rows = list.rows;
  renderTable();
}
```

`email-detail.js`: in `questionSection`, the `human` div gets
`class: "d-flex align-items-center gap-2 write-only"`; in `labellingForm`, the save row gets
`class: "d-flex justify-content-end mt-2 write-only"`.

`runs.js:80`: the row checkbox gets `class: "form-check-input write-only"`. `runs.html`: add `write-only`
to the `class` of the `#compare` and `#explore` buttons.

`generations.html`: `<div class="card shadow-sm mb-4 write-only">` for the "New generation" card.

`analyze.html`: add `write-only` to `#threshold-control`, `#pickers` and `#setup`
(`<div class="col-12 col-xl-5 write-only" id="setup">`). `analyze.js`: import `STATIC` from
`./deployment.js`; first line of `scheduleEstimate`: `if (STATIC) return;`.

- [ ] **Step 3: Verify in a browser, both modes**

Run: `node --check src/jev_bench/web/static/js/*.js && node --test tests/js/ && uv run pytest -q`. Expected: PASS.

Live mode: `uv run jev-bench serve --port 8123` (background). Playwright: Benchmark shows the model
pickers, estimates and Run buttons; the Runs checklist is enabled; no console errors. Stop the server.

Static mode (set `SCRATCH=/private/tmp/claude-501/-Users-yarnaid-projects-jev-benchmark/f2eb5943-c1f2-4355-9c94-dce033f767af/scratchpad`):
`uv run jev-bench export-site "$SCRATCH/pages/jev-benchmark"`, then
`uv run python -m http.server 8124 --bind 127.0.0.1 --directory "$SCRATCH/pages"` (background).
Playwright on `http://127.0.0.1:8124/jev-benchmark/`:
- the navbar shows the snapshot badge (no key button);
- the Benchmark cards show quality and stats, with no pickers, estimates or buttons;
- the Runs checklist is disabled;
- the slot toggle swaps Embeddings and Kev, and the comparison reloads;
- the slider at 50 % and at 100 % reloads the comparison;
- Explorer: the rows load; an email opens without the label controls;
- Runs / Generations: the lists show, with no selection or form;
- Analyze: the saved report shows full width; an `e001` link opens the Explorer on that email.

Expected: 0 console errors and 0 failed requests (`browser_console_messages`, `browser_network_requests`).
Stop the server and delete `$SCRATCH/pages`.

- [ ] **Step 4: Commit**

```bash
git add src/jev_bench/web/static
git commit -m "feat(ui): read-only static mode (snapshot badge, hidden write controls, fixed run pickers)"
```

---

### Task 13: GitHub Actions workflow

**Files:**
- Create: `.github/workflows/pages.yml`
- Test: `tests/test_workflows.py`

**Interfaces:**
- Consumes: `jev-bench export-site` (Task 10).

- [ ] **Step 1: Write the failing test**

`tests/test_workflows.py`:

```python
"""Contract tests for the GitHub Actions workflows: every action is pinned to a full commit SHA."""

import re
from pathlib import Path

import pytest

WORKFLOWS = sorted((Path(__file__).parents[1] / ".github" / "workflows").glob("*.yml"))
USES = re.compile(r"^\s*(?:-\s*)?uses:\s*(\S+)", re.MULTILINE)
PINNED = re.compile(r"[\w.-]+/[\w.-]+@[0-9a-f]{40}")


def test_the_pages_workflow_exists() -> None:
    assert [path.name for path in WORKFLOWS] == ["pages.yml"]


@pytest.mark.parametrize("workflow", WORKFLOWS, ids=lambda path: path.name)
def test_actions_are_pinned_to_commit_shas(workflow: Path) -> None:
    uses = USES.findall(workflow.read_text(encoding="utf-8"))
    assert uses
    assert [action for action in uses if not PINNED.fullmatch(action)] == []
```

Run: `uv run pytest tests/test_workflows.py -q`. Expected: FAIL (`[] == ['pages.yml']`).

- [ ] **Step 2: Look up the current releases and their commits**

```bash
for repo in actions/checkout astral-sh/setup-uv actions/upload-pages-artifact actions/deploy-pages; do
  tag=$(timeout 20 gh api "repos/$repo/releases/latest" --jq .tag_name)
  sha=$(timeout 20 gh api "repos/$repo/commits/$tag" --jq .sha)
  echo "$repo $tag $sha"
done
```

Check each action's inputs against its README for that tag (`upload-pages-artifact`: `path`; `setup-uv`:
`enable-cache`; `deploy-pages`: output `page_url`). Use Context7 if it has them.

- [ ] **Step 3: Write the workflow** (substitute each `<sha>` / `<tag>` from Step 2)

```yaml
name: Pages

on:
  push:
    branches: [main]
  workflow_dispatch:

permissions:
  contents: read
  pages: write
  id-token: write

concurrency:
  group: pages
  cancel-in-progress: false

jobs:
  build:
    runs-on: ubuntu-latest
    timeout-minutes: 10
    steps:
      - uses: actions/checkout@<sha> # <tag>
        with:
          persist-credentials: false
      - uses: astral-sh/setup-uv@<sha> # <tag>
        with:
          enable-cache: true
      - name: Install
        run: uv sync --locked --no-dev
      - name: Export the snapshot
        run: uv run --no-sync jev-bench export-site _site --commit "$GITHUB_SHA"
      - uses: actions/upload-pages-artifact@<sha> # <tag>
        with:
          path: _site

  deploy:
    needs: build
    runs-on: ubuntu-latest
    timeout-minutes: 5
    environment:
      name: github-pages
      url: ${{ steps.deployment.outputs.page_url }}
    steps:
      - id: deployment
        uses: actions/deploy-pages@<sha> # <tag>
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/test_workflows.py -q`. Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add .github/workflows/pages.yml tests/test_workflows.py
git commit -m "ci(pages): build and deploy the static snapshot on every push to main"
```

---

### Task 14: Docs

**Files:**
- Modify: `CLAUDE.md` (Commands, Architecture)
- Modify: `README.md` (new "Published snapshot" section before "Development")

- [ ] **Step 1: CLAUDE.md**

In the Commands block, after the `run` line:

```bash
uv run jev-bench export-site _site        # static read-only snapshot of data/ (what GitHub Pages serves)
```

In Architecture, after the **Web** bullet:

```markdown
- **Static snapshot** (`site/`, spec `docs/superpowers/specs/2026-09-29-pages-snapshot-design.md`):
  `.github/workflows/pages.yml` runs `jev-bench export-site` on every push to `main` and deploys to
  `https://yarnaid.github.io/jev-benchmark/`.
  - `site/views.py` derives the published views from `data/` (Benchmark "Latest" per slot state, the
    Explorer's all-columns default, each generation alone, each completed analysis). View ids are
    content hashes, so a cached file never shows another selection.
  - `site/export.py` calls the real GET routes in-process (`httpx2.ASGITransport`, no lifespan, so
    `data/` is only read) over `site/offline.py` services: the network is refused, there is no key, and
    `OfflineCatalog` lists no models. It writes each body unchanged; `site/static_copy.py` copies the
    UI with `js/deployment.js` set to `STATIC = true` and the CSP as `<meta>` tags.
  - In static mode `js/api.js` answers GETs from files through `api/site.json` and `js/static-api.js`;
    `.write-only` controls are hidden. The Python and JS halves of three rules share
    `tests/fixtures/*.json`: view selection, slider steps and request → file.
  - The UI uses relative URLs only (`tests/test_web_static.py` enforces it), so it works under a sub-path.
```

- [ ] **Step 2: README.md**

Before `## Development`:

```markdown
## Published snapshot

A read-only snapshot of the committed results is at https://yarnaid.github.io/jev-benchmark/. GitHub
Actions rebuilds it on every push to `main` with `jev-bench export-site`. It shows:
- the Benchmark comparison, for the latest runs with either Embeddings or Kev;
- the Explorer and the saved analyses;
- every step of the label threshold.

It cannot start runs, hold a key or save labels. Run the app locally for that, or to compare any other
combination of runs.
```

- [ ] **Step 3: Commit**

```bash
git add CLAUDE.md README.md
git commit -m "docs: static snapshot architecture and published site"
```

---

### Task 15: Final verification and first deploy

- [ ] **Step 1: Full local gates**

Run: `uv run ruff check --fix && uv run ruff format && uv run pyright && uv run pytest --cov --cov-fail-under=95 -q && node --check src/jev_bench/web/static/js/*.js && node --test tests/js/`
Also: `uv run pytest --durations=10 -q`. Expected: all green; the default suite < 5 s; no new test > 50 ms
(or listed in `slow_tests.txt`).

- [ ] **Step 2: Local end-to-end under the Pages sub-path**

Repeat Task 12 Step 3's static walk on a fresh export, and also check each page's first load with an empty
`localStorage` (a fresh Playwright context): Benchmark, Explorer, Analyze. Expected: 0 console errors,
0 failed requests, no "Not in this snapshot" on any first load.

- [ ] **Step 3: Whole-branch review**

Dispatch a reviewer on `git diff main...feat/pages-snapshot` against the spec and this plan. Fix what it
confirms (each fix in its own commit).

- [ ] **Step 4: Ask the owner for the two outward-facing steps**

Ask before doing either:
1. Enable Pages with source "GitHub Actions":
   `gh api -X POST repos/yarnaid/jev-benchmark/pages -f build_type=workflow`.
2. Merge `feat/pages-snapshot` into `main` and push, which triggers the first deploy.

- [ ] **Step 5: Watch the deploy and verify the live site**

Run: `gh run list --workflow pages.yml --limit 1`, then `gh run watch <id> --exit-status` (with `timeout 900`).
Expected: build and deploy succeed. Then run the Step 2 walk against
`https://yarnaid.github.io/jev-benchmark/`. Expected: 0 console errors, 0 failed requests.
