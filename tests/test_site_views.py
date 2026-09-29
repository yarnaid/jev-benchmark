"""Tests for jev_bench.site.views: the shared JS cases, then views of committed data."""

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
        pytest.param(
            case["generation_ids"], frozenset(case["hidden"]), case["expected"], id=case["id"]
        )
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
ANALYSIS_ID = "20260928-070000-analysis-a"


def _analysis(
    run_ids: tuple[str, ...], generation_ids: tuple[str, ...] = ("g1",), status: str = "completed"
) -> Analysis:
    return Analysis(ANALYSIS_ID, status, generation_ids, run_ids)


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
            ["g1"],
            [JEV],
            [],
            [("Latest without kev", ("g1",), (JEV.id,)), ("Generation only", ("g1",), ())],
            id="equal-selections-merge",
        ),
        pytest.param(
            ["g2"], FULL, [], [("Generation only", ("g2",), ())], id="generation-without-runs"
        ),
        pytest.param(
            ["g1"],
            FULL,
            [_analysis((OLD_JEV.id, CLAUDE.id))],
            [*SLOT_VIEWS, (f"Analysis {ANALYSIS_ID}", ("g1",), (OLD_JEV.id, CLAUDE.id))],
            id="analysis-view",
        ),
        pytest.param(
            ["g1"],
            FULL,
            [_analysis((JEV.id, CLAUDE.id, EMBEDDINGS.id))],
            SLOT_VIEWS,
            id="analysis-equal-to-latest",
        ),
        pytest.param(
            ["g1"],
            FULL,
            [_analysis(("20260925-000000-jev-gone",))],
            SLOT_VIEWS,
            id="analysis-missing-run",
        ),
        pytest.param(
            ["g1"],
            FULL,
            [_analysis((JEV.id,), ("g1", "g9"))],
            SLOT_VIEWS,
            id="analysis-missing-generation",
        ),
        pytest.param(
            ["g1"],
            FULL,
            [_analysis((OLD_JEV.id,), status="failed")],
            SLOT_VIEWS,
            id="analysis-not-completed",
        ),
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
