"""Published selections of the static snapshot: the (generation set, run set) pairs it precomputes.

A view mirrors a selection the UI makes by itself:
- the Benchmark's "Latest" for every card-slot state (`selection.defaultRunIds` over
  `slots.hiddenColumnIds`);
- the Explorer's default over all columns (`selection.latestCompletedPerColumn`);
- a generation alone;
- every completed analysis whose generations and runs are all committed.

`tests/fixtures/default_views.json` holds the cases the Python and JS rules must both pass. Runs are
listed in catalog column order, the order the Benchmark requests them in.

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
    def of(cls, label: str, generation_ids: Iterable[str], run_ids: Iterable[str]) -> View:
        generations, runs = tuple(generation_ids), tuple(run_ids)
        view_id = _view_id(generations, runs)
        return cls(id=view_id, label=label, generation_ids=generations, run_ids=runs)


def _view_id(generation_ids: tuple[str, ...], run_ids: tuple[str, ...]) -> str:
    text = ",".join(sorted(generation_ids)) + "|" + ",".join(sorted(run_ids))
    return "v" + hashlib.sha256(text.encode()).hexdigest()[:12]


def latest_per_column(
    runs: Iterable[RunLike], generation_ids: Iterable[str], hidden: frozenset[str] = frozenset()
) -> tuple[str, ...]:
    wanted = set(generation_ids)
    chosen: dict[str, str] = {}
    for run in sorted(runs, key=lambda run: run.id, reverse=True):
        on_generations = set(run.generation_ids) == wanted
        if run.status == "completed" and on_generations and run.column not in hidden:
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
    per_generation = [
        view for gid in generation_ids for view in _generation_views(columns, gid, runs, rank)
    ]
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
    latest = [
        (_latest_label(hidden), latest_per_column(runs, [generation_id], hidden))
        for hidden in hidden_sets(columns)
    ]
    pairs = [*latest, (ALL_COLUMNS, latest_per_column(runs, [generation_id]))]
    views = [
        View.of(label, [generation_id], sorted(ids, key=rank.__getitem__))
        for label, ids in pairs
        if ids
    ]
    return [*views, View.of(GENERATION_ONLY, [generation_id], [])]


def _analysis_views(
    analyses: Iterable[AnalysisLike], generations: set[str], runs: set[str]
) -> list[View]:
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
        logger.warning(
            "Analysis {} is not published: some of its generations or runs are not committed",
            analysis.id,
        )
    return complete


def _unique(views: Iterable[View]) -> list[View]:
    kept: dict[str, View] = {}
    for view in views:
        kept.setdefault(view.id, view)
    return list(kept.values())
