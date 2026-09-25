"""Comparison route: the full report for a set of runs plus the generator reference and human
labels.

Constants:
    LIGHT: response fields excluded from embedded run metas.
Functions:
    compare_runs: GET /compare?runs=a,b,c[&threshold=0.8]. The base question set is the snapshot of
        the most recent selected run (web.loaders.load_comparison), so the URL order of `runs`
        never changes the report.
"""

from fastapi import APIRouter

from jev_bench.compare import ComparisonReport, compare
from jev_bench.web.deps import ServicesDep, ThresholdQuery, split_ids
from jev_bench.web.loaders import load_comparison

__all__ = [
    "LIGHT",
    "compare_runs",
    "router",
]

router = APIRouter(tags=["compare"])
LIGHT = {"raters": {"__all__": {"run": {"question_set", "params"}}}}


@router.get("/compare", response_model_exclude=LIGHT)
def compare_runs(
    services: ServicesDep, runs: str, threshold: ThresholdQuery = None
) -> ComparisonReport:
    inputs = load_comparison(services, split_ids(runs))
    metas = {meta.id: meta for meta in inputs.metas}
    return compare(inputs.raters(), inputs.base, runs=metas, threshold=threshold)
