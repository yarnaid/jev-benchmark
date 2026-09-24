"""Comparison route: the full report for a set of runs plus the generator reference and human
labels.

Constants:
    LIGHT: response fields excluded from embedded run metas.
Functions:
    compare_runs: GET /compare?runs=a,b,c
"""

from fastapi import APIRouter, HTTPException

from jev_bench.compare import ComparisonReport, compare, human_rater, reference_rater
from jev_bench.web.deps import ServicesDep, split_ids
from jev_bench.web.loaders import (
    generation_snapshots,
    generations_of,
    load_emails,
    load_runs,
    run_raters,
)

router = APIRouter(tags=["compare"])
LIGHT = {"raters": {"__all__": {"run": {"question_set", "params"}}}}


@router.get("/compare", response_model_exclude=LIGHT)
def compare_runs(services: ServicesDep, runs: str) -> ComparisonReport:
    metas = load_runs(services, split_ids(runs))
    if not metas:
        raise HTTPException(status_code=400, detail="no run ids given")
    base = metas[0].question_set
    generation_ids = generations_of(metas)
    reference = reference_rater(
        load_emails(services, generation_ids),
        base,
        generation_snapshots(services, generation_ids),
    )
    raters = [*run_raters(services, metas), reference]
    human = human_rater(services.labels.for_generations(generation_ids), base)
    if human is not None:
        raters.append(human)
    return compare(raters, base, runs={meta.id: meta for meta in metas})
