"""Run routes: list (optionally by exact generation set), create (background job), read, cancel.

Constants:
    LIGHT: response fields excluded from run views.
Classes:
    RunView: meta + live progress.
Functions:
    list_runs, create_run, get_run, cancel_run: route handlers.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from jev_bench.jobs import ProgressView
from jev_bench.run_launcher import RunLaunchError, RunRequest, launch_run
from jev_bench.services import Services
from jev_bench.store.runs import RunMeta
from jev_bench.web.deps import ApiKeyDep, ServicesDep, split_ids
from jev_bench.web.views import CancelView, load_or_404

router = APIRouter(tags=["runs"])
LIGHT = {"meta": {"question_set", "params"}}


class RunView(BaseModel):
    meta: RunMeta
    progress: ProgressView | None


def _view(services: Services, meta: RunMeta) -> RunView:
    return RunView(meta=meta, progress=services.jobs.progress(meta.id))


@router.get("/runs", response_model_exclude={"__all__": LIGHT})
def list_runs(services: ServicesDep, generations: str | None = None) -> list[RunView]:
    metas = services.runs.list_metas()
    wanted = set(split_ids(generations))
    if wanted:
        metas = [meta for meta in metas if set(meta.generation_ids) == wanted]
    return [_view(services, meta) for meta in metas]


@router.post("/runs", status_code=202, response_model_exclude=LIGHT)
async def create_run(request: RunRequest, services: ServicesDep, api_key: ApiKeyDep) -> RunView:
    try:
        meta, _ = await launch_run(request, api_key, services)
    except RunLaunchError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return _view(services, meta)


@router.get("/runs/{run_id}", response_model_exclude=LIGHT)
def get_run(run_id: str, services: ServicesDep) -> RunView:
    meta = load_or_404(services.runs.get, run_id, "run")
    return _view(services, meta)


@router.post("/runs/{run_id}/cancel")
def cancel_run(run_id: str, services: ServicesDep) -> CancelView:
    load_or_404(services.runs.get, run_id, "run")
    return CancelView(cancelled=services.jobs.cancel(run_id))
