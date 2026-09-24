"""Generation routes: list, create (background job), read with live progress, cancel.

Constants:
    LIGHT: response fields excluded from generation views.
Classes:
    GenerationView: meta + live progress.
Functions:
    list_generations, create_generation, get_generation, cancel_generation: route handlers.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from jev_bench.generation.launcher import GenerationRequest, launch_generation
from jev_bench.jobs import ProgressView
from jev_bench.services import Services
from jev_bench.store.generations import GenerationMeta
from jev_bench.web.deps import ApiKeyDep, ServicesDep
from jev_bench.web.views import CancelView, load_or_404

router = APIRouter(tags=["generations"])
LIGHT = {"meta": {"question_set", "config"}}


class GenerationView(BaseModel):
    meta: GenerationMeta
    progress: ProgressView | None


def _view(services: Services, meta: GenerationMeta) -> GenerationView:
    return GenerationView(meta=meta, progress=services.jobs.progress(meta.id))


@router.get("/generations", response_model_exclude={"__all__": LIGHT})
def list_generations(services: ServicesDep) -> list[GenerationView]:
    return [_view(services, meta) for meta in services.generations.list_metas()]


@router.post("/generations", status_code=202, response_model_exclude=LIGHT)
async def create_generation(
    request: GenerationRequest, services: ServicesDep, api_key: ApiKeyDep
) -> GenerationView:
    try:
        meta, _ = await launch_generation(request, api_key, services)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return _view(services, meta)


@router.get("/generations/{generation_id}", response_model_exclude=LIGHT)
def get_generation(generation_id: str, services: ServicesDep) -> GenerationView:
    meta = load_or_404(services.generations.get, generation_id, "generation")
    return _view(services, meta)


@router.post("/generations/{generation_id}/cancel")
def cancel_generation(generation_id: str, services: ServicesDep) -> CancelView:
    load_or_404(services.generations.get, generation_id, "generation")
    return CancelView(cancelled=services.jobs.cancel(generation_id))
