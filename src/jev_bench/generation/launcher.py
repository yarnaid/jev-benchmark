"""Validates a generation request, builds its plan and meta, and starts it as a background job.

Classes:
    GenerationRequest: what the UI / CLI asks for.
Functions:
    launch_generation: persist the initial meta and start the job; returns (meta, task).
"""

import asyncio
import secrets
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field

from jev_bench.generation.config import GenerationConfig
from jev_bench.generation.generator import GeneratorDeps, execute_generation
from jev_bench.generation.plan import build_plan
from jev_bench.ids import new_id
from jev_bench.questions import QuestionSet
from jev_bench.store.generations import GenerationMeta

if TYPE_CHECKING:
    from jev_bench.services import Services


class GenerationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(default="generation", min_length=1, max_length=80)
    count: int = Field(ge=1, le=2000)
    seed: int | None = Field(default=None, ge=0)
    models: tuple[str, ...] | None = None


async def launch_generation(
    request: GenerationRequest, api_key: str, services: Services, *, now: datetime | None = None
) -> tuple[GenerationMeta, asyncio.Task[None]]:
    config = services.generation_config()
    questions = services.question_set()
    models = request.models or config.models
    seed = request.seed if request.seed is not None else secrets.randbelow(2**31)
    started = now or datetime.now(UTC)
    plan = build_plan(config, questions, count=request.count, seed=seed, models=models, now=started)
    meta = _new_meta(request, config, questions, seed, models, started)
    services.generations.save(meta)
    deps = GeneratorDeps(
        client=services.client,
        api_key=api_key,
        store=services.generations,
        questions=questions,
        config=config,
    )
    task = services.jobs.start(
        meta.id, request.count, lambda progress: execute_generation(meta, plan, deps, progress)
    )
    return meta, task


def _new_meta(
    request: GenerationRequest,
    config: GenerationConfig,
    questions: QuestionSet,
    seed: int,
    models: Sequence[str],
    started: datetime,
) -> GenerationMeta:
    return GenerationMeta(
        id=new_id(request.name, started),
        name=request.name,
        created_at=started,
        requested=request.count,
        seed=seed,
        models=tuple(models),
        question_set=questions,
        config=config,
    )
