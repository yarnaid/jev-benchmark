"""Executes one generation: prompts per plan item, calls the generator mix, persists emails/totals.

Constants:
    CHECKPOINT_EVERY: persist running totals after this many completed emails.
Classes:
    GeneratorDeps: collaborators of a generation job (`api_key` is a `SecretStr`, never persisted).
Functions:
    execute_generation: job body (always finalizes the meta; re-raises only CancelledError). An
        item whose output is unusable (empty, not JSON, invalid) is retried with the same model up
        to `config.max_attempts` times; a provider error is not retried here (the client already
        retries transient ones). Every failed attempt logs one line naming the item and model.
    mark_interrupted_generations: startup sweep for generations left `running`; recomputes `done`
        and `trait_mismatches` from the emails on disk (a trait missing from an email's own
        `traits` is skipped, never counted as a mismatch), `errors`/`total_cost` stay at the last
        checkpoint.
"""

import asyncio
import time
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from typing import NamedTuple

from loguru import logger
from pydantic import SecretStr, ValidationError

from jev_bench.classifiers.base import usage_from_body
from jev_bench.emails import Email, email_id
from jev_bench.failures import failure_text, log_job_failure
from jev_bench.generation.config import GenerationConfig
from jev_bench.generation.plan import PlanItem, ResolvedTrait, resolve_traits
from jev_bench.generation.prompt import (
    count_mismatches,
    generation_schema,
    parse_generator_output,
    render_prompts,
)
from jev_bench.jobs import JobProgress, cancel_status
from jev_bench.openrouter import (
    CHAT_PATH,
    ApiResponse,
    JsonObject,
    OpenRouterClient,
    OpenRouterError,
    chat_content,
    json_schema_format,
)
from jev_bench.questions import QuestionSet
from jev_bench.store.generations import GenerationMeta, GenerationStore
from jev_bench.store.status import JobStatus

__all__ = [
    "CHECKPOINT_EVERY",
    "GeneratorDeps",
    "execute_generation",
    "mark_interrupted_generations",
]

CHECKPOINT_EVERY = 10


class GeneratorDeps(NamedTuple):
    client: OpenRouterClient
    api_key: SecretStr
    store: GenerationStore
    questions: QuestionSet
    config: GenerationConfig


class _Tally:
    def __init__(self) -> None:
        self.mismatches = 0


async def execute_generation(
    meta: GenerationMeta, plan: Sequence[PlanItem], deps: GeneratorDeps, progress: JobProgress
) -> None:
    started = time.perf_counter()
    tally = _Tally()
    try:
        await _generate_all(meta, plan, deps, progress, tally)
    except asyncio.CancelledError as exc:
        _finish(deps.store, meta, progress, tally, cancel_status(exc), None, started)
        raise
    except Exception as exc:
        text = failure_text(exc)
        log_job_failure(exc, text, generation=meta.id)
        _finish(deps.store, meta, progress, tally, "failed", text, started)
        return
    _finish(deps.store, meta, progress, tally, "completed", None, started)


async def _generate_all(
    meta: GenerationMeta,
    plan: Sequence[PlanItem],
    deps: GeneratorDeps,
    progress: JobProgress,
    tally: _Tally,
) -> None:
    traits = resolve_traits(deps.config, deps.questions)
    response_format = json_schema_format("email_generation", generation_schema(deps.questions))
    semaphore = asyncio.Semaphore(deps.config.concurrency)
    async with asyncio.TaskGroup() as group:
        for item in plan:
            group.create_task(
                _generate_one(meta, item, traits, response_format, deps, semaphore, progress, tally)
            )


async def _generate_one(
    meta: GenerationMeta,
    item: PlanItem,
    traits: Sequence[ResolvedTrait],
    response_format: JsonObject,
    deps: GeneratorDeps,
    semaphore: asyncio.Semaphore,
    progress: JobProgress,
    tally: _Tally,
) -> None:
    attempts = deps.config.max_attempts
    for attempt in range(1, attempts + 1):
        try:
            email, mismatched = await _attempt(
                meta.id, item, traits, response_format, deps, semaphore, progress
            )
        except OpenRouterError as exc:
            if exc.fatal:
                raise
            _item_failed(meta.id, item, str(exc), progress)
            return
        except ValueError as exc:
            if attempt < attempts:
                _item_retried(meta.id, item, f"attempt {attempt}/{attempts}", _brief(exc))
                continue
            _item_failed(meta.id, item, f"after {attempts} attempts: {_brief(exc)}", progress)
            return
        _store_email(meta, email, mismatched, deps.store, progress, tally)
        return


async def _attempt(
    generation_id: str,
    item: PlanItem,
    traits: Sequence[ResolvedTrait],
    response_format: JsonObject,
    deps: GeneratorDeps,
    semaphore: asyncio.Semaphore,
    progress: JobProgress,
) -> tuple[Email, int]:
    async with semaphore:
        response = await _call(item, traits, response_format, deps)
    progress.cost += usage_from_body(response.body.get("usage"), None).cost
    return _to_email(generation_id, item, traits, response, deps.questions)


async def _call(
    item: PlanItem,
    traits: Sequence[ResolvedTrait],
    response_format: JsonObject,
    deps: GeneratorDeps,
) -> ApiResponse:
    system, user = render_prompts(deps.config, traits, item, deps.questions)
    body = {
        "model": item.model,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "temperature": deps.config.temperature,
        "response_format": response_format,
        "provider": {"require_parameters": True},
    }
    return await deps.client.post_json(CHAT_PATH, body, api_key=deps.api_key.get_secret_value())


def _to_email(
    generation_id: str,
    item: PlanItem,
    traits: Sequence[ResolvedTrait],
    response: ApiResponse,
    questions: QuestionSet,
) -> tuple[Email, int]:
    output = parse_generator_output(chat_content(response.body), questions)
    email = Email(
        id=email_id(generation_id, item.index),
        sent_at=item.sent_at,
        sender=output.email.sender,
        to=output.email.to,
        cc=output.email.cc,
        subject=output.email.subject,
        body=output.email.body,
        generator_model=response.body.get("model") or item.model,
        traits=dict(item.traits),
        reference_answers=output.answers,
    )
    return email, count_mismatches(item.traits, traits, output.answers)


def _store_email(
    meta: GenerationMeta,
    email: Email,
    mismatched: int,
    store: GenerationStore,
    progress: JobProgress,
    tally: _Tally,
) -> None:
    store.append_email(meta.id, email)
    progress.done += 1
    tally.mismatches += mismatched
    if progress.done % CHECKPOINT_EVERY == 0:
        store.save(_totals(meta, progress, tally))


def _item_failed(generation_id: str, item: PlanItem, detail: str, progress: JobProgress) -> None:
    progress.errors += 1
    logger.bind(generation=generation_id, item=item.index, model=item.model).warning(
        "item {} ({}) failed {}", item.index, item.model, detail
    )


def _item_retried(generation_id: str, item: PlanItem, attempt: str, detail: str) -> None:
    logger.bind(generation=generation_id, item=item.index, model=item.model).warning(
        "item {} ({}) {} failed, retrying: {}", item.index, item.model, attempt, detail
    )


def _brief(exc: ValueError) -> str:
    if not isinstance(exc, ValidationError):
        return str(exc)
    errors = exc.errors()
    if any(error["type"] == "json_invalid" for error in errors):
        return "output is not valid JSON"
    fields = [".".join(str(part) for part in error["loc"]) or error["type"] for error in errors]
    shown = ", ".join(fields[:4]) + (", …" if len(fields) > 4 else "")
    return f"invalid output ({len(errors)} errors: {shown})"


def _totals(meta: GenerationMeta, progress: JobProgress, tally: _Tally) -> GenerationMeta:
    update = {
        "done": progress.done,
        "errors": progress.errors,
        "total_cost": progress.cost,
        "trait_mismatches": tally.mismatches,
    }
    return meta.model_copy(update=update)


def _finish(
    store: GenerationStore,
    meta: GenerationMeta,
    progress: JobProgress,
    tally: _Tally,
    status: JobStatus,
    error: str | None,
    started: float,
) -> None:
    finished = {
        "status": status,
        "error": error,
        "finished_at": datetime.now(UTC),
        "duration_s": time.perf_counter() - started,
    }
    store.save(_totals(meta, progress, tally).model_copy(update=finished))


def mark_interrupted_generations(
    store: GenerationStore, is_live: Callable[[str], bool]
) -> list[str]:
    orphaned = [
        meta for meta in store.list_metas() if meta.status == "running" and not is_live(meta.id)
    ]
    for meta in orphaned:
        store.save(meta.model_copy(update=_recovered_totals(store, meta)))
    return [meta.id for meta in orphaned]


def _recovered_totals(store: GenerationStore, meta: GenerationMeta) -> dict[str, object]:
    emails = store.emails(meta.id)
    traits = resolve_traits(meta.config, meta.question_set)
    mismatches = sum(
        count_mismatches(email.traits, traits, email.reference_answers) for email in emails
    )
    return {"status": "interrupted", "done": len(emails), "trait_mismatches": mismatches}
