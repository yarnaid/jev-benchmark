"""Async bodies of the CLI commands: build services, launch a job, render live progress,
report the outcome.

Constants:
    NO_KEY
Functions:
    generate_and_wait: run one generation to completion; returns the exit code.
    run_and_wait: run one benchmark column to completion; returns the exit code.
"""

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import httpx2
from pydantic import ValidationError
from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn

from jev_bench.generation.launcher import GenerationRequest, launch_generation
from jev_bench.openrouter import build_http_client
from jev_bench.run_launcher import RunLaunchError, RunRequest, launch_run
from jev_bench.services import Services
from jev_bench.settings import Settings, load_settings

__all__ = [
    "NO_KEY",
    "console",
    "generate_and_wait",
    "run_and_wait",
]

NO_KEY = "OPENROUTER_API_KEY is not set (environment or .env)."
console = Console(stderr=True)


async def generate_and_wait(
    request: GenerationRequest,
    *,
    settings: Settings | None = None,
    http: httpx2.AsyncClient | None = None,
) -> int:
    async with _services(settings, http) as services:
        key = services.api_key(None)
        if key is None:
            return _fail(NO_KEY)
        try:
            meta, task = await launch_generation(request, key, services)
        except ValueError as exc:
            return _fail(str(exc))
        await _watch(services, meta.id, task, f"generation {meta.name}")
        final = services.generations.get(meta.id)
        summary = (
            f"generation {final.id}: {final.done}/{final.requested} emails, "
            f"{final.errors} errors, ${final.total_cost:.4f}"
        )
        return _report(final.status, summary, final.error)


async def run_and_wait(
    column: str,
    generations: list[str],
    model: str | None,
    mode: str | None,
    *,
    settings: Settings | None = None,
    http: httpx2.AsyncClient | None = None,
) -> int:
    try:
        request = RunRequest.model_validate(
            {"column": column, "model": model, "generation_ids": generations, "mode": mode}
        )
    except ValidationError as exc:
        return _fail(f"invalid run request: {exc.errors()[0]['msg']}")
    async with _services(settings, http) as services:
        key = services.api_key(None)
        if key is None:
            return _fail(NO_KEY)
        try:
            meta, task = await launch_run(request, key, services)
        except RunLaunchError as exc:
            return _fail(str(exc))
        await _watch(services, meta.id, task, f"{meta.column} · {meta.model}")
        final = services.runs.get(meta.id)
        summary = (
            f"run {final.id}: {final.n_done}/{final.n_emails} emails, {final.n_errors} errors, "
            f"${final.total_cost:.4f}, {final.duration_s or 0:.1f} s"
        )
        return _report(final.status, summary, final.error)


@asynccontextmanager
async def _services(
    settings: Settings | None, http: httpx2.AsyncClient | None
) -> AsyncIterator[Services]:
    resolved = settings or load_settings()
    client = http or build_http_client(resolved)
    try:
        yield Services(resolved, client)
    finally:
        if http is None:
            await client.aclose()


async def _watch(services: Services, job_id: str, task: asyncio.Task[None], label: str) -> None:
    columns = (
        TextColumn("[bold]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("{task.fields[cost]}"),
        TimeElapsedColumn(),
    )
    with Progress(*columns, console=console, transient=True) as progress:
        bar = progress.add_task(label, total=None, cost="$0.0000")
        while not task.done():
            view = services.jobs.progress(job_id)
            if view is not None:
                progress.update(
                    bar, total=view.total, completed=view.done, cost=f"${view.cost:.4f}"
                )
            await asyncio.wait({task}, timeout=0.5)
    await task


def _report(status: str, summary: str, error: str | None) -> int:
    if status == "completed":
        console.print(f"[green]✓[/] {summary}")
        return 0
    detail = f" — {error}" if error else ""
    console.print(f"[red]✗ {status}:[/] {summary}{detail}")
    return 1


def _fail(message: str) -> int:
    console.print(f"[red]error:[/] {message}")
    return 2
