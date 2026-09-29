"""Snapshot export: run every planned GET against the real app in-process and write the static site.

The app is built without its lifespan (so `sweep_interrupted` never runs and `data/` is only read),
over offline services (no network, no key, no OpenRouter model list); bodies are written unchanged.

Classes:
    ExportSummary: files written, their total size and the duration.
    SiteManifest: api/site.json, the build time, the commit and the published views.
Functions:
    export_site: refuse a non-empty output or a running job, copy the UI in static mode, write
        every planned file and the manifest.
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
    metas = [
        *services.generations.list_metas(),
        *services.runs.list_metas(),
        *services.analyses.list_metas(),
    ]
    running = [meta.id for meta in metas if meta.status == "running"]
    if running:
        detail = ", ".join(running)
        raise ExportError(f"jobs still running (finish or cancel them first): {detail}")


def _views(services: Services) -> list[View]:
    generation_ids = [meta.id for meta in services.generations.list_metas()]
    columns = services.benchmark_config().columns
    runs, analyses = services.runs.list_metas(), services.analyses.list_metas()
    return derive_views(columns, generation_ids, runs, analyses)


def _app_client(services: Services, http: httpx2.AsyncClient) -> httpx2.AsyncClient:
    app = create_app(services.settings, http=http)
    app.state.services = services
    transport = httpx2.ASGITransport(app=app)
    return httpx2.AsyncClient(transport=transport, base_url="http://snapshot.invalid")


async def _write_api(writer: SiteWriter, services: Services, views: list[View]) -> None:
    generations = [meta.id for meta in services.generations.list_metas()]
    runs = [meta.id for meta in services.runs.list_metas()]
    analyses = [meta.id for meta in services.analyses.list_metas()]
    for request in base_requests(generations, runs, analyses):
        await writer.write(request)
    for view in views:
        await _write_view(writer, services, view)


async def _write_view(writer: SiteWriter, services: Services, view: View) -> None:
    with_runs: tuple[ThresholdEndpoint, ...] = ("compare", "emails")
    endpoints: tuple[ThresholdEndpoint, ...] = with_runs if view.run_ids else ("emails",)
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
    size = sum(path.stat().st_size for path in files)
    return ExportSummary(files=len(files), bytes=size, seconds=seconds)
