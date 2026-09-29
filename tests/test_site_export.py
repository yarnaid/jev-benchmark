"""Tests for jev_bench.site.export on a mini data dir (one email, a Jev and a Kev run)."""

import json
from collections.abc import Callable
from datetime import UTC, datetime
from functools import partial
from pathlib import Path

import httpx2
import pytest
from tests.factories import ServicesFactory, seed_generation

from jev_bench.benchmark_config import JevParams
from jev_bench.compare import compare
from jev_bench.emails import Email
from jev_bench.questions import Distribution, QuestionSet, hard_distribution
from jev_bench.services import Services
from jev_bench.site import export as export_module
from jev_bench.site.errors import ExportError
from jev_bench.site.export import export_site
from jev_bench.site.static_copy import STATIC_MODULE
from jev_bench.site.views import View
from jev_bench.store.runs import Prediction, RunMeta
from jev_bench.store.status import JobStatus

GENERATION = "20260924-100000-seed-abcd"
JEV = "20260924-110000-jev-0001"
KEV = "20260924-110500-kev-0002"
SENTINEL = "sk-or-v1-SENTINEL-export"
LISTS = ("status", "catalog", "questions", "generations", "runs", "analyses", "analysis/defaults")


def _refuse(request: httpx2.Request) -> httpx2.Response:
    raise AssertionError(f"unexpected request {request.url}")


def _answers(questions: QuestionSet, email: Email) -> dict[str, Distribution]:
    pairs = (
        (question.id, hard_distribution(question, email.reference_answers.get(question.id)))
        for question in questions.questions
    )
    return {question_id: dist for question_id, dist in pairs if dist is not None}


def _seed_run(
    services: Services, run_id: str, column: str, status: JobStatus = "completed"
) -> None:
    questions = services.question_set()
    emails = services.generations.emails(GENERATION)
    meta = RunMeta(
        id=run_id,
        column=column,
        kind="decisions",
        model="m",
        generation_ids=(GENERATION,),
        mode="per_email",
        emails_per_request=1,
        question_set=questions,
        params=JevParams(),
        concurrency=1,
        created_at=datetime(2026, 9, 24, tzinfo=UTC),
        n_emails=len(emails),
        n_done=len(emails),
        status=status,
    )
    services.runs.save(meta)
    predictions = [Prediction(email_id=e.id, answers=_answers(questions, e)) for e in emails]
    services.runs.append_predictions(run_id, predictions)


def _seeded(services: Services) -> Services:
    seed_generation(services, emails=1, generation_id=GENERATION)
    _seed_run(services, JEV, "jev")
    _seed_run(services, KEV, "kev")
    return services


def _fake_ui(root: Path) -> Path:
    ui = root / "ui"
    (ui / "js").mkdir(parents=True)
    page = "<!doctype html><html><head><title>x</title></head></html>"
    (ui / "index.html").write_text(page, encoding="utf-8")
    (ui / "js" / "deployment.js").write_text("export const STATIC = false;\n", encoding="utf-8")
    return ui


def _view_files(view: View, endpoints: tuple[str, ...]) -> set[str]:
    thresholds = {
        f"api/{endpoint}/{view.id}/{name}.json"
        for endpoint in endpoints
        for name in ("t-default", "t100")
    }
    return {*thresholds, f"api/email/{view.id}/{GENERATION}.0001.json"}


def _expected_files() -> set[str]:
    lists = {f"api/{name}.json" for name in LISTS}
    details = {f"api/generations/{GENERATION}.json", f"api/runs/{JEV}.json", f"api/runs/{KEV}.json"}
    views = [View.of("", [GENERATION], [JEV]), View.of("", [GENERATION], [JEV, KEV])]
    per_view = set().union(*(_view_files(view, ("compare", "emails")) for view in views))
    alone = _view_files(View.of("", [GENERATION], []), ("emails",))
    return {"index.html", "js/deployment.js", "api/site.json", *lists, *details, *per_view, *alone}


@pytest.fixture(autouse=True)
def _small_export(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(export_module, "threshold_steps", lambda default: [100])
    monkeypatch.setattr("jev_bench.web.routes.compare.compare", partial(compare, resamples=20))


def _files(out: Path) -> set[str]:
    return {path.relative_to(out).as_posix() for path in out.rglob("*") if path.is_file()}


def _read_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


async def test_export_writes_every_planned_file_and_no_key(
    make_services: ServicesFactory, tmp_path: Path
) -> None:
    services = _seeded(make_services(_refuse, api_key=SENTINEL))
    out = tmp_path / "site"
    summary = await export_site(
        services.settings, out, commit="abc1234def", static_dir=_fake_ui(tmp_path)
    )
    assert _files(out) == _expected_files()
    assert summary.files == len(_expected_files())
    manifest = _read_json(out / "api" / "site.json")
    assert isinstance(manifest, dict)
    assert manifest["commit"] == "abc1234def"
    labels = [view["label"] for view in manifest["views"]]
    assert labels == ["Latest without kev", "Latest without embeddings", "Generation only"]
    assert _read_json(out / "api" / "status.json") == {"server_key": False}
    assert (out / "js" / "deployment.js").read_text(encoding="utf-8") == STATIC_MODULE
    leaks = [p for p in out.rglob("*") if p.is_file() and SENTINEL.encode() in p.read_bytes()]
    assert leaks == []


def _running_job(services: Services, out: Path) -> None:
    _seed_run(services, "20260924-120000-jev-0003", "jev", status="running")


def _stale_output(services: Services, out: Path) -> None:
    out.mkdir()
    (out / "old.txt").write_text("x", encoding="utf-8")


@pytest.mark.parametrize(
    ("prepare", "message"),
    [
        pytest.param(_running_job, "still running", id="running-job"),
        pytest.param(_stale_output, "not an empty directory", id="non-empty-output"),
    ],
)
async def test_export_refuses(
    make_services: ServicesFactory,
    tmp_path: Path,
    prepare: Callable[[Services, Path], None],
    message: str,
) -> None:
    services = _seeded(make_services(_refuse))
    out = tmp_path / "site"
    prepare(services, out)
    with pytest.raises(ExportError, match=message):
        await export_site(services.settings, out, static_dir=_fake_ui(tmp_path))
