"""Request plan of the snapshot export: the GETs the static site needs, each with its file.

Classes:
    SiteRequest: one GET of the JSON API (path under /api, query) and its file in the snapshot.
Functions:
    base_requests: status, catalog, questions, the lists, and every generation, run and analysis
        detail (analyses with their sent prompts).
    threshold_requests: compare or the email list of a view at the given threshold percents (None:
        without ?threshold=, i.e. each question's own default).
    email_requests: one email detail per email id, with the view's runs.
    multi_threshold: the threshold of the first multi-label question in a compare report or an email
        list (None without one); the Benchmark and Explorer sliders start from it.
"""

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from pydantic import BaseModel, Field

from jev_bench.site.paths import ThresholdEndpoint, base_file, email_file, threshold_file
from jev_bench.site.views import View

__all__ = [
    "LISTS",
    "SiteRequest",
    "base_requests",
    "email_requests",
    "multi_threshold",
    "threshold_requests",
]

LISTS = (
    "/status",
    "/catalog",
    "/questions",
    "/generations",
    "/runs",
    "/analyses",
    "/analysis/defaults",
)


class SiteRequest(BaseModel):
    path: str
    params: dict[str, str] = Field(default_factory=dict)
    file: str


def base_requests(
    generation_ids: Iterable[str], run_ids: Iterable[str], analysis_ids: Sequence[str]
) -> list[SiteRequest]:
    details = [
        *(f"/generations/{generation_id}" for generation_id in generation_ids),
        *(f"/runs/{run_id}" for run_id in run_ids),
        *(f"/analyses/{analysis_id}" for analysis_id in analysis_ids),
        *(f"/analyses/{analysis_id}/prompts" for analysis_id in analysis_ids),
    ]
    return [SiteRequest(path=path, file=base_file(path)) for path in [*LISTS, *details]]


def _runs_param(view: View) -> dict[str, str]:
    return {"runs": ",".join(view.run_ids)} if view.run_ids else {}


def _view_params(endpoint: ThresholdEndpoint, view: View) -> dict[str, str]:
    generations = {"generations": ",".join(view.generation_ids)} if endpoint == "emails" else {}
    return {**generations, **_runs_param(view)}


def threshold_requests(
    endpoint: ThresholdEndpoint, view: View, percents: Sequence[int | None]
) -> list[SiteRequest]:
    base = _view_params(endpoint, view)
    return [
        SiteRequest(
            path=f"/{endpoint}",
            params=base if percent is None else {**base, "threshold": str(percent / 100)},
            file=threshold_file(endpoint, view.id, percent),
        )
        for percent in percents
    ]


def email_requests(view: View, email_ids: Iterable[str]) -> list[SiteRequest]:
    params = _runs_param(view)
    return [
        SiteRequest(path=f"/emails/{email_id}", params=params, file=email_file(view.id, email_id))
        for email_id in email_ids
    ]


def multi_threshold(endpoint: ThresholdEndpoint, body: Mapping[str, Any]) -> float | None:
    questions = body["questions"] if endpoint == "compare" else body["questions"]["questions"]
    return next(
        (question["threshold"] for question in questions if question["type"] == "multi"), None
    )
