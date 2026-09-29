"""Tests for jev_bench.site.plan."""

from typing import Any

import pytest

from jev_bench.site.paths import ThresholdEndpoint
from jev_bench.site.plan import (
    base_requests,
    email_requests,
    multi_threshold,
    threshold_requests,
)
from jev_bench.site.views import View

RUNS = View.of("Latest", ["g1"], ["r1", "r2"])
ALONE = View.of("Generation only", ["g1"], [])


def test_base_requests_cover_lists_and_every_detail() -> None:
    requests = base_requests(["g1"], ["r1"], ["a1"])
    assert [(request.path, request.params, request.file) for request in requests] == [
        ("/status", {}, "api/status.json"),
        ("/catalog", {}, "api/catalog.json"),
        ("/questions", {}, "api/questions.json"),
        ("/generations", {}, "api/generations.json"),
        ("/runs", {}, "api/runs.json"),
        ("/analyses", {}, "api/analyses.json"),
        ("/analysis/defaults", {}, "api/analysis/defaults.json"),
        ("/generations/g1", {}, "api/generations/g1.json"),
        ("/runs/r1", {}, "api/runs/r1.json"),
        ("/analyses/a1", {}, "api/analyses/a1.json"),
        ("/analyses/a1/prompts", {}, "api/analyses/a1/prompts.json"),
    ]


@pytest.mark.parametrize(
    ("endpoint", "view", "percent", "params", "file"),
    [
        pytest.param(
            "compare",
            RUNS,
            None,
            {"runs": "r1,r2"},
            f"api/compare/{RUNS.id}/t-default.json",
            id="compare-default",
        ),
        pytest.param(
            "compare",
            RUNS,
            55,
            {"runs": "r1,r2", "threshold": "0.55"},
            f"api/compare/{RUNS.id}/t55.json",
            id="compare-step",
        ),
        pytest.param(
            "emails",
            RUNS,
            100,
            {"generations": "g1", "runs": "r1,r2", "threshold": "1.0"},
            f"api/emails/{RUNS.id}/t100.json",
            id="emails-step",
        ),
        pytest.param(
            "emails",
            ALONE,
            None,
            {"generations": "g1"},
            f"api/emails/{ALONE.id}/t-default.json",
            id="emails-without-runs",
        ),
    ],
)
def test_threshold_requests(
    endpoint: ThresholdEndpoint, view: View, percent: int | None, params: dict[str, str], file: str
) -> None:
    (request,) = threshold_requests(endpoint, view, [percent])
    assert (request.path, request.params, request.file) == (f"/{endpoint}", params, file)


@pytest.mark.parametrize(
    ("view", "params"),
    [
        pytest.param(RUNS, {"runs": "r1,r2"}, id="with-runs"),
        pytest.param(ALONE, {}, id="without-runs"),
    ],
)
def test_email_requests(view: View, params: dict[str, str]) -> None:
    (request,) = email_requests(view, ["g1.0001"])
    expected = ("/emails/g1.0001", params, f"api/email/{view.id}/g1.0001.json")
    assert (request.path, request.params, request.file) == expected


MULTI = {"type": "multi", "threshold": 0.75}
CHOICE = {"type": "choice"}


@pytest.mark.parametrize(
    ("endpoint", "body", "expected"),
    [
        pytest.param("compare", {"questions": [CHOICE, MULTI]}, 0.75, id="compare-report"),
        pytest.param(
            "emails", {"questions": {"questions": [CHOICE, MULTI]}}, 0.75, id="email-list"
        ),
        pytest.param("compare", {"questions": [CHOICE]}, None, id="no-multi-question"),
    ],
)
def test_multi_threshold(
    endpoint: ThresholdEndpoint, body: dict[str, Any], expected: float | None
) -> None:
    assert multi_threshold(endpoint, body) == expected
