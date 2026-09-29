"""Tests for jev_bench.site.paths against the shared cases in tests/fixtures/site_paths.json."""

import json
from pathlib import Path
from typing import Any

import pytest

from jev_bench.site.paths import base_file, email_file, threshold_file

FIXTURE = json.loads(
    (Path(__file__).parent / "fixtures" / "site_paths.json").read_text(encoding="utf-8")
)


def _python_file(case: dict[str, Any]) -> str:
    if "base" in case:
        return base_file(case["base"])
    if "email" in case:
        return email_file(case["view"], case["email"])
    return threshold_file(case["endpoint"], case["view"], case["percent"])


@pytest.mark.parametrize(
    "case", [pytest.param(case, id=case["id"]) for case in FIXTURE["published"]]
)
def test_files_match_the_shared_cases(case: dict[str, Any]) -> None:
    assert _python_file(case) == case["file"]
