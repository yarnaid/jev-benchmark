"""Tests for jev_bench.site.thresholds against the shared cases in tests/fixtures."""

import json
from pathlib import Path

import pytest

from jev_bench.site.thresholds import threshold_steps

CASES = json.loads(
    (Path(__file__).parent / "fixtures" / "threshold_steps.json").read_text(encoding="utf-8")
)


@pytest.mark.parametrize(
    ("default", "low", "step"),
    [pytest.param(case["default"], case["min"], case["step"], id=case["id"]) for case in CASES],
)
def test_threshold_steps_match_the_slider(default: float, low: int, step: int) -> None:
    assert threshold_steps(default) == list(range(low, 101, step))
