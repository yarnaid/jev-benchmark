"""Tests for jev_bench.web.views."""

import pytest
from fastapi import HTTPException

from jev_bench.web.views import load_or_404


def _load(item_id: str) -> str:
    if item_id == "missing":
        raise KeyError(item_id)
    return f"found:{item_id}"


def test_load_or_404_returns_the_loaded_value() -> None:
    assert load_or_404(_load, "present", "generation") == "found:present"


@pytest.mark.parametrize(
    ("label", "expected_detail"),
    [
        pytest.param("generation", "unknown generation 'missing'", id="generation"),
        pytest.param("run", "unknown run 'missing'", id="run"),
    ],
)
def test_load_or_404_raises_404_on_key_error(label: str, expected_detail: str) -> None:
    with pytest.raises(HTTPException) as exc_info:
        load_or_404(_load, "missing", label)
    assert exc_info.value.status_code == 404
    assert exc_info.value.detail == expected_detail
