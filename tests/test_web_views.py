"""Tests for jev_bench.web.views."""

import pytest
from fastapi import HTTPException

from jev_bench.web.views import load_or_404


def _load(item_id: str) -> str:
    if item_id == "missing":
        raise KeyError(item_id)
    if item_id == "corrupt":
        raise ValueError("invalid JSON")
    return f"found:{item_id}"


def test_load_or_404_returns_the_loaded_value() -> None:
    assert load_or_404(_load, "present", "generation") == "found:present"


@pytest.mark.parametrize(
    ("item_id", "label", "expected_detail"),
    [
        pytest.param("missing", "generation", "unknown generation 'missing'", id="generation"),
        pytest.param("missing", "run", "unknown run 'missing'", id="run"),
        pytest.param(
            "corrupt", "generation", "unknown generation 'corrupt'", id="corrupt-meta-404"
        ),
    ],
)
def test_load_or_404_raises_404_on_key_error(
    item_id: str, label: str, expected_detail: str
) -> None:
    with pytest.raises(HTTPException) as exc_info:
        load_or_404(_load, item_id, label)
    assert exc_info.value.status_code == 404
    assert exc_info.value.detail == expected_detail
