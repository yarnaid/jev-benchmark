"""Tests for jev_bench.json_schema."""

import pytest

from jev_bench.json_schema import strict_object


@pytest.mark.parametrize(
    ("description", "expected_extra"),
    [
        pytest.param(None, {}, id="plain"),
        pytest.param("Kind?", {"description": "Kind?"}, id="described"),
    ],
)
def test_strict_object(description: str | None, expected_extra: dict[str, str]) -> None:
    properties = {"a": {"type": "string"}, "b": {"type": "number"}}
    assert strict_object(properties, description) == {
        "type": "object",
        "additionalProperties": False,
        "required": ["a", "b"],
        "properties": properties,
        **expected_extra,
    }
