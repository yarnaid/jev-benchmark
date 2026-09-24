"""Tests for jev_bench.templates."""

import pytest

from jev_bench.templates import check_template


@pytest.mark.parametrize(
    "template",
    [
        pytest.param("Q:\n$questions", id="allowed"),
        pytest.param("no placeholders", id="static"),
        pytest.param("costs $$5 per $questions", id="escaped-dollar"),
    ],
)
def test_valid_templates_are_returned(template: str) -> None:
    assert check_template(template, ("questions",)) == template


@pytest.mark.parametrize(
    "template",
    [
        pytest.param("$questions and $other", id="unknown-placeholder"),
        pytest.param("costs $5", id="invalid-dollar"),
        pytest.param("${questions", id="unclosed-brace"),
    ],
)
def test_invalid_templates_raise(template: str) -> None:
    with pytest.raises(ValueError, match="template may only use"):
        check_template(template, ("questions",))
