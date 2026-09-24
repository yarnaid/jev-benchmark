"""Tests for jev_bench.classifiers.base."""

from typing import Any

import pytest
from tests.factories import EmailFactory

from jev_bench.catalog import ModelInfo
from jev_bench.classifiers.base import (
    EmailOutcome,
    Usage,
    failed_result,
    outcome_from_parsed,
    usage_from_body,
)

_PRICED = ModelInfo(id="m", name="m", prompt_price=0.001, completion_price=0.002)


@pytest.mark.parametrize(
    ("usage", "pricing", "expected"),
    [
        pytest.param(
            {"input_tokens": 476, "output_tokens": 70, "cost": 0.00002},
            None,
            Usage(input_tokens=476, output_tokens=70, cost=0.00002),
            id="decisions-style",
        ),
        pytest.param(
            {"prompt_tokens": 10, "completion_tokens": 5, "cost": 0.5},
            None,
            Usage(input_tokens=10, output_tokens=5, cost=0.5),
            id="chat-style",
        ),
        pytest.param(
            {"prompt_tokens": 10, "total_tokens": 10},
            _PRICED,
            Usage(input_tokens=10, cost=0.01, cost_estimated=True),
            id="missing-cost-estimated",
        ),
        pytest.param(None, None, Usage(cost_estimated=True), id="no-usage-no-pricing"),
        pytest.param(
            {"prompt_tokens": 3, "cost": True},
            _PRICED,
            Usage(input_tokens=3, cost=0.003, cost_estimated=True),
            id="bool-cost-ignored",
        ),
    ],
)
def test_usage_from_body(
    usage: dict[str, Any] | None, pricing: ModelInfo | None, expected: Usage
) -> None:
    result = usage_from_body(usage, pricing)
    assert result.model_copy(update={"cost": round(result.cost, 9)}) == expected


def test_usage_plus() -> None:
    total = Usage(input_tokens=1, output_tokens=2, cost=0.1).plus(
        Usage(input_tokens=3, cost=0.2, cost_estimated=True)
    )
    assert (total.input_tokens, total.output_tokens, total.cost_estimated) == (4, 2, True)
    assert total.cost == pytest.approx(0.3)


def test_failed_result_marks_every_email() -> None:
    emails = [EmailFactory(), EmailFactory()]
    result = failed_result(emails, "HTTP 500: boom")
    assert result.error == "HTTP 500: boom"
    assert {email_id: outcome.error for email_id, outcome in result.outcomes.items()} == {
        emails[0].id: "HTTP 500: boom",
        emails[1].id: "HTTP 500: boom",
    }


@pytest.mark.parametrize(
    ("answers", "notes", "expected"),
    [
        pytest.param(
            {"q": {"a": 1.0}},
            ["x: missing"],
            EmailOutcome(answers={"q": {"a": 1.0}}, notes=("x: missing",)),
            id="partial",
        ),
        pytest.param(
            {},
            ["q: missing", "x: missing"],
            EmailOutcome(
                error="q: missing; x: missing",
                notes=("q: missing", "x: missing"),
            ),
            id="nothing-parsed",
        ),
        pytest.param({}, [], EmailOutcome(error="no answers"), id="nothing-at-all"),
    ],
)
def test_outcome_from_parsed(
    answers: dict[str, dict[str, float]], notes: list[str], expected: EmailOutcome
) -> None:
    assert outcome_from_parsed(answers, notes) == expected
