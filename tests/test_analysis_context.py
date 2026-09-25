"""Tests for jev_bench.analysis.context."""

import pytest
from tests.analysis_data import GENERATION, build_source

from jev_bench.analysis.config import PLACEHOLDERS
from jev_bench.analysis.context import prompt_inputs, render_prompts
from jev_bench.questions import QuestionSet


def test_every_placeholder_has_a_value(multi_questions: QuestionSet) -> None:
    inputs = prompt_inputs(build_source(multi_questions), max_disputed=1)
    assert tuple(inputs.values) == PLACEHOLDERS
    assert inputs.email_refs == {f"e00{i}": f"{GENERATION}.000{i}" for i in (1, 2, 3)}
    assert inputs.n_disputed == 1


def test_values_never_name_emails_or_generators(multi_questions: QuestionSet) -> None:
    values = prompt_inputs(build_source(multi_questions), max_disputed=12).values
    leaked = [name for name, text in values.items() if f"{GENERATION}." in text or "gen/" in text]
    assert leaked == []


@pytest.mark.parametrize(
    ("system", "user", "expected"),
    [
        pytest.param("S $report", "U", ("S R", "U"), id="system-placeholder"),
        pytest.param("S", "U ${emails}!", ("S", "U E!"), id="braced"),
        pytest.param("costs $$5", "U", ("costs $5", "U"), id="escaped-dollar"),
    ],
)
def test_render_prompts(system: str, user: str, expected: tuple[str, str]) -> None:
    values = dict.fromkeys(PLACEHOLDERS, "") | {"report": "R", "emails": "E"}
    assert render_prompts(system, user, values) == expected


def test_values_with_dollars_are_not_re_expanded() -> None:
    values = dict.fromkeys(PLACEHOLDERS, "") | {"emails": "body mentions $report"}
    assert render_prompts("$emails", "", values) == ("body mentions $report", "")
