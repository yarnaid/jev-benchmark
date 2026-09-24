"""Shared pytest configuration and fixtures.

Pre-warms numpy.random at collection time so Hypothesis's lazy import never lands
inside a timed test.

Fixtures:
    questions: a three-question set covering every question type.
Hooks:
    pytest_configure: pre-warm numpy.random module.
    pytest_collection_modifyitems: mark tests listed in tests/slow_tests.txt as `slow`.
"""

import importlib
from pathlib import Path

import pytest

from jev_bench.questions import ChoiceQuestion, NoulQuestion, QuestionSet, ScoreQuestion

_SLOW_LIST = Path(__file__).with_name("slow_tests.txt")


def _slow_node_ids() -> set[str]:
    if not _SLOW_LIST.exists():
        return set()
    lines = (line.strip() for line in _SLOW_LIST.read_text(encoding="utf-8").splitlines())
    return {line for line in lines if line and not line.startswith("#")}


def pytest_configure(config: pytest.Config) -> None:
    importlib.import_module("numpy.random")


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    slow = _slow_node_ids()
    for item in items:
        if item.nodeid in slow:
            item.add_marker(pytest.mark.slow)


@pytest.fixture
def questions() -> QuestionSet:
    return QuestionSet(
        name="mini",
        questions=(
            ChoiceQuestion(
                type="choice",
                id="category",
                instructions="What kind of email?",
                options={"spam": "Junk", "personal": "From a friend", "work": "From a colleague"},
            ),
            ScoreQuestion(
                type="score",
                id="urgency",
                instructions="How urgent?",
                options={"low": "Whenever", "today": "Within a day", "now": "Immediately"},
            ),
            NoulQuestion(
                type="noul",
                id="needs_reply",
                instructions="Needs a reply?",
                options={"yes": "Reply expected", "no": "No reply expected"},
            ),
        ),
    )
