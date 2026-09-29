"""Contract tests for the GitHub Actions workflows: every action is pinned to a full commit SHA."""

import re
from pathlib import Path

import pytest

WORKFLOWS = sorted((Path(__file__).parents[1] / ".github" / "workflows").glob("*.yml"))
USES = re.compile(r"^\s*(?:-\s*)?uses:\s*(\S+)", re.MULTILINE)
PINNED = re.compile(r"[\w.-]+/[\w.-]+@[0-9a-f]{40}")


def test_the_pages_workflow_exists() -> None:
    assert [path.name for path in WORKFLOWS] == ["pages.yml"]


@pytest.mark.parametrize("workflow", WORKFLOWS, ids=lambda path: path.name)
def test_actions_are_pinned_to_commit_shas(workflow: Path) -> None:
    uses = USES.findall(workflow.read_text(encoding="utf-8"))
    assert uses
    assert [action for action in uses if not PINNED.fullmatch(action)] == []
