"""Shared pytest configuration and fixtures.

Hooks:
    pytest_collection_modifyitems: mark tests listed in tests/slow_tests.txt as `slow`.
"""

from pathlib import Path

import pytest

_SLOW_LIST = Path(__file__).with_name("slow_tests.txt")


def _slow_node_ids() -> set[str]:
    if not _SLOW_LIST.exists():
        return set()
    lines = (line.strip() for line in _SLOW_LIST.read_text(encoding="utf-8").splitlines())
    return {line for line in lines if line and not line.startswith("#")}


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    slow = _slow_node_ids()
    for item in items:
        if item.nodeid in slow:
            item.add_marker(pytest.mark.slow)
