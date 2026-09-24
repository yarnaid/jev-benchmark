"""Tests for jev_bench.failures."""

import pytest

from jev_bench.failures import failure_text
from jev_bench.openrouter import OpenRouterError


@pytest.mark.parametrize(
    ("exc", "expected"),
    [
        pytest.param(
            OpenRouterError("HTTP 401: no auth", status=401),
            "HTTP 401: no auth",
            id="openrouter-leaf",
        ),
        pytest.param(KeyError("needs_reply"), "KeyError: 'needs_reply'", id="key-error"),
        pytest.param(
            ExceptionGroup(
                "group", [OpenRouterError("HTTP 500: boom", status=500, retryable=True)]
            ),
            "HTTP 500: boom",
            id="openrouter-in-group",
        ),
        pytest.param(ValueError(""), "ValueError", id="empty-message"),
    ],
)
def test_failure_text(exc: BaseException, expected: str) -> None:
    assert failure_text(exc) == expected
