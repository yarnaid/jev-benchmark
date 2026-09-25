"""Tests for jev_bench.log_setup."""

import sys
from collections.abc import Iterator

import pytest
from loguru import logger

from jev_bench.log_setup import configure_logging


@pytest.fixture(autouse=True)
def _restore_default_logging() -> Iterator[None]:
    try:
        yield
    finally:
        logger.remove()
        logger.add(sys.stderr)


def _post(path: str, *, api_key: str) -> None:
    raise RuntimeError("boom")


def _call_with_local_secret(secret: str) -> None:
    _post("/v1/x", api_key=secret)


def test_configure_logging_hides_frame_locals_on_exception(
    capsys: pytest.CaptureFixture[str],
) -> None:
    configure_logging()
    secret = "sk-or-v1-SUPERSECRET"
    try:
        _call_with_local_secret(secret)
    except RuntimeError as exc:
        logger.opt(exception=exc).error("job failed")
    captured = capsys.readouterr()
    assert secret not in captured.err


@pytest.mark.parametrize(
    ("debug", "expect_debug_line"),
    [
        pytest.param(False, False, id="default-is-info"),
        pytest.param(True, True, id="debug-enables-debug"),
    ],
)
def test_configure_logging_level(
    debug: bool, expect_debug_line: bool, capsys: pytest.CaptureFixture[str]
) -> None:
    configure_logging(debug=debug)
    logger.debug("debug-line")
    logger.info("info-line")
    captured = capsys.readouterr()
    assert "info-line" in captured.err
    assert ("debug-line" in captured.err) is expect_debug_line


def test_configure_logging_can_write_through_a_callable() -> None:
    lines: list[str] = []
    configure_logging(write=lines.append)
    secret = "sk-or-v1-SUPERSECRET"
    try:
        _call_with_local_secret(secret)
    except RuntimeError as exc:
        logger.opt(exception=exc).warning("item failed")
    assert len(lines) == 1
    assert "item failed" in lines[0]
    assert secret not in lines[0]


class _FakeStderr:
    def __init__(self, tty: bool) -> None:
        self._tty = tty

    def isatty(self) -> bool:
        return self._tty


@pytest.mark.parametrize(
    ("no_color", "is_tty", "expected"),
    [
        pytest.param("1", True, False, id="no-color-env-wins"),
        pytest.param(None, False, False, id="non-tty-disables-color"),
        pytest.param(None, True, True, id="tty-without-no-color-enables-color"),
    ],
)
def test_configure_logging_colorize(
    no_color: str | None,
    is_tty: bool,
    expected: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if no_color is None:
        monkeypatch.delenv("NO_COLOR", raising=False)
    else:
        monkeypatch.setenv("NO_COLOR", no_color)
    monkeypatch.setattr(sys, "stderr", _FakeStderr(is_tty))
    captured: dict[str, object] = {}
    monkeypatch.setattr(logger, "add", lambda sink, **kwargs: captured.update(kwargs))
    configure_logging()
    assert captured["colorize"] is expected
