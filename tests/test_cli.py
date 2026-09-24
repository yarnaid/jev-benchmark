"""Tests for jev_bench.cli."""

from typing import Any

import pytest
from typer.testing import CliRunner

from jev_bench import cli_jobs
from jev_bench.cli import app

runner = CliRunner()


def test_help_lists_commands() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for command in ("serve", "generate", "run"):
        assert command in result.output


def test_pretty_exceptions_never_show_locals() -> None:
    assert app.pretty_exceptions_show_locals is False


@pytest.mark.parametrize(
    ("host", "warned"),
    [
        pytest.param("127.0.0.1", False, id="loopback"),
        pytest.param("8.8.8.8", True, id="exposed"),
    ],
)
def test_serve_starts_uvicorn(monkeypatch: pytest.MonkeyPatch, host: str, warned: bool) -> None:
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(
        "uvicorn.run", lambda target, **kwargs: calls.append({"target": target, **kwargs})
    )
    result = runner.invoke(app, ["serve", "--host", host, "--port", "8123"])
    assert result.exit_code == 0
    assert calls == [
        {
            "target": "jev_bench.web.app:create_default_app",
            "factory": True,
            "host": host,
            "port": 8123,
            "reload": False,
        }
    ]
    assert ("plain HTTP" in result.output) is warned


def test_generate_builds_the_request(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[Any] = []

    async def fake(request: Any, **_: Any) -> int:
        seen.append(request)
        return 0

    monkeypatch.setattr(cli_jobs, "generate_and_wait", fake)
    args = [
        "generate",
        "--count",
        "5",
        "--name",
        "t",
        "--seed",
        "3",
        "--model",
        "a/b",
        "--model",
        "c/d",
    ]
    result = runner.invoke(app, args)
    assert result.exit_code == 0
    got = (seen[0].count, seen[0].name, seen[0].seed, seen[0].models)
    assert got == (5, "t", 3, ("a/b", "c/d"))


def test_run_passes_arguments_and_exit_code(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[tuple[Any, ...]] = []

    async def fake(
        column: str, generations: list[str], model: str | None, mode: str | None, **_: Any
    ) -> int:
        seen.append((column, generations, model, mode))
        return 1

    monkeypatch.setattr(cli_jobs, "run_and_wait", fake)
    args = ["run", "anthropic", "-g", "g1", "-g", "g2", "--mode", "all_in_one"]
    result = runner.invoke(app, args)
    assert result.exit_code == 1
    assert seen == [("anthropic", ["g1", "g2"], None, "all_in_one")]


@pytest.mark.parametrize(
    "args",
    [
        pytest.param(["generate", "--count", "0"], id="count-too-small"),
        pytest.param(["run", "jev"], id="missing-generation"),
    ],
)
def test_invalid_arguments(args: list[str]) -> None:
    assert runner.invoke(app, args).exit_code == 2


@pytest.mark.parametrize(
    ("args", "attr"),
    [
        pytest.param(["generate"], "generate_and_wait", id="generate"),
        pytest.param(["run", "jev", "-g", "g1"], "run_and_wait", id="run"),
    ],
)
def test_configures_logging_before_the_job(
    monkeypatch: pytest.MonkeyPatch, args: list[str], attr: str
) -> None:
    calls: list[str] = []
    monkeypatch.setattr(
        "jev_bench.log_setup.configure_logging", lambda **_: calls.append("logging")
    )

    async def fake(*_: Any, **__: Any) -> int:
        calls.append("job")
        return 0

    monkeypatch.setattr(cli_jobs, attr, fake)
    result = runner.invoke(app, args)
    assert result.exit_code == 0
    assert calls == ["logging", "job"]
