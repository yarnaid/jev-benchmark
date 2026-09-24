"""Tests for jev_bench.cli_jobs."""

from pathlib import Path

import httpx2
import pytest
from tests.factories import FakeOpenRouter, mini_settings, seed_generation

from jev_bench.cli_jobs import generate_and_wait, run_and_wait
from jev_bench.generation.launcher import GenerationRequest
from jev_bench.services import Services


def _http(handler: FakeOpenRouter) -> httpx2.AsyncClient:
    return httpx2.AsyncClient(
        base_url="https://openrouter.test/api", transport=httpx2.MockTransport(handler)
    )


@pytest.fixture(autouse=True)
def _no_env_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)


async def test_generate_and_wait_completes(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    http = _http(FakeOpenRouter())
    code = await generate_and_wait(
        GenerationRequest(count=2), settings=mini_settings(tmp_path, "sk-test"), http=http
    )
    assert code == 0
    assert "2/2 emails" in capsys.readouterr().err
    await http.aclose()


async def test_generate_without_key_is_exit_2(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    http = _http(FakeOpenRouter())
    assert (
        await generate_and_wait(
            GenerationRequest(count=1), settings=mini_settings(tmp_path), http=http
        )
        == 2
    )
    assert "OPENROUTER_API_KEY" in capsys.readouterr().err
    await http.aclose()


async def test_generate_launch_error_is_exit_2(tmp_path: Path) -> None:
    settings = mini_settings(tmp_path, "sk-test")
    config = settings.config_dir / "generation.toml"
    config.write_text(
        config.read_text(encoding="utf-8").replace('question = "category"', 'question = "missing"'),
        encoding="utf-8",
    )
    http = _http(FakeOpenRouter())
    assert await generate_and_wait(GenerationRequest(count=1), settings=settings, http=http) == 2
    await http.aclose()


@pytest.mark.parametrize(
    ("column", "mode", "api_key", "expected"),
    [
        pytest.param("jev", None, "sk-test", 0, id="completed"),
        pytest.param("anthropic", "all_in_one", "sk-test", 0, id="chat-all-in-one"),
        pytest.param("jev", "batched", "sk-test", 2, id="invalid-mode"),
        pytest.param("missing", None, "sk-test", 2, id="unknown-column"),
        pytest.param("jev", None, None, 2, id="no-key"),
    ],
)
async def test_run_and_wait(
    tmp_path: Path, column: str, mode: str | None, api_key: str | None, expected: int
) -> None:
    settings = mini_settings(tmp_path, api_key)
    http = _http(FakeOpenRouter())
    generation_id = seed_generation(Services(settings, http))
    assert (
        await run_and_wait(column, [generation_id], None, mode, settings=settings, http=http)
        == expected
    )
    await http.aclose()


async def test_failed_run_is_exit_1(tmp_path: Path) -> None:
    settings = mini_settings(tmp_path, "sk-test")

    def unauthorized(request: httpx2.Request) -> httpx2.Response:
        if request.url.path.endswith("/v1/models"):
            return FakeOpenRouter()(request)
        return httpx2.Response(401, json={"error": {"message": "no auth"}})

    http = httpx2.AsyncClient(
        base_url="https://openrouter.test/api", transport=httpx2.MockTransport(unauthorized)
    )
    generation_id = seed_generation(Services(settings, http))
    assert await run_and_wait("jev", [generation_id], None, None, settings=settings, http=http) == 1
    await http.aclose()
