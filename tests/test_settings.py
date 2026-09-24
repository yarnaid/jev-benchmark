"""Tests for jev_bench.settings."""

from pathlib import Path

import pytest
from pydantic import SecretStr

from jev_bench.settings import Settings, load_settings

_ENV_NAMES = (
    "OPENROUTER_API_KEY",
    "JEV_BENCH_DATA_DIR",
    "JEV_BENCH_CONFIG_DIR",
    "JEV_BENCH_MAX_RETRIES",
    "JEV_BENCH_REQUEST_TIMEOUT_S",
)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in _ENV_NAMES:
        monkeypatch.delenv(name, raising=False)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        pytest.param(None, None, id="unset"),
        pytest.param("", None, id="empty"),
        pytest.param("   ", None, id="blank"),
        pytest.param("sk-or-v1-abc", "sk-or-v1-abc", id="set"),
    ],
)
def test_server_api_key(
    monkeypatch: pytest.MonkeyPatch, raw: str | None, expected: str | None
) -> None:
    if raw is not None:
        monkeypatch.setenv("OPENROUTER_API_KEY", raw)
    assert Settings(_env_file=None).server_api_key() == expected


def test_prefixed_env_overrides(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("JEV_BENCH_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("JEV_BENCH_MAX_RETRIES", "5")
    settings = Settings(_env_file=None)
    assert settings.data_dir == tmp_path
    assert settings.max_retries == 5


def test_defaults() -> None:
    settings = Settings(_env_file=None)
    assert settings.openrouter_base_url == "https://openrouter.ai/api"
    assert settings.data_dir == Path("data")
    assert settings.config_dir == Path("config")
    assert settings.request_timeout_s == 60.0
    assert settings.max_retries == 3


def test_env_file_is_read_and_unrelated_keys_ignored(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("OPENROUTER_API_KEY=from-file\nUNRELATED=1\n", encoding="utf-8")
    assert Settings(_env_file=env_file).server_api_key() == "from-file"


def test_key_accepted_by_field_name() -> None:
    settings = Settings(_env_file=None, openrouter_api_key=SecretStr("by-name"))
    assert settings.server_api_key() == "by-name"


def test_secret_never_in_repr(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-v1-secret")
    assert "sk-or-v1-secret" not in repr(load_settings())
