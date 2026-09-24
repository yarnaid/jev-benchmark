### Task 1: Project scaffold, settings and ids

**Files:**
- Create: `pyproject.toml`, `.gitignore`, `.env.example`
- Create: `src/jev_bench/__init__.py`, `src/jev_bench/settings.py`, `src/jev_bench/ids.py`
- Create: `tests/conftest.py`, `tests/slow_tests.txt`
- Test: `tests/test_settings.py`, `tests/test_ids.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `jev_bench.settings.Settings` (fields: `openrouter_api_key: SecretStr | None`,
    `openrouter_base_url: str`, `data_dir: Path`, `config_dir: Path`, `request_timeout_s: float`,
    `connect_timeout_s: float`, `max_retries: int`, `retry_base_delay_s: float`; method
    `server_api_key() -> str | None`);
  - `load_settings() -> Settings`;
  - `jev_bench.ids.slugify(text: str) -> str`;
  - `new_id(label: str, now: datetime, suffix: str | None = None) -> str`;
  - `is_safe_id(value: str) -> bool`;
  - `split_email_id(email_id: str) -> tuple[str, int]` (raises `KeyError`).

- [ ] **Step 1: Write `pyproject.toml`**

```toml
[project]
name = "jev-bench"
version = "0.1.0"
description = "Benchmark Jev against chat LLMs and an embedding baseline on email triage via OpenRouter."
requires-python = ">=3.14"
dependencies = []

[project.scripts]
jev-bench = "jev_bench.cli:app"

[build-system]
requires = ["uv_build>=0.9.5,<0.10.0"]
build-backend = "uv_build"

[tool.ruff]
line-length = 100
target-version = "py314"
src = ["src", "tests"]

[tool.ruff.lint]
select = ["E", "F", "W", "I", "B", "UP", "ANN", "SIM", "RUF", "PT", "ASYNC", "S", "C4"]
ignore = ["ANN401", "S311"]

[tool.ruff.lint.per-file-ignores]
"tests/**" = ["S101", "S105", "S106", "ANN"]

[tool.pyright]
pythonVersion = "3.14"
typeCheckingMode = "standard"
include = ["src", "tests"]
venvPath = "."
venv = ".venv"

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["."]
addopts = "-q -m 'not integration and not slow'"
asyncio_mode = "auto"
asyncio_default_fixture_loop_scope = "function"
timeout = 1
markers = [
  "integration: calls the real OpenRouter API (needs OPENROUTER_API_KEY)",
  "slow: listed in tests/slow_tests.txt",
]

[tool.coverage.run]
source = ["jev_bench"]
branch = true

[tool.coverage.report]
show_missing = true
skip_covered = true
exclude_also = ["if TYPE_CHECKING:", "raise AssertionError", "\\.\\.\\."]
```

- [ ] **Step 2: Add dependencies**

Run:
```bash
uv add fastapi uvicorn httpx2 pydantic pydantic-settings typer loguru rich numpy
uv add --dev pytest pytest-asyncio pytest-cov pytest-timeout hypothesis factory-boy ruff pyright
```
Expected: `pyproject.toml` `dependencies` / `[dependency-groups] dev` filled with pinned lower bounds; `uv.lock` and `.venv/` created.

- [ ] **Step 3: Write `.gitignore` and `.env.example`**

`.gitignore`:
```gitignore
.venv/
__pycache__/
*.py[cod]
.pytest_cache/
.ruff_cache/
.coverage
htmlcov/
.env
data/embeddings/
*.tmp
.DS_Store
```

`.env.example`:
```dotenv
# Copy to .env. Leave OPENROUTER_API_KEY empty to enter a key in the web UI instead.
OPENROUTER_API_KEY=
# JEV_BENCH_DATA_DIR=data
# JEV_BENCH_CONFIG_DIR=config
# JEV_BENCH_REQUEST_TIMEOUT_S=60
# JEV_BENCH_MAX_RETRIES=3
```

- [ ] **Step 4: Write test infrastructure**

`tests/slow_tests.txt`:
```text
# Node ids of tests slower than 20 ms; regenerate with ~/.claude/scripts/test-profile.sh
```

`tests/conftest.py`:
```python
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
```

- [ ] **Step 5: Write the failing tests**

`tests/test_settings.py`:
```python
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
def test_server_api_key(monkeypatch: pytest.MonkeyPatch, raw: str | None, expected: str | None) -> None:
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
```

`tests/test_ids.py`:
```python
"""Tests for jev_bench.ids."""

import re
from datetime import UTC, datetime, timedelta, timezone

import pytest

from jev_bench.ids import is_safe_id, new_id, slugify, split_email_id

_NOW = datetime(2026, 9, 24, 15, 30, 12, tzinfo=UTC)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        pytest.param("Anthropic/Claude Sonnet 5", "anthropic-claude-sonnet-5", id="model-id"),
        pytest.param("  --Hello__World--  ", "hello-world", id="trims-separators"),
        pytest.param("Привет", "x", id="non-ascii-only"),
        pytest.param("a" * 60, "a" * 40, id="truncates"),
        pytest.param("", "x", id="empty"),
    ],
)
def test_slugify(text: str, expected: str) -> None:
    assert slugify(text) == expected


@pytest.mark.parametrize(
    ("now", "expected"),
    [
        pytest.param(_NOW, "20260924-153012-spam-test-ab12", id="utc"),
        pytest.param(
            datetime(2026, 9, 24, 19, 30, 12, tzinfo=timezone(timedelta(hours=4))),
            "20260924-153012-spam-test-ab12",
            id="converted-to-utc",
        ),
    ],
)
def test_new_id_with_suffix(now: datetime, expected: str) -> None:
    assert new_id("Spam test", now, suffix="ab12") == expected


def test_new_id_random_suffix_format() -> None:
    assert re.fullmatch(r"20260924-153012-x-[0-9a-f]{4}", new_id("x", _NOW))


@pytest.mark.parametrize(
    ("value", "safe"),
    [
        pytest.param("20260924-153012-spam-test-ab12", True, id="valid"),
        pytest.param("../../etc/passwd", False, id="traversal"),
        pytest.param("20260924-153012-a/b", False, id="slash"),
        pytest.param("20260924-153012-a b", False, id="space"),
        pytest.param("", False, id="empty"),
    ],
)
def test_is_safe_id(value: str, safe: bool) -> None:
    assert is_safe_id(value) is safe


def test_split_email_id() -> None:
    assert split_email_id("20260924-153012-gen-ab12.0042") == ("20260924-153012-gen-ab12", 42)


@pytest.mark.parametrize(
    "value",
    [
        pytest.param("20260924-153012-gen-ab12", id="no-index"),
        pytest.param("../x.0001", id="traversal"),
        pytest.param("20260924-153012-gen-ab12.01", id="short-index"),
    ],
)
def test_split_email_id_rejects(value: str) -> None:
    with pytest.raises(KeyError):
        split_email_id(value)
```

- [ ] **Step 6: Run tests to verify they fail**

Run: `uv run pytest tests/test_settings.py tests/test_ids.py -v`
Expected: collection errors `ModuleNotFoundError: No module named 'jev_bench.settings'` / `'jev_bench.ids'`.

- [ ] **Step 7: Write the implementation**

`src/jev_bench/__init__.py`:
```python
"""Jev benchmark: synthetic email generation and multi-model triage benchmarking via OpenRouter."""
```

`src/jev_bench/settings.py`:
```python
"""Runtime settings from the environment and `.env`.

Classes:
    Settings: OpenRouter key and base URL, data/config directories, HTTP budgets.
Functions:
    load_settings: read Settings from the current environment.
"""

from pathlib import Path

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="JEV_BENCH_",
        env_file=".env",
        extra="ignore",
        validate_by_name=True,
    )

    openrouter_api_key: SecretStr | None = Field(default=None, validation_alias="OPENROUTER_API_KEY")
    openrouter_base_url: str = "https://openrouter.ai/api"
    data_dir: Path = Path("data")
    config_dir: Path = Path("config")
    request_timeout_s: float = Field(default=60.0, gt=0)
    connect_timeout_s: float = Field(default=10.0, gt=0)
    max_retries: int = Field(default=3, ge=0)
    retry_base_delay_s: float = Field(default=0.5, ge=0)

    @field_validator("openrouter_api_key", mode="before")
    @classmethod
    def _blank_is_none(cls, value: object) -> object:
        return None if isinstance(value, str) and not value.strip() else value

    def server_api_key(self) -> str | None:
        return self.openrouter_api_key.get_secret_value() if self.openrouter_api_key else None


def load_settings() -> Settings:
    return Settings()
```

`src/jev_bench/ids.py`:
```python
"""Identifier helpers for generations, runs and emails.

Functions:
    slugify: lowercase ASCII slug (at most 40 characters, never empty).
    new_id: `YYYYMMDD-HHMMSS-<slug>-<4hex>` from a timestamp converted to UTC.
    is_safe_id: whether a generation/run id is well-formed (safe as a path segment).
    split_email_id: `<generation_id>.<index>` -> (generation_id, index); KeyError when malformed.
"""

import re
import secrets
from datetime import UTC, datetime

_SAFE_ID = re.compile(r"\d{8}-\d{6}-[a-z0-9-]+")
_EMAIL_ID = re.compile(r"(?P<generation>\d{8}-\d{6}-[a-z0-9-]+)\.(?P<index>\d{4,})")


def slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:40].strip("-")
    return slug or "x"


def new_id(label: str, now: datetime, suffix: str | None = None) -> str:
    stamp = now.astimezone(UTC).strftime("%Y%m%d-%H%M%S")
    return f"{stamp}-{slugify(label)}-{suffix or secrets.token_hex(2)}"


def is_safe_id(value: str) -> bool:
    return _SAFE_ID.fullmatch(value) is not None


def split_email_id(email_id: str) -> tuple[str, int]:
    match = _EMAIL_ID.fullmatch(email_id)
    if match is None:
        raise KeyError(email_id)
    return match["generation"], int(match["index"])
```

- [ ] **Step 8: Run tests to verify they pass, then lint and type-check**

Run: `uv run pytest tests/test_settings.py tests/test_ids.py -v`
Expected: all PASS.

Run: `uv run ruff check --fix && uv run ruff format && uv run pyright`
Expected: `0 errors`.

- [ ] **Step 9: Commit**

```bash
git add pyproject.toml uv.lock .gitignore .env.example src tests
git commit -m "feat: scaffold uv project with settings and id helpers

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```
