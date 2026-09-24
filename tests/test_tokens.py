"""Tests for jev_bench.tokens."""

import pytest

from jev_bench.benchmark_config import EmbeddingParams, TokenParams
from jev_bench.catalog import ModelInfo
from jev_bench.tokens import Budget, chat_budget, embedding_budget, estimate_tokens, jev_budget

_TOKENS = TokenParams()
_BIG = ModelInfo(id="m", name="m", context_length=1_000_000, max_completion_tokens=128_000)
_BARE = ModelInfo(id="m", name="m")


@pytest.mark.parametrize(
    ("text", "bytes_per_token", "expected"),
    [
        pytest.param("", 3.0, 0, id="empty"),
        pytest.param("ab", 3.0, 1, id="rounds-up"),
        pytest.param("abcdef", 3.0, 2, id="ascii"),
        pytest.param("привет", 3.0, 4, id="cyrillic-two-bytes-per-char"),
        pytest.param("abcd", 4.0, 1, id="custom-ratio"),
    ],
)
def test_estimate_tokens(text: str, bytes_per_token: float, expected: int) -> None:
    assert estimate_tokens(text, bytes_per_token) == expected


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        pytest.param(_BIG, Budget(total=1_000_000, output=128_000), id="catalog-limits"),
        pytest.param(None, Budget(total=32_000, output=4_096), id="no-model"),
        pytest.param(_BARE, Budget(total=32_000, output=4_096), id="model-without-limits"),
    ],
)
def test_chat_budget(model: ModelInfo | None, expected: Budget) -> None:
    assert chat_budget(model, _TOKENS) == expected


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        pytest.param(
            ModelInfo(id="j", name="j", context_length=32_000),
            Budget(total=32_000),
            id="catalog",
        ),
        pytest.param(None, Budget(total=32_000), id="fallback"),
    ],
)
def test_jev_budget(model: ModelInfo | None, expected: Budget) -> None:
    assert jev_budget(model, _TOKENS) == expected


def test_embedding_budget_limits_request_and_item() -> None:
    params = EmbeddingParams(
        email_template="$body",
        option_template="$description",
        max_request_tokens=100_000,
    )
    model = ModelInfo(id="e", name="e", context_length=8192)
    assert embedding_budget(model, params, _TOKENS) == Budget(total=100_000, item=8192)
