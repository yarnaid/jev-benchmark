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
