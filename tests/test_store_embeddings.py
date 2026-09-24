"""Tests for jev_bench.store.embeddings."""

from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest

from jev_bench.store.embeddings import (
    CachedVector,
    EmbeddingCache,
    EmbeddingCaches,
    decode_vector,
    encode_vector,
    text_key,
)
from jev_bench.store.jsonfiles import read_jsonl


def _entry(text: str, values: list[float], cost: float = 0.001) -> CachedVector:
    return CachedVector(
        key=text_key(text),
        kind="email",
        ref=text,
        dim=len(values),
        vector=encode_vector(values),
        cost=cost,
        input_tokens=3,
        created_at=datetime(2026, 9, 24, tzinfo=UTC),
    )


@pytest.mark.parametrize(
    "values",
    [
        pytest.param([0.5, -1.25, 3.0], id="exact-float32"),
        pytest.param([0.0], id="zero"),
        pytest.param([], id="empty"),
    ],
)
def test_vector_codec_round_trip(values: list[float]) -> None:
    assert decode_vector(encode_vector(values)).tolist() == values


def test_text_key_is_sha256_hex() -> None:
    assert text_key("abc") == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    assert text_key("abc") != text_key("abd")


def test_cache_put_get_and_reload(tmp_path: Path) -> None:
    path = tmp_path / "m" / "vectors.jsonl"
    cache = EmbeddingCache(path)
    assert cache.get(text_key("hello")) is None
    cache.put([_entry("hello", [1.0, 0.0])])
    reloaded = EmbeddingCache(path).get(text_key("hello"))
    assert reloaded is not None
    assert reloaded.array().tolist() == [1.0, 0.0]
    assert cache.path == path


def test_put_nothing_creates_no_file(tmp_path: Path) -> None:
    cache = EmbeddingCache(tmp_path / "vectors.jsonl")
    cache.put([])
    assert not (tmp_path / "vectors.jsonl").exists()


def test_invalid_cache_lines_are_skipped(tmp_path: Path) -> None:
    path = tmp_path / "vectors.jsonl"
    path.write_text('{"key": "only-a-key"}\n', encoding="utf-8")
    cache = EmbeddingCache(path)
    cache.put([_entry("ok", [1.0])])
    assert cache.get(text_key("ok")) is not None
    assert EmbeddingCache(path).get("only-a-key") is None


def test_registry_shares_one_cache_per_model(tmp_path: Path) -> None:
    caches = EmbeddingCaches(tmp_path)
    first = caches.for_model("openai/text-embedding-3-large")
    assert caches.for_model("openai/text-embedding-3-large") is first
    assert first.path == tmp_path / "openai-text-embedding-3-large" / "vectors.jsonl"
    assert caches.for_model("qwen/qwen3-embedding-8b") is not first


def test_duplicate_keys_last_line_wins(tmp_path: Path) -> None:
    path = tmp_path / "vectors.jsonl"
    EmbeddingCache(path).put([_entry("x", [1.0], cost=0.1)])
    EmbeddingCache(path).put([_entry("x", [2.0], cost=0.2)])
    assert len(read_jsonl(path)) == 2
    entry = EmbeddingCache(path).get(text_key("x"))
    assert entry is not None
    assert (entry.cost, np.asarray(entry.array()).tolist()) == (0.2, [2.0])
