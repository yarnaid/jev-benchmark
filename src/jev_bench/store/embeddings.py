"""Per-model embedding cache: append-only JSONL of float32 vectors keyed by the exact input text.

Classes:
    CachedVector: one cached embedding with its original cost share.
    EmbeddingCache: lazy load, lookup and append for one model (appends are synchronous);
        exposes `.lock`, an `asyncio.Lock` callers share to serialize a check-then-fetch-then-put
        sequence across concurrent users of the same instance.
    EmbeddingCaches: registry handing out one shared cache per model.
Functions:
    text_key: sha256 hex digest of a text.
    encode_vector, decode_vector: float32 little-endian base64 round-trip.
"""

import asyncio
import base64
import hashlib
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

import numpy as np
from loguru import logger
from pydantic import AwareDatetime, BaseModel, ConfigDict, ValidationError

from jev_bench.ids import slugify
from jev_bench.metrics.distributions import FloatArray
from jev_bench.store.jsonfiles import append_jsonl, read_jsonl

__all__ = [
    "VECTORS_FILE",
    "CachedVector",
    "EmbeddingCache",
    "EmbeddingCaches",
    "decode_vector",
    "encode_vector",
    "text_key",
]

VECTORS_FILE = "vectors.jsonl"


class CachedVector(BaseModel):
    model_config = ConfigDict(frozen=True)

    key: str
    kind: Literal["option", "email"]
    ref: str
    resolved_model: str | None = None
    dim: int
    vector: str
    cost: float = 0.0
    input_tokens: int = 0
    created_at: AwareDatetime

    def array(self) -> FloatArray:
        return decode_vector(self.vector)


def text_key(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def encode_vector(values: FloatArray | Sequence[float]) -> str:
    return base64.b64encode(np.asarray(values, dtype="<f4").tobytes()).decode("ascii")


def decode_vector(encoded: str) -> FloatArray:
    return np.frombuffer(base64.b64decode(encoded), dtype="<f4").astype(np.float64)


class EmbeddingCache:
    def __init__(self, path: Path) -> None:
        self._path = path
        self._entries: dict[str, CachedVector] | None = None
        self.lock: asyncio.Lock = asyncio.Lock()

    @property
    def path(self) -> Path:
        return self._path

    def get(self, key: str) -> CachedVector | None:
        return self._load().get(key)

    def put(self, entries: Sequence[CachedVector]) -> None:
        append_jsonl(self._path, [entry.model_dump(mode="json") for entry in entries])
        self._load().update({entry.key: entry for entry in entries})

    def _load(self) -> dict[str, CachedVector]:
        if self._entries is None:
            self._entries = {
                entry.key: entry for entry in _valid_entries(read_jsonl(self._path), self._path)
            }
        return self._entries


def _valid_entries(records: list[object], path: Path) -> list[CachedVector]:
    entries: list[CachedVector] = []
    for record in records:
        try:
            entries.append(CachedVector.model_validate(record))
        except ValidationError:
            logger.bind(path=str(path)).warning("skipping invalid embedding cache entry")
    return entries


class EmbeddingCaches:
    def __init__(self, root: Path) -> None:
        self._root = root
        self._caches: dict[str, EmbeddingCache] = {}

    def for_model(self, model: str) -> EmbeddingCache:
        if model not in self._caches:
            self._caches[model] = EmbeddingCache(self._root / slugify(model) / VECTORS_FILE)
        return self._caches[model]
