### Task 12: Embedding cache and the embedding classifier

**Files:**
- Create: `src/jev_bench/store/embeddings.py`, `src/jev_bench/classifiers/embeddings.py`
- Test: `tests/test_store_embeddings.py`, `tests/test_classifiers_embeddings.py`

**Interfaces:**
- Consumes:
  - Task 1: `slugify`;
  - Task 2: `Email`, `QuestionSet`, `Distribution`;
  - Task 3: `append_jsonl`, `read_jsonl`;
  - Task 4: `softmax`, `FloatArray`;
  - Task 5: `OpenRouterClient`, `OpenRouterError`, `ApiResponse`;
  - Task 6: `ModelInfo`, `EmbeddingParams`, `TokenParams`;
  - Task 7: `embedding_budget`, `estimate_tokens`, `Sizing`, `chunk_by_count`;
  - Task 8: `Usage`, `EmailOutcome`, `PrepareResult`, `RequestResult`, `ProgressCallback`,
    `usage_from_body`.
- Produces:
  - `jev_bench.store.embeddings`:
    - `CachedVector(key, kind, ref, resolved_model, dim, vector, cost, input_tokens, created_at)` with
      `.array() -> FloatArray`;
    - `text_key(text) -> str`, `encode_vector(values) -> str`, `decode_vector(encoded) -> FloatArray`;
    - `EmbeddingCache(path)` with `.get(key)`, `.put(entries)` and `.path`;
    - `EmbeddingCaches(root)` with `.for_model(model) -> EmbeddingCache` (same instance per model).
  - `jev_bench.classifiers.embeddings`:
    - `EMBEDDINGS_PATH = "/v1/embeddings"`;
    - `option_texts(qs, template) -> list[tuple[str, str, str]]`;
    - `email_text(email, template) -> str`;
    - `EmbeddingClassifier(*, client, api_key, model, model_info, questions, params: EmbeddingParams,
      tokens: TokenParams, cache: EmbeddingCache)`.

Behaviour:
- `prepare()` resolves option vectors cache-first and embeds only the missing option texts, in chunks of
  `emails_per_request`. It returns:
  - `usage` = what was paid;
  - `cached_cost` = the originally recorded cost of the option vectors that were already cached;
  - `resolved` = emails already in the cache, each with `cached=True` and `cached_cost` = its recorded
    share;
  - `pending` = the rest.
- `classify(batch)` embeds the batch, caches it, and scores it. For each question: cosine against the unit
  option vectors, then `softmax(cos / τ)`. Similarities are kept in the outcome. The raw body is stored
  without vectors.

Review Focus #5 is pinned here: two classifiers sharing one cache instance run concurrently, both finish,
and the cache file stays parseable.

- [ ] **Step 1: Write the failing tests**

`tests/test_store_embeddings.py`:
```python
"""Tests for jev_bench.store.embeddings."""

from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest

from jev_bench.store.embeddings import CachedVector, EmbeddingCache, EmbeddingCaches, decode_vector, encode_vector, text_key
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
```

`tests/test_classifiers_embeddings.py`:
```python
"""Tests for jev_bench.classifiers.embeddings."""

import asyncio
import json
from pathlib import Path
from typing import Any

import httpx2
import pytest

from jev_bench.benchmark_config import EmbeddingParams, TokenParams
from jev_bench.catalog import ModelInfo
from jev_bench.classifiers.embeddings import EmbeddingClassifier, email_text, option_texts
from jev_bench.emails import Email, Party
from jev_bench.openrouter import OpenRouterClient, OpenRouterError
from jev_bench.questions import QuestionSet
from jev_bench.store.embeddings import EmbeddingCache
from jev_bench.store.jsonfiles import read_jsonl
from jev_bench.tokens import Budget
from tests.factories import ClientFactory, EmailFactory

_VECTORS: dict[str, list[float]] = {
    "spam": [1.0, 0.0, 0.0],
    "personal": [0.0, 1.0, 0.0],
    "work": [0.0, 0.0, 1.0],
    "low": [1.0, 0.0, 0.0],
    "today": [0.0, 1.0, 0.0],
    "now": [0.0, 0.0, 1.0],
    "yes": [1.0, 0.0, 0.0],
    "no": [0.0, 1.0, 0.0],
    "junk mail": [2.0, 0.0, 0.0],
    "hi friend": [0.0, 3.0, 0.0],
    "blank": [0.0, 0.0, 0.0],
    "flat": [1.0, 0.0],
}
_PARAMS = EmbeddingParams(email_template="$subject", option_template="$option", temperature=0.05, emails_per_request=2, concurrency=2)
_INFO = ModelInfo(id="openai/text-embedding-3-large", name="E", context_length=8192)


class EmbeddingServer:
    def __init__(self, vectors: dict[str, list[float]] = _VECTORS, drop_one: bool = False) -> None:
        self.vectors = vectors
        self.drop_one = drop_one
        self.inputs: list[list[str]] = []

    def __call__(self, request: httpx2.Request) -> httpx2.Response:
        texts = json.loads(request.content)["input"]
        self.inputs.append(texts)
        data = [{"object": "embedding", "index": i, "embedding": self.vectors[text]} for i, text in enumerate(texts)]
        if self.drop_one:
            data = data[1:]
        usage = {"prompt_tokens": 2 * len(texts), "total_tokens": 2 * len(texts), "cost": 0.0001 * len(texts)}
        return httpx2.Response(200, json={"model": "openai/text-embedding-3-large", "data": list(reversed(data)), "usage": usage})


def _classifier(client: OpenRouterClient, questions: QuestionSet, cache: EmbeddingCache) -> EmbeddingClassifier:
    return EmbeddingClassifier(
        client=client,
        api_key="sk-test",
        model="openai/text-embedding-3-large",
        model_info=_INFO,
        questions=questions,
        params=_PARAMS,
        tokens=TokenParams(),
        cache=cache,
    )


def _email(subject: str) -> Email:
    return EmailFactory(subject=subject)


def test_option_and_email_texts(questions: QuestionSet) -> None:
    texts = option_texts(questions, "$instructions | $option | $description")
    assert texts[0] == ("category", "spam", "What kind of email? | spam | Junk")
    assert [option for _, option, _ in texts] == ["spam", "personal", "work", "low", "today", "now", "yes", "no"]
    email = EmailFactory(sender=Party(name="Ann", address="ann@x.test"), to=(Party(name="", address="b@x.test"),), subject="S", body="B")
    template = "$sent_at|$from|$to|$cc|$subject|$body"
    assert email_text(email, template) == "2026-09-20T09:30:00+00:00|Ann <ann@x.test>|b@x.test||S|B"


async def test_cold_run_embeds_options_then_emails(make_client: ClientFactory, questions: QuestionSet, tmp_path: Path) -> None:
    server = EmbeddingServer()
    classifier = _classifier(make_client(server), questions, EmbeddingCache(tmp_path / "v.jsonl"))
    junk, friend = _email("junk mail"), _email("hi friend")
    prepared = await classifier.prepare([junk, friend])
    assert [len(batch) for batch in server.inputs] == [2, 2, 2, 2]
    assert prepared.usage.cost == pytest.approx(0.0008)
    assert prepared.cached_cost == 0.0
    assert prepared.resolved == {}
    assert prepared.pending == (junk, friend)
    result = await classifier.classify([junk, friend])
    junk_outcome = result.outcomes[junk.id]
    assert junk_outcome.answers is not None
    assert junk_outcome.similarities is not None
    assert max(junk_outcome.answers["category"], key=junk_outcome.answers["category"].__getitem__) == "spam"
    assert junk_outcome.similarities["category"] == pytest.approx({"spam": 1.0, "personal": 0.0, "work": 0.0})
    assert sum(junk_outcome.answers["urgency"].values()) == pytest.approx(1.0)
    friend_answers = result.outcomes[friend.id].answers
    assert friend_answers is not None
    assert friend_answers["needs_reply"]["no"] > 0.99
    assert result.usage.cost == pytest.approx(0.0002)
    assert result.raw is not None
    assert all("embedding" not in item for item in result.raw["data"])


async def test_warm_run_uses_the_cache(make_client: ClientFactory, questions: QuestionSet, tmp_path: Path) -> None:
    cache = EmbeddingCache(tmp_path / "v.jsonl")
    junk = _email("junk mail")
    cold = _classifier(make_client(EmbeddingServer()), questions, cache)
    await cold.prepare([junk])
    await cold.classify([junk])
    server = EmbeddingServer()
    warm = _classifier(make_client(server), questions, EmbeddingCache(tmp_path / "v.jsonl"))
    prepared = await warm.prepare([junk, _email("hi friend")])
    assert server.inputs == []
    assert prepared.usage.cost == 0.0
    assert prepared.cached_cost == pytest.approx(0.0008)
    assert prepared.resolved[junk.id].cached is True
    assert prepared.resolved[junk.id].cached_cost == pytest.approx(0.0001)
    assert [email.subject for email in prepared.pending] == ["hi friend"]


@pytest.mark.parametrize(
    ("subject", "error"),
    [
        pytest.param("blank", "zero-length email embedding", id="zero-vector"),
        pytest.param("flat", "dimension mismatch", id="dimension-mismatch"),
    ],
)
async def test_degenerate_email_vectors(make_client: ClientFactory, questions: QuestionSet, tmp_path: Path, subject: str, error: str) -> None:
    classifier = _classifier(make_client(EmbeddingServer()), questions, EmbeddingCache(tmp_path / "v.jsonl"))
    email = _email(subject)
    await classifier.prepare([email])
    outcome = (await classifier.classify([email])).outcomes[email.id]
    assert outcome.answers is None
    assert outcome.error is not None
    assert error in outcome.error


async def test_embedding_count_mismatch_is_an_openrouter_error(make_client: ClientFactory, questions: QuestionSet, tmp_path: Path) -> None:
    classifier = _classifier(make_client(EmbeddingServer(drop_one=True)), questions, EmbeddingCache(tmp_path / "v.jsonl"))
    with pytest.raises(OpenRouterError, match="expected 2 embeddings"):
        await classifier.prepare([])


async def test_concurrent_classifiers_share_one_cache(make_client: ClientFactory, questions: QuestionSet, tmp_path: Path) -> None:
    cache = EmbeddingCache(tmp_path / "v.jsonl")
    first = _classifier(make_client(EmbeddingServer()), questions, cache)
    second = _classifier(make_client(EmbeddingServer()), questions, cache)
    junk, friend = _email("junk mail"), _email("hi friend")
    await asyncio.gather(first.prepare([junk]), second.prepare([friend]))
    results = await asyncio.gather(first.classify([junk]), second.classify([friend]))
    assert all(outcome.answers is not None for result in results for outcome in result.outcomes.values())
    records: list[Any] = read_jsonl(tmp_path / "v.jsonl")
    assert {record["ref"] for record in records} >= {junk.id, friend.id, "category:spam"}


async def test_classifier_shape(make_client: ClientFactory, questions: QuestionSet, tmp_path: Path) -> None:
    classifier = _classifier(make_client(EmbeddingServer()), questions, EmbeddingCache(tmp_path / "v.jsonl"))
    assert classifier.emails_per_request == 2
    assert classifier.concurrency == 2
    assert classifier.budget == Budget(total=100_000, item=8192)
    assert classifier.sizing.overhead == 0
    assert classifier.input_tokens(_email("junk mail")) == 3
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_store_embeddings.py tests/test_classifiers_embeddings.py -v`
Expected: `ModuleNotFoundError` for `jev_bench.store.embeddings` / `jev_bench.classifiers.embeddings`.

- [ ] **Step 3: Write the implementation**

`src/jev_bench/store/embeddings.py`:
```python
"""Per-model embedding cache: append-only JSONL of float32 vectors keyed by the exact input text.

Classes:
    CachedVector: one cached embedding with its original cost share.
    EmbeddingCache: lazy load, lookup and append for one model (appends are synchronous).
    EmbeddingCaches: registry handing out one shared cache per model.
Functions:
    text_key: sha256 hex digest of a text.
    encode_vector, decode_vector: float32 little-endian base64 round-trip.
"""

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
            self._entries = {entry.key: entry for entry in _valid_entries(read_jsonl(self._path), self._path)}
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
```

`src/jev_bench/classifiers/embeddings.py`:
```python
"""Embedding column: cosine similarity between each email and every option text, softmaxed per question.

Constants:
    EMBEDDINGS_PATH
Classes:
    EmbeddingClassifier: cache-first Classifier (only texts missing from the cache are embedded).
Functions:
    option_texts: (question id, option id, rendered text) for every option.
    email_text: rendered email text (the cache key's source).
"""

from collections.abc import Sequence
from datetime import UTC, datetime
from string import Template
from typing import Any, Literal

import numpy as np

from jev_bench.benchmark_config import EmbeddingParams, TokenParams
from jev_bench.catalog import ModelInfo
from jev_bench.classifiers.base import (
    EmailOutcome,
    PrepareResult,
    ProgressCallback,
    RequestResult,
    Usage,
    usage_from_body,
)
from jev_bench.emails import Email
from jev_bench.metrics.distributions import FloatArray, softmax
from jev_bench.openrouter import ApiResponse, JsonObject, OpenRouterClient, OpenRouterError
from jev_bench.questions import AnyQuestion, QuestionSet
from jev_bench.request_plan import Sizing, chunk_by_count
from jev_bench.store.embeddings import CachedVector, EmbeddingCache, encode_vector, text_key
from jev_bench.tokens import embedding_budget, estimate_tokens

EMBEDDINGS_PATH = "/v1/embeddings"

type CacheItem = tuple[Literal["option", "email"], str, str]


def option_texts(questions: QuestionSet, template: str) -> list[tuple[str, str, str]]:
    parsed = Template(template)
    return [
        (question.id, option, parsed.substitute(instructions=question.instructions, option=option, description=text))
        for question in questions.questions
        for option, text in question.options.items()
    ]


def email_text(email: Email, template: str) -> str:
    state = email.to_state()
    values = {
        "sent_at": state["sent_at"],
        "from": state["from"],
        "to": ", ".join(state["to"]),
        "cc": ", ".join(state["cc"]),
        "subject": state["subject"],
        "body": state["body"],
    }
    return Template(template).substitute(values)


class EmbeddingClassifier:
    def __init__(
        self,
        *,
        client: OpenRouterClient,
        api_key: str,
        model: str,
        model_info: ModelInfo | None,
        questions: QuestionSet,
        params: EmbeddingParams,
        tokens: TokenParams,
        cache: EmbeddingCache,
    ) -> None:
        self._client = client
        self._api_key = api_key
        self._model = model
        self._info = model_info
        self._questions = questions
        self._params = params
        self._cache = cache
        self._bytes_per_token = tokens.bytes_per_token
        self._options = option_texts(questions, params.option_template)
        self._matrices: dict[str, FloatArray] = {}
        self.emails_per_request: int | None = params.emails_per_request
        self.concurrency = params.concurrency
        self.budget = embedding_budget(model_info, params, tokens)
        self.sizing = Sizing()

    def input_tokens(self, email: Email) -> int:
        return estimate_tokens(email_text(email, self._params.email_template), self._bytes_per_token)

    async def prepare(self, emails: Sequence[Email]) -> PrepareResult:
        items: list[CacheItem] = [("option", f"{question}:{option}", text) for question, option, text in self._options]
        cached_cost = sum(entry.cost for entry in self._cached(items))
        usage = await self._ensure_cached(items)
        self._matrices = self._option_matrices()
        resolved, pending = self._split_cached(emails)
        return PrepareResult(usage=usage, cached_cost=cached_cost, resolved=resolved, pending=tuple(pending))

    async def classify(
        self, emails: Sequence[Email], on_progress: ProgressCallback | None = None
    ) -> RequestResult:
        items: list[CacheItem] = [("email", email.id, self._text(email)) for email in emails]
        usage, response = await self._embed_and_cache(items)
        outcomes = {email.id: self._outcome(self._vector(item[2])) for email, item in zip(emails, items, strict=True)}
        return RequestResult(
            outcomes=outcomes,
            usage=usage,
            latency_ms=response.latency_ms,
            resolved_model=response.body.get("model"),
            raw=_without_vectors(response.body),
        )

    def _text(self, email: Email) -> str:
        return email_text(email, self._params.email_template)

    def _cached(self, items: Sequence[CacheItem]) -> list[CachedVector]:
        return [entry for _, _, text in items if (entry := self._cache.get(text_key(text))) is not None]

    async def _ensure_cached(self, items: Sequence[CacheItem]) -> Usage:
        missing = [item for item in _unique(items) if self._cache.get(text_key(item[2])) is None]
        usage = Usage()
        for chunk in chunk_by_count(len(missing), self._params.emails_per_request):
            chunk_usage, _ = await self._embed_and_cache([missing[index] for index in chunk])
            usage = usage.plus(chunk_usage)
        return usage

    async def _embed_and_cache(self, items: Sequence[CacheItem]) -> tuple[Usage, ApiResponse]:
        body = {"model": self._model, "input": [text for _, _, text in items]}
        response = await self._client.post_json(EMBEDDINGS_PATH, body, api_key=self._api_key)
        vectors = _vectors(response.body, len(items))
        usage = usage_from_body(response.body.get("usage"), self._info)
        model = response.body.get("model")
        self._cache.put([_entry(item, vector, usage, len(items), model) for item, vector in zip(items, vectors, strict=True)])
        return usage, response

    def _split_cached(self, emails: Sequence[Email]) -> tuple[dict[str, EmailOutcome], list[Email]]:
        resolved: dict[str, EmailOutcome] = {}
        pending: list[Email] = []
        for email in emails:
            entry = self._cache.get(text_key(self._text(email)))
            if entry is None:
                pending.append(email)
            else:
                outcome = self._outcome(entry.array())
                resolved[email.id] = outcome.model_copy(update={"cached": True, "cached_cost": entry.cost})
        return resolved, pending

    def _option_matrices(self) -> dict[str, FloatArray]:
        rows: dict[str, list[FloatArray]] = {}
        for question_id, _, text in self._options:
            rows.setdefault(question_id, []).append(_unit(self._vector(text)))
        return {question_id: np.vstack(vectors) for question_id, vectors in rows.items()}

    def _vector(self, text: str) -> FloatArray:
        entry = self._cache.get(text_key(text))
        if entry is None:
            raise OpenRouterError(f"embedding missing from cache for {text[:40]!r}")
        return entry.array()

    def _outcome(self, vector: FloatArray) -> EmailOutcome:
        norm = float(np.linalg.norm(vector))
        if norm == 0.0:
            return EmailOutcome(error="zero-length email embedding")
        unit = vector / norm
        if any(matrix.shape[1] != unit.shape[0] for matrix in self._matrices.values()):
            return EmailOutcome(error="embedding dimension mismatch between email and options")
        scored = {question.id: self._score(question, unit) for question in self._questions.questions}
        answers = {question_id: probabilities for question_id, (probabilities, _) in scored.items()}
        similarities = {question_id: cosines for question_id, (_, cosines) in scored.items()}
        return EmailOutcome(answers=answers, similarities=similarities)

    def _score(self, question: AnyQuestion, unit: FloatArray) -> tuple[dict[str, float], dict[str, float]]:
        cosines = self._matrices[question.id] @ unit
        probabilities = softmax(cosines, self._params.temperature)
        return _labelled(question.option_ids, probabilities), _labelled(question.option_ids, cosines)


def _unique(items: Sequence[CacheItem]) -> list[CacheItem]:
    by_text: dict[str, CacheItem] = {}
    for item in items:
        by_text.setdefault(item[2], item)
    return list(by_text.values())


def _unit(vector: FloatArray) -> FloatArray:
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 0 else vector


def _labelled(options: Sequence[str], values: FloatArray) -> dict[str, float]:
    return {option: float(value) for option, value in zip(options, values, strict=True)}


def _vectors(body: JsonObject, expected: int) -> list[FloatArray]:
    data = body.get("data")
    if not isinstance(data, list) or len(data) != expected:
        count = len(data) if isinstance(data, list) else "no"
        raise OpenRouterError(f"expected {expected} embeddings, got {count}")
    try:
        ordered = sorted(data, key=lambda item: int(item.get("index", 0)))
        return [np.asarray(item["embedding"], dtype=np.float64) for item in ordered]
    except (AttributeError, KeyError, TypeError, ValueError) as exc:
        raise OpenRouterError(f"malformed embeddings response: {exc!r}") from exc


def _entry(item: CacheItem, vector: FloatArray, usage: Usage, batch: int, model: str | None) -> CachedVector:
    kind, ref, text = item
    return CachedVector(
        key=text_key(text),
        kind=kind,
        ref=ref,
        resolved_model=model,
        dim=int(vector.shape[0]),
        vector=encode_vector(vector),
        cost=usage.cost / batch,
        input_tokens=round(usage.input_tokens / batch),
        created_at=datetime.now(UTC),
    )


def _without_vectors(body: JsonObject) -> dict[str, Any]:
    data = [{key: value for key, value in item.items() if key != "embedding"} for item in body.get("data", []) if isinstance(item, dict)]
    return {**body, "data": data}
```

- [ ] **Step 4: Run tests to verify they pass, lint, type-check**

Run: `uv run pytest tests/test_store_embeddings.py tests/test_classifiers_embeddings.py -v`
Expected: all PASS.

Run: `uv run ruff check --fix && uv run ruff format && uv run pyright`
Expected: `0 errors`.

- [ ] **Step 5: Commit**

```bash
git add src/jev_bench/store/embeddings.py src/jev_bench/classifiers/embeddings.py tests/test_store_embeddings.py tests/test_classifiers_embeddings.py
git commit -m "feat(embeddings): per-model vector cache and cosine-softmax classifier

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```
