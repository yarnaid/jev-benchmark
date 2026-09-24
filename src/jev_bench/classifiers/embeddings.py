"""Embedding column: cosine similarity between email and option texts, softmaxed per question.

Constants:
    EMBEDDINGS_PATH
Classes:
    EmbeddingClassifier: cache-first Classifier (only texts missing from the cache are embedded).
        `prepare()` holds the cache's lock across the whole option-resolution step, so concurrent
        classifiers sharing one `EmbeddingCache` never both pay to embed the same option text.
Functions:
    option_texts: (question id, option id, rendered text) for every option.
    email_text: rendered email text (the cache key's source).
"""

from collections.abc import Sequence
from datetime import UTC, datetime
from string import Template
from typing import Any, Literal

import numpy as np
from pydantic import SecretStr

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
        (
            question.id,
            option,
            parsed.substitute(instructions=question.instructions, option=option, description=text),
        )
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
        self._api_key = SecretStr(api_key)
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
        return estimate_tokens(
            email_text(email, self._params.email_template), self._bytes_per_token
        )

    async def prepare(self, emails: Sequence[Email]) -> PrepareResult:
        items: list[CacheItem] = [
            ("option", f"{question}:{option}", text) for question, option, text in self._options
        ]
        async with self._cache.lock:
            cached_cost = sum(entry.cost for entry in self._cached(items))
            usage = await self._ensure_cached(items)
            self._matrices = self._option_matrices()
        resolved, pending = self._split_cached(emails)
        return PrepareResult(
            usage=usage, cached_cost=cached_cost, resolved=resolved, pending=tuple(pending)
        )

    async def classify(
        self, emails: Sequence[Email], on_progress: ProgressCallback | None = None
    ) -> RequestResult:
        items: list[CacheItem] = [("email", email.id, self._text(email)) for email in emails]
        usage, response = await self._embed_and_cache(items)
        outcomes = {
            email.id: self._outcome(self._vector(item[2]))
            for email, item in zip(emails, items, strict=True)
        }
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
        return [
            entry for _, _, text in items if (entry := self._cache.get(text_key(text))) is not None
        ]

    async def _ensure_cached(self, items: Sequence[CacheItem]) -> Usage:
        missing = [item for item in _unique(items) if self._cache.get(text_key(item[2])) is None]
        usage = Usage()
        for chunk in chunk_by_count(len(missing), self._params.emails_per_request):
            chunk_usage, _ = await self._embed_and_cache([missing[index] for index in chunk])
            usage = usage.plus(chunk_usage)
        return usage

    async def _embed_and_cache(self, items: Sequence[CacheItem]) -> tuple[Usage, ApiResponse]:
        body = {"model": self._model, "input": [text for _, _, text in items]}
        response = await self._client.post_json(
            EMBEDDINGS_PATH, body, api_key=self._api_key.get_secret_value()
        )
        vectors = _vectors(response.body, len(items))
        usage = usage_from_body(response.body.get("usage"), self._info)
        model = response.body.get("model")
        self._cache.put(
            [
                _entry(item, vector, usage, len(items), model)
                for item, vector in zip(items, vectors, strict=True)
            ]
        )
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
                resolved[email.id] = outcome.model_copy(
                    update={"cached": True, "cached_cost": entry.cost}
                )
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
        scored = {
            question.id: self._score(question, unit) for question in self._questions.questions
        }
        answers = {question_id: probabilities for question_id, (probabilities, _) in scored.items()}
        similarities = {question_id: cosines for question_id, (_, cosines) in scored.items()}
        return EmailOutcome(answers=answers, similarities=similarities)

    def _score(
        self, question: AnyQuestion, unit: FloatArray
    ) -> tuple[dict[str, float], dict[str, float]]:
        cosines = self._matrices[question.id] @ unit
        probabilities = softmax(cosines, self._params.temperature)
        return _labelled(question.option_ids, probabilities), _labelled(
            question.option_ids, cosines
        )


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


def _entry(
    item: CacheItem, vector: FloatArray, usage: Usage, batch: int, model: str | None
) -> CachedVector:
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
    data = [
        {key: value for key, value in item.items() if key != "embedding"}
        for item in body.get("data", [])
        if isinstance(item, dict)
    ]
    return {**body, "data": data}
