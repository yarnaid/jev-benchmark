"""Tests for jev_bench.classifiers.embeddings."""

import asyncio
import json
import math
from pathlib import Path
from typing import Any

import httpx2
import pytest
from tests.factories import ClientFactory, EmailFactory

from jev_bench.benchmark_config import EmbeddingParams, TokenParams
from jev_bench.catalog import ModelInfo
from jev_bench.classifiers.embeddings import EmbeddingClassifier, email_text, option_texts
from jev_bench.emails import Email, Party
from jev_bench.openrouter import OpenRouterClient, OpenRouterError
from jev_bench.questions import QuestionSet
from jev_bench.store.embeddings import EmbeddingCache
from jev_bench.store.jsonfiles import read_jsonl
from jev_bench.tokens import Budget

_VECTORS: dict[str, list[float]] = {
    "spam": [1.0, 0.0, 0.0],
    "personal": [0.0, 1.0, 0.0],
    "work": [0.0, 0.0, 1.0],
    "low": [1.0, 0.0, 0.0],
    "today": [0.0, 1.0, 0.0],
    "now": [0.0, 0.0, 1.0],
    "yes": [1.0, 0.0, 0.0],
    "no": [0.0, 1.0, 0.0],
    "billing": [1.0, 0.0, 0.0],
    "meeting": [1.0, 0.1, 0.0],
    "travel": [0.0, 1.0, 0.0],
    "junk mail": [2.0, 0.0, 0.0],
    "hi friend": [0.0, 3.0, 0.0],
    "blank": [0.0, 0.0, 0.0],
    "flat": [1.0, 0.0],
}
_PARAMS = EmbeddingParams(
    email_template="$subject",
    option_template="$option",
    temperature=0.05,
    emails_per_request=2,
    concurrency=2,
)
_INFO = ModelInfo(id="openai/text-embedding-3-large", name="E", context_length=8192)


class EmbeddingServer:
    def __init__(
        self,
        vectors: dict[str, list[float]] = _VECTORS,
        drop_one: bool = False,
        omit_embedding: bool = False,
        bad_embedding: bool = False,
    ) -> None:
        self.vectors = vectors
        self.drop_one = drop_one
        self.omit_embedding = omit_embedding
        self.bad_embedding = bad_embedding
        self.inputs: list[list[str]] = []

    async def __call__(self, request: httpx2.Request) -> httpx2.Response:
        await asyncio.sleep(0)
        texts = json.loads(request.content)["input"]
        self.inputs.append(texts)
        data = [
            {"object": "embedding", "index": i, "embedding": self.vectors[text]}
            for i, text in enumerate(texts)
        ]
        if self.drop_one:
            data = data[1:]
        if self.omit_embedding:
            del data[0]["embedding"]
        if self.bad_embedding:
            data[0]["embedding"] = "not-a-vector"
        usage = {
            "prompt_tokens": 2 * len(texts),
            "total_tokens": 2 * len(texts),
            "cost": 0.0001 * len(texts),
        }
        return httpx2.Response(
            200,
            json={
                "model": "openai/text-embedding-3-large",
                "data": list(reversed(data)),
                "usage": usage,
            },
        )


def _classifier(
    client: OpenRouterClient, questions: QuestionSet, cache: EmbeddingCache
) -> EmbeddingClassifier:
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
    assert [option for _, option, _ in texts] == [
        "spam",
        "personal",
        "work",
        "low",
        "today",
        "now",
        "yes",
        "no",
    ]
    email = EmailFactory(
        sender=Party(name="Ann", address="ann@x.test"),
        to=(Party(name="", address="b@x.test"),),
        subject="S",
        body="B",
    )
    template = "$sent_at|$from|$to|$cc|$subject|$body"
    assert email_text(email, template) == "2026-09-20T09:30:00+00:00|Ann <ann@x.test>|b@x.test||S|B"


async def test_cold_run_embeds_options_then_emails(
    make_client: ClientFactory, questions: QuestionSet, tmp_path: Path
) -> None:
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
    assert (
        max(junk_outcome.answers["category"], key=junk_outcome.answers["category"].__getitem__)
        == "spam"
    )
    assert junk_outcome.similarities["category"] == pytest.approx(
        {"spam": 1.0, "personal": 0.0, "work": 0.0}
    )
    assert sum(junk_outcome.answers["urgency"].values()) == pytest.approx(1.0)
    friend_answers = result.outcomes[friend.id].answers
    assert friend_answers is not None
    assert friend_answers["needs_reply"]["no"] > 0.99
    assert result.usage.cost == pytest.approx(0.0002)
    assert result.raw is not None
    assert all("embedding" not in item for item in result.raw["data"])


async def test_warm_run_uses_the_cache(
    make_client: ClientFactory, questions: QuestionSet, tmp_path: Path
) -> None:
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
async def test_degenerate_email_vectors(
    make_client: ClientFactory, questions: QuestionSet, tmp_path: Path, subject: str, error: str
) -> None:
    classifier = _classifier(
        make_client(EmbeddingServer()), questions, EmbeddingCache(tmp_path / "v.jsonl")
    )
    email = _email(subject)
    await classifier.prepare([email])
    outcome = (await classifier.classify([email])).outcomes[email.id]
    assert outcome.answers is None
    assert outcome.error is not None
    assert error in outcome.error


@pytest.mark.parametrize(
    ("server", "match"),
    [
        pytest.param(EmbeddingServer(drop_one=True), "expected 2 embeddings", id="count-mismatch"),
        pytest.param(
            EmbeddingServer(omit_embedding=True),
            "malformed embeddings response",
            id="missing-embedding-key",
        ),
        pytest.param(
            EmbeddingServer(bad_embedding=True),
            "malformed embeddings response",
            id="non-numeric-embedding",
        ),
    ],
)
async def test_malformed_embedding_response_is_an_openrouter_error(
    make_client: ClientFactory,
    questions: QuestionSet,
    tmp_path: Path,
    server: EmbeddingServer,
    match: str,
) -> None:
    classifier = _classifier(make_client(server), questions, EmbeddingCache(tmp_path / "v.jsonl"))
    with pytest.raises(OpenRouterError, match=match):
        await classifier.prepare([])


async def test_concurrent_classifiers_share_one_cache(
    make_client: ClientFactory, questions: QuestionSet, tmp_path: Path
) -> None:
    cache = EmbeddingCache(tmp_path / "v.jsonl")
    first_server, second_server = EmbeddingServer(), EmbeddingServer()
    first = _classifier(make_client(first_server), questions, cache)
    second = _classifier(make_client(second_server), questions, cache)
    junk, friend = _email("junk mail"), _email("hi friend")
    await asyncio.gather(first.prepare([junk]), second.prepare([friend]))
    expected_options = [text for _, _, text in option_texts(questions, _PARAMS.option_template)]
    requested_options = [
        text
        for server in (first_server, second_server)
        for batch in server.inputs
        for text in batch
    ]
    assert sorted(requested_options) == sorted(expected_options)
    results = await asyncio.gather(first.classify([junk]), second.classify([friend]))
    assert all(
        outcome.answers is not None for result in results for outcome in result.outcomes.values()
    )
    records: list[Any] = read_jsonl(tmp_path / "v.jsonl")
    assert {record["ref"] for record in records} >= {junk.id, friend.id, "category:spam"}


async def test_classifier_shape(
    make_client: ClientFactory, questions: QuestionSet, tmp_path: Path
) -> None:
    classifier = _classifier(
        make_client(EmbeddingServer()), questions, EmbeddingCache(tmp_path / "v.jsonl")
    )
    assert classifier.emails_per_request == 2
    assert classifier.concurrency == 2
    assert classifier.budget == Budget(total=100_000, item=8192)
    assert classifier.sizing.overhead == 0
    assert classifier.input_tokens(_email("junk mail")) == 3


async def test_multi_questions_get_a_softmax_distribution(
    make_client: ClientFactory, multi_questions: QuestionSet, tmp_path: Path
) -> None:
    classifier = _classifier(
        make_client(EmbeddingServer()), multi_questions, EmbeddingCache(tmp_path / "v.jsonl")
    )
    junk = _email("junk mail")
    await classifier.prepare([junk])
    answers = (await classifier.classify([junk])).outcomes[junk.id].answers
    assert answers is not None
    topics = answers["topics"]
    assert sum(topics.values()) == pytest.approx(1.0)
    assert topics["meeting"] / topics["billing"] == pytest.approx(
        math.exp((1 / math.sqrt(1.01) - 1) / 0.05)
    )
