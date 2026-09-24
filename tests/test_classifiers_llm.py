"""Tests for jev_bench.classifiers.llm."""

import asyncio
import json
from typing import Any

import httpx2
import pytest
from tests.factories import ClientFactory, EmailFactory, chat_body, sse_body

from jev_bench.benchmark_config import ChatMode, LlmParams, TokenParams
from jev_bench.catalog import ModelInfo
from jev_bench.classifiers.llm import LlmClassifier
from jev_bench.classifiers.llm_schema import all_in_one_schema, answers_schema
from jev_bench.emails import Email
from jev_bench.openrouter import OpenRouterClient, OpenRouterError
from jev_bench.questions import QuestionSet, render_questions

_INFO = ModelInfo(
    id="anthropic/claude-sonnet-5",
    name="Sonnet",
    context_length=1_000_000,
    max_completion_tokens=128_000,
)
_ANSWER = {
    "category": {"spam": 0.6, "personal": 0.3, "work": 0.1},
    "urgency": {"low": 0.2, "today": 0.5, "now": 0.3},
    "needs_reply": 0.8,
}


def _params(**overrides: Any) -> LlmParams:
    return LlmParams(
        system_prompt="Classify.\n$questions",
        system_prompt_all_in_one="Batch.\n$questions",
        **overrides,
    )


def _classifier(
    client: OpenRouterClient,
    questions: QuestionSet,
    mode: ChatMode = "per_email",
    *,
    cache: bool = False,
    **params: Any,
) -> LlmClassifier:
    return LlmClassifier(
        client=client,
        api_key="sk-test",
        model="anthropic/claude-sonnet-5",
        model_info=_INFO,
        questions=questions,
        params=_params(**params),
        tokens=TokenParams(),
        mode=mode,
        cache_system_prompt=cache,
    )


class Recorder:
    def __init__(self, response: httpx2.Response) -> None:
        self.response = response
        self.bodies: list[dict[str, Any]] = []

    def __call__(self, request: httpx2.Request) -> httpx2.Response:
        self.bodies.append(json.loads(request.content))
        return self.response


def _json_response(content: str, **kwargs: Any) -> httpx2.Response:
    return httpx2.Response(200, json=chat_body(content, **kwargs))


def _sse_response(parts: list[str], **kwargs: Any) -> httpx2.Response:
    return httpx2.Response(
        200, content=sse_body(parts, **kwargs), headers={"content-type": "text/event-stream"}
    )


async def test_per_email_request_body(make_client: ClientFactory, questions: QuestionSet) -> None:
    recorder = Recorder(_json_response(json.dumps(_ANSWER)))
    classifier = _classifier(make_client(recorder), questions)
    email = EmailFactory()
    await classifier.classify([email])
    body = recorder.bodies[0]
    expected_input = classifier.sizing.overhead + classifier.input_tokens(email)
    assert body["model"] == "anthropic/claude-sonnet-5"
    assert body["messages"] == [
        {"role": "system", "content": "Classify.\n" + render_questions(questions)},
        {"role": "user", "content": json.dumps(email.to_state(), ensure_ascii=False)},
    ]
    assert body["temperature"] == 0.0
    assert body["reasoning"] == {"enabled": False}
    assert body["provider"] == {"require_parameters": True}
    assert body["response_format"]["json_schema"] == {
        "name": "email_triage",
        "strict": True,
        "schema": answers_schema(questions),
    }
    assert body["max_tokens"] == min(128_000, 1_000_000 - expected_input)
    assert "stream" not in body


async def test_cached_system_prompt_uses_a_cache_control_part(
    make_client: ClientFactory, questions: QuestionSet
) -> None:
    recorder = Recorder(_json_response(json.dumps(_ANSWER)))
    await _classifier(make_client(recorder), questions, cache=True).classify([EmailFactory()])
    system = recorder.bodies[0]["messages"][0]
    assert system["content"] == [
        {
            "type": "text",
            "text": "Classify.\n" + render_questions(questions),
            "cache_control": {"type": "ephemeral"},
        }
    ]


@pytest.mark.parametrize(
    ("content", "finish_reason", "error"),
    [
        pytest.param(json.dumps(_ANSWER), "stop", None, id="valid"),
        pytest.param("```json\n" + json.dumps(_ANSWER) + "\n```", "stop", None, id="code-fence"),
        pytest.param("{not json", "stop", "response is not valid JSON", id="invalid-json"),
        pytest.param('{"category": {"spam"', "length", "truncated", id="truncated"),
        pytest.param("   ", "stop", "empty response", id="blank"),
    ],
)
async def test_per_email_outcomes(
    make_client: ClientFactory,
    questions: QuestionSet,
    content: str,
    finish_reason: str,
    error: str | None,
) -> None:
    email = EmailFactory()
    result = await _classifier(
        make_client(Recorder(_json_response(content, finish_reason=finish_reason))), questions
    ).classify([email])
    outcome = result.outcomes[email.id]
    assert result.usage.cost == 0.001
    assert result.resolved_model == "test/model"
    if error is None:
        assert outcome.answers is not None
        assert outcome.answers["needs_reply"] == pytest.approx({"yes": 0.8, "no": 0.2})
    else:
        assert outcome.answers is None
        assert outcome.error is not None
        assert error in outcome.error


async def test_all_in_one_streams_refs_and_parses(
    make_client: ClientFactory, questions: QuestionSet
) -> None:
    emails = [EmailFactory(), EmailFactory(), EmailFactory()]
    results = {
        "results": [
            {"ref": "e001", **_ANSWER},
            {"ref": "e003", **_ANSWER},
            {"ref": "e001", **_ANSWER},
        ]
    }
    text = json.dumps(results)
    recorder = Recorder(_sse_response([text[:30], text[30:]]))
    progress: list[int] = []
    classifier = _classifier(make_client(recorder), questions, "all_in_one")
    result = await classifier.classify(emails, progress.append)
    body = recorder.bodies[0]
    user = body["messages"][1]["content"]
    assert body["stream"] is True
    assert body["messages"][0]["content"] == "Batch.\n" + render_questions(questions)
    assert json.loads(user) == [
        {"ref": f"e00{i + 1}", **email.to_state()} for i, email in enumerate(emails)
    ]
    assert all(email.id not in user for email in emails)
    assert body["response_format"]["json_schema"]["schema"] == all_in_one_schema(
        questions, ["e001", "e002", "e003"]
    )
    assert result.outcomes[emails[0].id].notes == (
        "duplicate ref in response; first occurrence kept",
    )
    assert result.outcomes[emails[0].id].answers is not None
    assert result.outcomes[emails[1].id].error == "missing from all-in-one response"
    assert result.outcomes[emails[2].id].answers is not None
    assert result.usage.cost == 0.002
    assert progress[-1] == 3


async def test_all_in_one_truncated_fails_every_email(
    make_client: ClientFactory, questions: QuestionSet
) -> None:
    emails = [EmailFactory(), EmailFactory()]
    recorder = Recorder(_sse_response(['{"results": [{"ref": "e001"'], finish_reason="length"))
    result = await _classifier(make_client(recorder), questions, "all_in_one").classify(emails)
    for email in emails:
        error = result.outcomes[email.id].error
        assert error is not None
        assert error.startswith("unusable all-in-one response: response truncated")


async def test_all_in_one_timeout_raises(
    make_client: ClientFactory, questions: QuestionSet
) -> None:
    async def slow(request: httpx2.Request) -> httpx2.Response:
        await asyncio.sleep(0.05)
        return _sse_response(["{}"])

    classifier = _classifier(make_client(slow), questions, "all_in_one", all_in_one_timeout_s=0.01)
    with pytest.raises(OpenRouterError, match="exceeded"):
        await classifier.classify([EmailFactory()])


def _unused(request: httpx2.Request) -> httpx2.Response:
    return httpx2.Response(500)


@pytest.mark.parametrize(
    ("mode", "per_request"),
    [
        pytest.param("per_email", 1, id="per-email"),
        pytest.param("all_in_one", None, id="all-in-one"),
    ],
)
async def test_classifier_shape(
    make_client: ClientFactory, questions: QuestionSet, mode: ChatMode, per_request: int | None
) -> None:
    classifier = _classifier(make_client(_unused), questions, mode)
    email: Email = EmailFactory()
    assert classifier.emails_per_request == per_request
    assert classifier.concurrency == 8
    assert classifier.sizing.output_per_email == 400
    assert classifier.sizing.overhead > 0
    assert (await classifier.prepare([email])).pending == (email,)


async def test_all_in_one_counts_ref_overhead(
    make_client: ClientFactory, questions: QuestionSet
) -> None:
    client = make_client(_unused)
    email = EmailFactory()
    batched = _classifier(client, questions, "all_in_one").input_tokens(email)
    assert batched > _classifier(client, questions).input_tokens(email)
