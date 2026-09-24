"""Tests for jev_bench.openrouter."""

import json
from collections.abc import Callable
from typing import Any

import httpx2
import pytest
from tests.factories import ClientFactory

from jev_bench.openrouter import (
    ChatContentError,
    OpenRouterError,
    build_http_client,
    chat_content,
    json_schema_format,
)
from jev_bench.settings import Settings

type Step = Callable[[httpx2.Request], httpx2.Response]


def _json(status: int, payload: object) -> Step:
    return lambda request: httpx2.Response(status, json=payload)


def _text(status: int, text: str) -> Step:
    return lambda request: httpx2.Response(status, text=text)


def _connect_error(request: httpx2.Request) -> httpx2.Response:
    raise httpx2.ConnectError("boom", request=request)


def _chunk(**fields: Any) -> str:
    return "data: " + json.dumps(fields)


def _sse(*events: str) -> Step:
    content = "".join(f"{event}\n\n" for event in events).encode()
    return lambda request: httpx2.Response(
        200, content=content, headers={"content-type": "text/event-stream"}
    )


class Script:
    def __init__(self, *steps: Step) -> None:
        self.steps = list(steps)
        self.requests: list[httpx2.Request] = []

    def __call__(self, request: httpx2.Request) -> httpx2.Response:
        self.requests.append(request)
        return self.steps.pop(0)(request)


async def test_post_json_sends_key_path_and_body(make_client: ClientFactory) -> None:
    script = Script(_json(200, {"ok": 1}))
    response = await make_client(script).post_json(
        "/v1/embeddings", {"input": ["x"]}, api_key="sk-1"
    )
    request = script.requests[0]
    assert response.body == {"ok": 1}
    assert response.latency_ms >= 0
    assert request.url.path == "/api/v1/embeddings"
    assert request.headers["Authorization"] == "Bearer sk-1"
    assert json.loads(request.content) == {"input": ["x"]}


@pytest.mark.parametrize(
    ("steps", "calls", "status", "retryable", "fatal"),
    [
        pytest.param(
            [_json(400, {"error": {"message": "bad"}})], 1, 400, False, False, id="400-no-retry"
        ),
        pytest.param(
            [_json(401, {"error": {"message": "no auth"}})], 1, 401, False, True, id="401-fatal"
        ),
        pytest.param(
            [_json(402, {"error": {"message": "credits"}})], 1, 402, False, True, id="402-fatal"
        ),
        pytest.param([_json(500, {})] * 3, 3, 500, True, False, id="500-exhausts-retries"),
        pytest.param([_text(503, "down")] * 3, 3, 503, True, False, id="503-text-body"),
        pytest.param([_connect_error] * 3, 3, None, True, False, id="transport-error"),
        pytest.param([_text(200, "not json")] * 3, 3, None, True, False, id="invalid-json-body"),
        pytest.param([_json(200, [1, 2])], 1, None, False, False, id="non-object-body"),
        pytest.param(
            [_json(200, {"error": {"code": 401, "message": "x"}})],
            1,
            401,
            False,
            True,
            id="embedded-401",
        ),
        pytest.param(
            [_json(200, {"error": {"code": 429, "message": "slow"}})] * 3,
            3,
            429,
            True,
            False,
            id="embedded-429",
        ),
        pytest.param(
            [_json(200, {"error": {"code": "server_error"}})] * 3,
            3,
            None,
            True,
            False,
            id="embedded-string-code",
        ),
    ],
)
async def test_post_json_errors(
    make_client: ClientFactory,
    steps: list[Step],
    calls: int,
    status: int | None,
    retryable: bool,
    fatal: bool,
) -> None:
    script = Script(*steps)
    with pytest.raises(OpenRouterError) as info:
        await make_client(script, max_retries=2).post_json("/v1/x", {}, api_key="k")
    assert len(script.requests) == calls
    assert info.value.status == status
    assert info.value.retryable is retryable
    assert info.value.fatal is fatal


@pytest.mark.parametrize(
    "first",
    [
        pytest.param(_json(429, {}), id="after-429"),
        pytest.param(_connect_error, id="after-transport-error"),
    ],
)
async def test_post_json_retries_then_succeeds(make_client: ClientFactory, first: Step) -> None:
    script = Script(first, _json(200, {"ok": True}))
    response = await make_client(script, max_retries=1).post_json("/v1/x", {}, api_key="k")
    assert response.body == {"ok": True}
    assert len(script.requests) == 2


async def test_get_json_passes_params_without_auth(make_client: ClientFactory) -> None:
    script = Script(_json(200, {"data": []}))
    body = await make_client(script).get_json("/v1/models", {"output_modalities": "decisions"})
    request = script.requests[0]
    assert body == {"data": []}
    assert request.url.params["output_modalities"] == "decisions"
    assert "Authorization" not in request.headers


async def test_post_stream_assembles_a_chat_body(make_client: ClientFactory) -> None:
    script = Script(
        _sse(
            ": OPENROUTER PROCESSING",
            _chunk(
                id="gen-1",
                model="anthropic/claude-sonnet-5",
                choices=[{"delta": {"content": '{"a":'}}],
            ),
            _chunk(choices=[{"delta": {"content": "1}"}, "finish_reason": "stop"}]),
            _chunk(
                choices=[{"delta": {"content": ""}, "finish_reason": "stop"}],
                usage={"cost": 0.01, "prompt_tokens": 10},
            ),
            "data: [DONE]",
        )
    )
    deltas: list[str] = []
    response = await make_client(script).post_stream(
        "/v1/chat/completions", {"model": "m"}, api_key="k", on_text=deltas.append
    )
    assert json.loads(script.requests[0].content) == {"model": "m", "stream": True}
    assert response.body["choices"][0]["message"]["content"] == '{"a":1}'
    assert response.body["choices"][0]["finish_reason"] == "stop"
    assert response.body["usage"] == {"cost": 0.01, "prompt_tokens": 10}
    assert response.body["model"] == "anthropic/claude-sonnet-5"
    assert deltas == ['{"a":', "1}"]


async def test_post_stream_accepts_data_without_space(make_client: ClientFactory) -> None:
    script = Script(_sse('data:{"choices":[{"delta":{"content":"x"},"finish_reason":"stop"}]}'))
    response = await make_client(script).post_stream("/v1/chat/completions", {}, api_key="k")
    assert response.body["choices"][0]["message"]["content"] == "x"


@pytest.mark.parametrize(
    ("step", "status", "fatal"),
    [
        pytest.param(
            _sse(
                _chunk(
                    error={"code": 502, "message": "provider down"},
                    choices=[{"delta": {"content": ""}, "finish_reason": "error"}],
                )
            ),
            502,
            False,
            id="mid-stream-error",
        ),
        pytest.param(_json(401, {"error": {"message": "no auth"}}), 401, True, id="http-401"),
        pytest.param(_sse("data: {not json"), None, False, id="invalid-chunk"),
    ],
)
async def test_post_stream_errors(
    make_client: ClientFactory, step: Step, status: int | None, fatal: bool
) -> None:
    with pytest.raises(OpenRouterError) as info:
        await make_client(Script(step)).post_stream("/v1/chat/completions", {}, api_key="k")
    assert info.value.status == status
    assert info.value.fatal is fatal


async def test_build_http_client_uses_settings() -> None:
    client = build_http_client(
        Settings.model_validate({"request_timeout_s": 5, "connect_timeout_s": 2})
    )
    try:
        assert str(client.base_url).rstrip("/") == "https://openrouter.ai/api"
        assert client.timeout.read == 5
        assert client.timeout.connect == 2
    finally:
        await client.aclose()


def _body(content: object, finish_reason: str | None = "stop") -> dict[str, Any]:
    return {"choices": [{"message": {"content": content}, "finish_reason": finish_reason}]}


@pytest.mark.parametrize(
    ("body", "expected"),
    [
        pytest.param(_body('{"a": 1}'), '{"a": 1}', id="plain"),
        pytest.param(_body('```json\n{"a": 1}\n```'), '{"a": 1}', id="json-fence"),
        pytest.param(_body('```{"a": 1}```'), '{"a": 1}', id="bare-fence"),
        pytest.param(_body('  {"a": 1}\n'), '{"a": 1}', id="whitespace"),
    ],
)
def test_chat_content(body: dict[str, Any], expected: str) -> None:
    assert chat_content(body) == expected


@pytest.mark.parametrize(
    ("body", "message"),
    [
        pytest.param({"choices": []}, "no choices", id="no-choices"),
        pytest.param(_body('{"a"', "length"), "truncated", id="truncated"),
        pytest.param(_body("   "), "empty", id="blank"),
        pytest.param(_body(None), "empty", id="none"),
        pytest.param({"choices": [None]}, "not an object", id="choice-not-object"),
        pytest.param({"choices": [{"message": "x"}]}, "not an object", id="message-not-object"),
    ],
)
def test_chat_content_errors(body: dict[str, Any], message: str) -> None:
    with pytest.raises(ChatContentError, match=message):
        chat_content(body)


def test_json_schema_format() -> None:
    assert json_schema_format("x", {"type": "object"}) == {
        "type": "json_schema",
        "json_schema": {"name": "x", "strict": True, "schema": {"type": "object"}},
    }
