### Task 5: OpenRouter client

**Files:**
- Create: `src/jev_bench/openrouter.py`
- Modify: `tests/conftest.py` (add the async `make_client` fixture), `tests/factories.py` (add `ClientFactory`)
- Test: `tests/test_openrouter.py`

**Interfaces:**
- Consumes: `jev_bench.settings.Settings` (Task 1).
- Produces:
  - `jev_bench.openrouter`:
    - constants `CHAT_PATH = "/v1/chat/completions"`, `FATAL_STATUSES`, `RETRY_STATUSES`;
    - type `JsonObject = dict[str, Any]`;
    - `OpenRouterError(message, *, status=None, retryable=False)` with `.status`, `.retryable` and
      `.fatal` (status in {401, 402, 403});
    - `ChatContentError(ValueError)`;
    - `ApiResponse(body: JsonObject, latency_ms: float)`;
    - `OpenRouterClient(http, *, max_retries, retry_base_delay_s)`:
      - `await get_json(path, params=None) -> JsonObject` (no auth header);
      - `await post_json(path, body, *, api_key) -> ApiResponse`;
      - `await post_stream(path, body, *, api_key, on_text=None) -> ApiResponse` (SSE assembled into a
        non-stream chat body `{id, model, choices:[{message:{content}, finish_reason}], usage}`);
    - `build_http_client(settings) -> httpx2.AsyncClient`;
    - `chat_content(body) -> str` (raises `ChatContentError`);
    - `json_schema_format(name, schema) -> JsonObject`.
  - Tests: fixture `make_client(handler, *, max_retries=0) -> OpenRouterClient`, with base URL
    `https://openrouter.test/api`, zero retry delay, a `MockTransport`, and clients closed at teardown.

Review Focus #4 (malformed provider responses) is pinned here at the transport level: a 200 carrying an
error object, a non-JSON body, and mid-stream error chunks.

- [ ] **Step 1: Add the `make_client` fixture**

Append to `tests/factories.py`: add the imports at the top, and add
`ClientFactory: type of the make_client fixture` to its docstring.

```python
from collections.abc import Callable

from jev_bench.openrouter import OpenRouterClient

type ClientFactory = Callable[..., OpenRouterClient]
```

Append to `tests/conftest.py`: add the imports at the top, and the fixture at the end. Also add
`make_client: async OpenRouterClient factory` under `Fixtures:` in the module docstring.

```python
from collections.abc import AsyncIterator, Callable

import httpx2

from jev_bench.openrouter import OpenRouterClient
from tests.factories import ClientFactory

type Handler = Callable[[httpx2.Request], httpx2.Response]


@pytest.fixture
async def make_client() -> AsyncIterator[ClientFactory]:
    opened: list[httpx2.AsyncClient] = []

    def build(handler: Handler, *, max_retries: int = 0) -> OpenRouterClient:
        http = httpx2.AsyncClient(base_url="https://openrouter.test/api", transport=httpx2.MockTransport(handler))
        opened.append(http)
        return OpenRouterClient(http, max_retries=max_retries, retry_base_delay_s=0.0)

    yield build
    for http in opened:
        await http.aclose()
```

- [ ] **Step 2: Write the failing tests**

`tests/test_openrouter.py`:
````python
"""Tests for jev_bench.openrouter."""

import json
from collections.abc import Callable
from typing import Any

import httpx2
import pytest

from jev_bench.openrouter import (
    ChatContentError,
    OpenRouterError,
    build_http_client,
    chat_content,
    json_schema_format,
)
from jev_bench.settings import Settings
from tests.factories import ClientFactory

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
    return lambda request: httpx2.Response(200, content=content, headers={"content-type": "text/event-stream"})


class Script:
    def __init__(self, *steps: Step) -> None:
        self.steps = list(steps)
        self.requests: list[httpx2.Request] = []

    def __call__(self, request: httpx2.Request) -> httpx2.Response:
        self.requests.append(request)
        return self.steps.pop(0)(request)


async def test_post_json_sends_key_path_and_body(make_client: ClientFactory) -> None:
    script = Script(_json(200, {"ok": 1}))
    response = await make_client(script).post_json("/v1/embeddings", {"input": ["x"]}, api_key="sk-1")
    request = script.requests[0]
    assert response.body == {"ok": 1}
    assert response.latency_ms >= 0
    assert request.url.path == "/api/v1/embeddings"
    assert request.headers["Authorization"] == "Bearer sk-1"
    assert json.loads(request.content) == {"input": ["x"]}


@pytest.mark.parametrize(
    ("steps", "calls", "status", "retryable", "fatal"),
    [
        pytest.param([_json(400, {"error": {"message": "bad"}})], 1, 400, False, False, id="400-no-retry"),
        pytest.param([_json(401, {"error": {"message": "no auth"}})], 1, 401, False, True, id="401-fatal"),
        pytest.param([_json(402, {"error": {"message": "credits"}})], 1, 402, False, True, id="402-fatal"),
        pytest.param([_json(500, {})] * 3, 3, 500, True, False, id="500-exhausts-retries"),
        pytest.param([_text(503, "down")] * 3, 3, 503, True, False, id="503-text-body"),
        pytest.param([_connect_error] * 3, 3, None, True, False, id="transport-error"),
        pytest.param([_text(200, "not json")] * 3, 3, None, True, False, id="invalid-json-body"),
        pytest.param([_json(200, [1, 2])], 1, None, False, False, id="non-object-body"),
        pytest.param([_json(200, {"error": {"code": 401, "message": "x"}})], 1, 401, False, True, id="embedded-401"),
        pytest.param([_json(200, {"error": {"code": 429, "message": "slow"}})] * 3, 3, 429, True, False, id="embedded-429"),
        pytest.param([_json(200, {"error": {"code": "server_error"}})] * 3, 3, None, True, False, id="embedded-string-code"),
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
            _chunk(id="gen-1", model="anthropic/claude-sonnet-5", choices=[{"delta": {"content": '{"a":'}}]),
            _chunk(choices=[{"delta": {"content": "1}"}, "finish_reason": "stop"}]),
            _chunk(choices=[{"delta": {"content": ""}, "finish_reason": "stop"}], usage={"cost": 0.01, "prompt_tokens": 10}),
            "data: [DONE]",
        )
    )
    deltas: list[str] = []
    response = await make_client(script).post_stream("/v1/chat/completions", {"model": "m"}, api_key="k", on_text=deltas.append)
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
            _sse(_chunk(error={"code": 502, "message": "provider down"}, choices=[{"delta": {"content": ""}, "finish_reason": "error"}])),
            502,
            False,
            id="mid-stream-error",
        ),
        pytest.param(_json(401, {"error": {"message": "no auth"}}), 401, True, id="http-401"),
        pytest.param(_sse("data: {not json"), None, False, id="invalid-chunk"),
    ],
)
async def test_post_stream_errors(make_client: ClientFactory, step: Step, status: int | None, fatal: bool) -> None:
    with pytest.raises(OpenRouterError) as info:
        await make_client(Script(step)).post_stream("/v1/chat/completions", {}, api_key="k")
    assert info.value.status == status
    assert info.value.fatal is fatal


async def test_build_http_client_uses_settings() -> None:
    client = build_http_client(Settings(_env_file=None, request_timeout_s=5, connect_timeout_s=2))
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
````

- [ ] **Step 3: Run tests to verify they fail**

Run: `uv run pytest tests/test_openrouter.py -v`
Expected: `ImportError` / `ModuleNotFoundError` for `jev_bench.openrouter`.

- [ ] **Step 4: Write the implementation**

`src/jev_bench/openrouter.py`:
````python
"""Async OpenRouter client: JSON and SSE-stream POSTs, GETs, retries with jittered backoff, typed errors.

Constants:
    CHAT_PATH, FATAL_STATUSES, RETRY_STATUSES
Classes:
    OpenRouterError: HTTP / protocol failure with `status`, `retryable` and `fatal`.
    ChatContentError: a chat completion carries no usable assistant content.
    ApiResponse: parsed JSON body plus client-observed latency.
    OpenRouterClient: per-call API keys over one shared `httpx2.AsyncClient`.
Functions:
    build_http_client: shared client configured from Settings.
    chat_content: assistant text of a chat completion (code fences stripped).
    json_schema_format: strict `response_format` payload for a JSON schema.
"""

import asyncio
import json
import random
import re
import time
from collections.abc import Awaitable, Callable, Mapping
from typing import Any, NamedTuple

import httpx2
from loguru import logger

from jev_bench.settings import Settings

type JsonObject = dict[str, Any]
type TextCallback = Callable[[str], None]

CHAT_PATH = "/v1/chat/completions"
FATAL_STATUSES: frozenset[int] = frozenset({401, 402, 403})
RETRY_STATUSES: frozenset[int] = frozenset({408, 429, 500, 502, 503, 504, 524, 529})
_FENCE = re.compile(r"^```[A-Za-z0-9_-]*\s*(.*?)\s*```$", re.DOTALL)


class OpenRouterError(Exception):
    def __init__(self, message: str, *, status: int | None = None, retryable: bool = False) -> None:
        super().__init__(message)
        self.status = status
        self.retryable = retryable

    @property
    def fatal(self) -> bool:
        return self.status in FATAL_STATUSES


class ChatContentError(ValueError):
    pass


class ApiResponse(NamedTuple):
    body: JsonObject
    latency_ms: float


def build_http_client(settings: Settings) -> httpx2.AsyncClient:
    return httpx2.AsyncClient(
        base_url=settings.openrouter_base_url,
        timeout=httpx2.Timeout(settings.request_timeout_s, connect=settings.connect_timeout_s),
        limits=httpx2.Limits(max_connections=64),
        headers={"X-Title": "jev-bench"},
    )


class OpenRouterClient:
    def __init__(self, http: httpx2.AsyncClient, *, max_retries: int, retry_base_delay_s: float) -> None:
        self._http = http
        self._max_retries = max_retries
        self._base_delay = retry_base_delay_s

    async def get_json(self, path: str, params: Mapping[str, str] | None = None) -> JsonObject:
        response = await self._retrying(lambda: self._send(path, lambda: self._http.get(path, params=params)))
        return response.body

    async def post_json(self, path: str, body: Mapping[str, Any], *, api_key: str) -> ApiResponse:
        headers = _auth(api_key)
        return await self._retrying(
            lambda: self._send(path, lambda: self._http.post(path, json=body, headers=headers))
        )

    async def post_stream(
        self,
        path: str,
        body: Mapping[str, Any],
        *,
        api_key: str,
        on_text: TextCallback | None = None,
    ) -> ApiResponse:
        payload = {**body, "stream": True}
        return await self._retrying(lambda: self._stream_once(path, payload, api_key, on_text))

    async def _retrying(self, attempt: Callable[[], Awaitable[ApiResponse]]) -> ApiResponse:
        for number in range(self._max_retries + 1):
            try:
                return await attempt()
            except OpenRouterError as exc:
                if not exc.retryable or number == self._max_retries:
                    raise
                logger.bind(attempt=number + 1).debug("retrying OpenRouter call: {}", exc)
            await asyncio.sleep(self._base_delay * 2**number * random.uniform(0.5, 1.5))
        raise AssertionError("unreachable")

    async def _send(self, path: str, request: Callable[[], Awaitable[httpx2.Response]]) -> ApiResponse:
        started = time.perf_counter()
        try:
            response = await request()
        except httpx2.TransportError as exc:
            raise OpenRouterError(f"{path}: transport error: {exc!r}", retryable=True) from exc
        _raise_for_status(response)
        return ApiResponse(_json_body(response), _elapsed_ms(started))

    async def _stream_once(
        self,
        path: str,
        body: Mapping[str, Any],
        api_key: str,
        on_text: TextCallback | None,
    ) -> ApiResponse:
        started = time.perf_counter()
        accumulator = _StreamAccumulator(on_text)
        try:
            async with self._http.stream("POST", path, json=body, headers=_auth(api_key)) as response:
                await _raise_for_stream_status(response)
                async for line in response.aiter_lines():
                    accumulator.feed(line)
        except httpx2.TransportError as exc:
            raise OpenRouterError(f"{path}: transport error: {exc!r}", retryable=True) from exc
        return ApiResponse(accumulator.body(), _elapsed_ms(started))


class _StreamAccumulator:
    def __init__(self, on_text: TextCallback | None) -> None:
        self._on_text = on_text
        self._parts: list[str] = []
        self._meta: JsonObject = {}
        self._finish_reason: str | None = None
        self._usage: JsonObject = {}

    def feed(self, line: str) -> None:
        if not line.startswith("data:"):
            return
        payload = line.removeprefix("data:").strip()
        if payload != "[DONE]":
            self._absorb(_decode_chunk(payload))

    def _absorb(self, chunk: JsonObject) -> None:
        _raise_embedded_error(chunk)
        self._meta.update({key: chunk[key] for key in ("id", "model", "provider") if chunk.get(key)})
        self._usage = chunk.get("usage") or self._usage
        for choice in chunk.get("choices") or []:
            self._absorb_choice(choice)

    def _absorb_choice(self, choice: JsonObject) -> None:
        text = (choice.get("delta") or {}).get("content") or ""
        if text:
            self._parts.append(text)
            if self._on_text is not None:
                self._on_text(text)
        self._finish_reason = choice.get("finish_reason") or self._finish_reason

    def body(self) -> JsonObject:
        message = {"role": "assistant", "content": "".join(self._parts)}
        choice = {"message": message, "finish_reason": self._finish_reason}
        return {**self._meta, "choices": [choice], "usage": self._usage}


def _decode_chunk(payload: str) -> JsonObject:
    try:
        chunk = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise OpenRouterError(f"invalid stream chunk: {payload[:200]!r}", retryable=True) from exc
    if not isinstance(chunk, dict):
        raise OpenRouterError("stream chunk is not an object", retryable=True)
    return chunk


async def _raise_for_stream_status(response: httpx2.Response) -> None:
    if response.status_code >= 400:
        await response.aread()
        _raise_for_status(response)


def _auth(api_key: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {api_key}"}


def _elapsed_ms(started: float) -> float:
    return (time.perf_counter() - started) * 1000.0


def _raise_for_status(response: httpx2.Response) -> None:
    if response.status_code < 400:
        return
    raise OpenRouterError(
        f"HTTP {response.status_code}: {_error_message(response)}",
        status=response.status_code,
        retryable=response.status_code in RETRY_STATUSES,
    )


def _error_message(response: httpx2.Response) -> str:
    try:
        payload = response.json()
    except ValueError:
        return response.text[:500]
    error = payload.get("error") if isinstance(payload, dict) else None
    return str(error.get("message") or error) if isinstance(error, dict) else response.text[:500]


def _json_body(response: httpx2.Response) -> JsonObject:
    try:
        payload = response.json()
    except ValueError as exc:
        raise OpenRouterError(f"invalid JSON body: {response.text[:200]!r}", retryable=True) from exc
    if not isinstance(payload, dict):
        raise OpenRouterError("JSON body is not an object")
    _raise_embedded_error(payload)
    return payload


def _raise_embedded_error(payload: Mapping[str, Any]) -> None:
    error = payload.get("error")
    if not isinstance(error, dict):
        return
    code = error.get("code")
    status = code if isinstance(code, int) and not isinstance(code, bool) else None
    raise OpenRouterError(
        f"provider error: {error.get('message') or error}",
        status=status,
        retryable=status is None or status in RETRY_STATUSES,
    )


def chat_content(body: Mapping[str, Any]) -> str:
    choices = body.get("choices") or []
    if not choices:
        raise ChatContentError("response has no choices")
    choice = choices[0]
    if choice.get("finish_reason") == "length":
        raise ChatContentError("response truncated (finish_reason=length)")
    content = (choice.get("message") or {}).get("content")
    if not isinstance(content, str) or not content.strip():
        raise ChatContentError("empty response")
    match = _FENCE.match(content.strip())
    return match.group(1) if match else content.strip()


def json_schema_format(name: str, schema: JsonObject) -> JsonObject:
    return {"type": "json_schema", "json_schema": {"name": name, "strict": True, "schema": schema}}
````

- [ ] **Step 5: Run tests to verify they pass, lint, type-check**

Run: `uv run pytest tests/test_openrouter.py -v`
Expected: all PASS.

Run: `uv run ruff check --fix && uv run ruff format && uv run pyright`
Expected: `0 errors`.

- [ ] **Step 6: Commit**

```bash
git add src/jev_bench/openrouter.py tests/conftest.py tests/factories.py tests/test_openrouter.py
git commit -m "feat: OpenRouter client with retries, SSE streaming and typed errors

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```
