"""Async OpenRouter client: JSON/SSE-stream POSTs, GETs, retries with jittered backoff, errors.

Constants:
    CHAT_PATH, RETRY_STATUSES
    FATAL_STATUSES: abort the whole job: 401/402/403 (key or credits) and 404 (unknown model, or
        no provider can serve the requested parameters; every request would fail the same way).
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

__all__ = [
    "CHAT_PATH",
    "FATAL_STATUSES",
    "RETRY_STATUSES",
    "ApiResponse",
    "ChatContentError",
    "JsonObject",
    "OpenRouterClient",
    "OpenRouterError",
    "TextCallback",
    "build_http_client",
    "chat_content",
    "json_schema_format",
]

type JsonObject = dict[str, Any]
type TextCallback = Callable[[str], None]

CHAT_PATH = "/v1/chat/completions"
FATAL_STATUSES: frozenset[int] = frozenset({401, 402, 403, 404})
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
    def __init__(
        self, http: httpx2.AsyncClient, *, max_retries: int, retry_base_delay_s: float
    ) -> None:
        self._http = http
        self._max_retries = max_retries
        self._base_delay = retry_base_delay_s

    async def get_json(self, path: str, params: Mapping[str, str] | None = None) -> JsonObject:
        response = await self._retrying(
            lambda: self._send(path, lambda: self._http.get(path, params=params))
        )
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

    async def _send(
        self, path: str, request: Callable[[], Awaitable[httpx2.Response]]
    ) -> ApiResponse:
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
            stream = self._http.stream("POST", path, json=body, headers=_auth(api_key))
            async with stream as response:
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
        keys = ("id", "model", "provider")
        self._meta.update({key: chunk[key] for key in keys if chunk.get(key)})
        self._usage = chunk.get("usage") or self._usage
        choices = chunk.get("choices") or []
        if not isinstance(choices, list):
            raise OpenRouterError("malformed stream chunk: choices is not a list", retryable=True)
        for choice in choices:
            self._absorb_choice(choice)

    def _absorb_choice(self, choice: object) -> None:
        if not isinstance(choice, Mapping):
            raise OpenRouterError("malformed stream chunk: choice is not an object", retryable=True)
        delta = choice.get("delta") or {}
        if not isinstance(delta, Mapping):
            raise OpenRouterError("malformed stream chunk: delta is not an object", retryable=True)
        text = delta.get("content") or ""
        if not isinstance(text, str):
            raise OpenRouterError("malformed stream chunk: content is not a string", retryable=True)
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
        message = f"invalid JSON body: {response.text[:200]!r}"
        raise OpenRouterError(message, retryable=True) from exc
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
    choices = body.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ChatContentError("response has no choices")
    choice = choices[0]
    if not isinstance(choice, Mapping):
        raise ChatContentError("choice is not an object")
    if choice.get("finish_reason") == "length":
        raise ChatContentError("response truncated (finish_reason=length)")
    message = choice.get("message")
    if not isinstance(message, Mapping):
        raise ChatContentError("message is not an object")
    content = message.get("content")
    if not isinstance(content, str) or not content.strip():
        raise ChatContentError("empty response")
    match = _FENCE.match(content.strip())
    return match.group(1) if match else content.strip()


def json_schema_format(name: str, schema: JsonObject) -> JsonObject:
    return {"type": "json_schema", "json_schema": {"name": name, "strict": True, "schema": schema}}
