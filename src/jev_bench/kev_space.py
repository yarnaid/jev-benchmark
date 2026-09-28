"""Client for Kev's public Hugging Face Space: the Gradio REST API on ZeroGPU.

A decision is two calls. `POST {space}/gradio_api/call/{api_name}` with the endpoint's positional
`data` returns an event id; `GET .../{event_id}` streams server-sent events until `complete`
(`data[1]` is the `/v1/systemone` response) or `error` (`data` is `{"error": message, ...}`);
`heartbeat` events are ignored. An HF token, when given, is sent as a bearer token on every call.
A whole attempt is bounded by `timeout_s`; a timeout fails the email and is not retried.

Constants:
    INFO_PATH, CALL_PATH
Classes:
    KevSpaceError: an OpenRouterError, so the runner's retry and fatal handling apply unchanged.
        Its message is masked for `hf_…` tokens; `fatal` also holds for an exhausted GPU quota.
    SpaceDecision: the arguments of one decide call; `data()` is the positional payload.
    KevSpaceClient: `info` and `decide` over the shared httpx2 client, with absolute URLs.
Functions:
    model_choices: the endpoint's `model_choice` enum from `/gradio_api/info` (None: no endpoint).
    mask_hf_tokens: hides `hf_…` tokens in text, keeping the last four characters.
"""

import asyncio
import json
import re
import time
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from typing import Any, NamedTuple

import httpx2

from jev_bench.openrouter import (
    RETRY_STATUSES,
    ApiResponse,
    JsonObject,
    OpenRouterError,
    with_retries,
)

__all__ = [
    "CALL_PATH",
    "INFO_PATH",
    "KevSpaceClient",
    "KevSpaceError",
    "SpaceDecision",
    "mask_hf_tokens",
    "model_choices",
]

INFO_PATH = "/gradio_api/info"
CALL_PATH = "/gradio_api/call"
_HF_TOKEN = re.compile(r"\bhf_[A-Za-z0-9]{8,}")
_EVENT_ID = re.compile(r"^[A-Za-z0-9_-]{1,128}$")
_RESULT_EVENTS = frozenset({"complete", "error"})


def mask_hf_tokens(text: str) -> str:
    return _HF_TOKEN.sub(lambda match: f"hf_…{match.group(0)[-4:]}", text)


class KevSpaceError(OpenRouterError):
    def __init__(
        self,
        message: str,
        *,
        status: int | None = None,
        retryable: bool = False,
        fatal: bool = False,
    ) -> None:
        super().__init__(mask_hf_tokens(message), status=status, retryable=retryable)
        self._fatal = fatal

    @property
    def fatal(self) -> bool:
        return self._fatal or super().fatal


class SpaceDecision(NamedTuple):
    state: str
    questions: str
    model: str
    calibrated: bool

    def data(self) -> list[object]:
        return [self.state, self.questions, self.model, self.calibrated, False, False, 4]


class KevSpaceClient:
    def __init__(
        self, http: httpx2.AsyncClient, *, max_retries: int, retry_base_delay_s: float
    ) -> None:
        self._http = http
        self._max_retries = max_retries
        self._base_delay = retry_base_delay_s

    async def info(self, space_url: str, *, token: str | None) -> JsonObject:
        return await self._retrying(lambda: self._info_once(space_url, token))

    async def decide(
        self,
        space_url: str,
        api_name: str,
        decision: SpaceDecision,
        *,
        token: str | None,
        timeout_s: float,
    ) -> ApiResponse:
        url = f"{space_url}{CALL_PATH}/{api_name}"
        return await self._retrying(lambda: self._decide_once(url, decision, token, timeout_s))

    async def _retrying[T](self, attempt: Callable[[], Awaitable[T]]) -> T:
        return await with_retries(
            attempt, max_retries=self._max_retries, base_delay_s=self._base_delay
        )

    async def _info_once(self, space_url: str, token: str | None) -> JsonObject:
        response = await _sent(self._http.get(f"{space_url}{INFO_PATH}", headers=_auth(token)))
        _check_status(response, "info")
        return _body(response)

    async def _decide_once(
        self, url: str, decision: SpaceDecision, token: str | None, timeout_s: float
    ) -> ApiResponse:
        started = time.perf_counter()
        try:
            async with asyncio.timeout(timeout_s):
                event_id = await self._submit(url, decision, _auth(token))
                body = await self._result(f"{url}/{event_id}", _auth(token))
        except TimeoutError as exc:
            raise KevSpaceError(f"Kev Space: no answer within {timeout_s:g} s") from exc
        return ApiResponse(body, (time.perf_counter() - started) * 1000.0)

    async def _submit(self, url: str, decision: SpaceDecision, headers: dict[str, str]) -> str:
        payload = {"data": decision.data()}
        response = await _sent(self._http.post(url, json=payload, headers=headers))
        _check_status(response, "call")
        event_id = _body(response).get("event_id")
        if not isinstance(event_id, str) or not _EVENT_ID.match(event_id):
            raise KevSpaceError("Kev Space: the call returned no usable event id")
        return event_id

    async def _result(self, url: str, headers: dict[str, str]) -> JsonObject:
        try:
            async with self._http.stream("GET", url, headers=headers) as response:
                if response.status_code >= 400:
                    await response.aread()
                    _check_status(response, "result")
                return await _read_events(response.aiter_lines())
        except httpx2.TransportError as exc:
            raise _transport_error(exc) from exc


def model_choices(info: Mapping[str, Any], api_name: str) -> tuple[str, ...] | None:
    endpoints = info.get("named_endpoints")
    endpoint = endpoints.get(f"/{api_name}") if isinstance(endpoints, dict) else None
    if not isinstance(endpoint, dict):
        return None
    return _enum(_parameter(endpoint.get("parameters"), "model_choice"))


def _parameter(parameters: object, name: str) -> Mapping[str, Any]:
    items = parameters if isinstance(parameters, list) else []
    found = (
        item for item in items if isinstance(item, dict) and item.get("parameter_name") == name
    )
    return next(found, {})


def _enum(parameter: Mapping[str, Any]) -> tuple[str, ...]:
    kind = parameter.get("type")
    values = kind.get("enum") if isinstance(kind, dict) else None
    if not isinstance(values, list):
        return ()
    return tuple(value for value in values if isinstance(value, str))


async def _read_events(lines: AsyncIterator[str]) -> JsonObject:
    event = ""
    async for line in lines:
        if line.startswith("event:"):
            event = line.removeprefix("event:").strip()
        elif line.startswith("data:") and event in _RESULT_EVENTS:
            return _outcome(event, line.removeprefix("data:").strip())
    raise KevSpaceError("Kev Space: the event stream ended without a result", retryable=True)


def _outcome(event: str, payload: str) -> JsonObject:
    data = _decode(payload)
    if event == "error":
        raise _event_error(data)
    response = data[1] if isinstance(data, list) and len(data) > 1 else None
    if not isinstance(response, dict):
        raise KevSpaceError("Kev Space: malformed complete event")
    return response


def _event_error(data: object) -> KevSpaceError:
    message = data.get("error") if isinstance(data, dict) else None
    text = message if isinstance(message, str) and message else "the Space reported an error"
    return KevSpaceError(f"Kev Space: {text}", fatal="quota" in text.lower())


def _decode(payload: str) -> object:
    try:
        return json.loads(payload)
    except json.JSONDecodeError as exc:
        raise KevSpaceError(f"Kev Space: invalid event data {payload[:200]!r}") from exc


async def _sent(request: Awaitable[httpx2.Response]) -> httpx2.Response:
    try:
        return await request
    except httpx2.TransportError as exc:
        raise _transport_error(exc) from exc


def _transport_error(exc: httpx2.TransportError) -> KevSpaceError:
    return KevSpaceError(f"Kev Space: transport error: {exc!r}", retryable=True)


def _check_status(response: httpx2.Response, step: str) -> None:
    if response.status_code < 400:
        return
    raise KevSpaceError(
        f"Kev Space {step}: HTTP {response.status_code}: {response.text[:300]}",
        status=response.status_code,
        retryable=response.status_code in RETRY_STATUSES,
    )


def _body(response: httpx2.Response) -> JsonObject:
    try:
        payload = response.json()
    except ValueError as exc:
        text = response.text[:200]
        raise KevSpaceError(f"Kev Space: invalid JSON {text!r}", retryable=True) from exc
    if not isinstance(payload, dict):
        raise KevSpaceError("Kev Space: the JSON body is not an object")
    return payload


def _auth(token: str | None) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"} if token else {}
