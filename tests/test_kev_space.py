"""Tests for jev_bench.kev_space."""

import asyncio
import json
from collections.abc import AsyncIterator, Callable
from typing import Any

import httpx2
import pytest
from tests.factories import KEV_RESPONSE, SPACE_ASLEEP_PAGE, space_events, space_info

from jev_bench.kev_space import (
    KevSpaceClient,
    KevSpaceError,
    SpaceDecision,
    mask_hf_tokens,
    model_choices,
)
from jev_bench.openrouter import ApiResponse

type Step = Callable[[], httpx2.Response]
type SpaceFactory = Callable[..., KevSpaceClient]

_URL = "https://kev.test"
_TOKEN = "hf_SENTINELtoken4242"
_DECISION = SpaceDecision(
    state='{"subject": "Hi"}', questions='{"q": {}}', model="Kev-4B", calibrated=True
)
_COMPLETE = ("complete", ["<div>", KEV_RESPONSE, ""])


class Script:
    def __init__(self, *steps: Step) -> None:
        self.steps = list(steps)
        self.requests: list[httpx2.Request] = []

    def __call__(self, request: httpx2.Request) -> httpx2.Response:
        self.requests.append(request)
        return self.steps.pop(0)()


def _reply(status: int, **kwargs: Any) -> Step:
    return lambda: httpx2.Response(status, **kwargs)


def _submitted(event_id: str = "ev1") -> Step:
    return _reply(200, json={"event_id": event_id})


def _events(*events: tuple[str, object]) -> Step:
    return _raw(space_events(*events))


def _raw(content: bytes) -> Step:
    return _reply(200, content=content, headers={"content-type": "text/event-stream"})


def _connect_error() -> httpx2.Response:
    raise httpx2.ConnectError("connection refused")


@pytest.fixture
async def make_space() -> AsyncIterator[SpaceFactory]:
    opened: list[httpx2.AsyncClient] = []

    def build(handler: Callable[[httpx2.Request], Any], *, max_retries: int = 1) -> KevSpaceClient:
        http = httpx2.AsyncClient(
            base_url="https://openrouter.test/api", transport=httpx2.MockTransport(handler)
        )
        opened.append(http)
        return KevSpaceClient(http, max_retries=max_retries, retry_base_delay_s=0.0)

    yield build
    for http in opened:
        await http.aclose()


async def _decide(
    client: KevSpaceClient, *, token: str | None = None, timeout_s: float = 1.0
) -> ApiResponse:
    return await client.decide(_URL, "decide", _DECISION, token=token, timeout_s=timeout_s)


@pytest.mark.parametrize(
    ("steps", "calls"),
    [
        pytest.param(
            [_submitted(), _events(("heartbeat", None), _COMPLETE)], 2, id="heartbeat-then-complete"
        ),
        pytest.param(
            [_reply(503, text="<html>starting</html>"), _submitted(), _events(_COMPLETE)],
            3,
            id="503-then-success",
        ),
        pytest.param(
            [_submitted(), _events(("heartbeat", None)), _submitted(), _events(_COMPLETE)],
            4,
            id="stream-ended-early-then-success",
        ),
    ],
)
async def test_decide_returns_the_systemone_body(
    make_space: SpaceFactory, steps: list[Step], calls: int
) -> None:
    script = Script(*steps)
    response = await _decide(make_space(script))
    assert response.body == KEV_RESPONSE
    assert response.latency_ms >= 0
    assert len(script.requests) == calls


@pytest.mark.parametrize(
    ("token", "authorization"),
    [
        pytest.param(None, None, id="anonymous"),
        pytest.param(_TOKEN, f"Bearer {_TOKEN}", id="token-on-both-calls"),
    ],
)
async def test_decide_request_shape(
    make_space: SpaceFactory, token: str | None, authorization: str | None
) -> None:
    script = Script(_submitted("ev1"), _events(_COMPLETE))
    await _decide(make_space(script), token=token)
    post, get = script.requests
    assert (post.method, str(post.url)) == ("POST", "https://kev.test/gradio_api/call/decide")
    assert json.loads(post.content) == {
        "data": ['{"subject": "Hi"}', '{"q": {}}', "Kev-4B", True, False, False, 4]
    }
    assert (get.method, str(get.url)) == ("GET", "https://kev.test/gradio_api/call/decide/ev1")
    assert [post.headers.get("authorization"), get.headers.get("authorization")] == [
        authorization,
        authorization,
    ]


_HTML = {"content-type": "text/html; charset=utf-8"}
_ASLEEP = r"HTML page \(asleep, restarting or down\); open https://kev.test to wake it"
_QUOTA = "You have exceeded your GPU quota (15s requested vs. 3s left). Try again in 0:10:00"


@pytest.mark.parametrize(
    ("steps", "calls", "message", "retryable", "fatal"),
    [
        pytest.param(
            [_submitted(), _events(("error", {"error": "State too long", "title": "Error"}))],
            2,
            "Kev Space: State too long",
            False,
            False,
            id="error-event-fails-the-email",
        ),
        pytest.param(
            [_submitted(), _events(("error", {"error": _QUOTA}))],
            2,
            "GPU quota",
            False,
            True,
            id="quota-error-is-fatal",
        ),
        pytest.param(
            [_submitted(), _events(("error", None))],
            2,
            "the Space reported an error",
            False,
            False,
            id="error-event-without-message",
        ),
        pytest.param(
            [_submitted(), _events(("complete", ["only html"]))],
            2,
            "malformed complete event",
            False,
            False,
            id="complete-without-response",
        ),
        pytest.param(
            [_submitted(), _raw(b"event: complete\ndata: {not json\n\n")],
            2,
            "invalid event data",
            False,
            False,
            id="complete-not-json",
        ),
        pytest.param(
            [_submitted(), _events(("heartbeat", None))] * 2,
            4,
            "ended without a result",
            True,
            False,
            id="stream-ends-early-twice",
        ),
        pytest.param(
            [_reply(401, text=f"Invalid credentials {_TOKEN}")],
            1,
            "HTTP 401",
            False,
            True,
            id="401-is-fatal",
        ),
        pytest.param(
            [_reply(404, text="Not Found")], 1, "HTTP 404", False, True, id="404-is-fatal"
        ),
        pytest.param(
            [_reply(503, text="<html>starting</html>")] * 2,
            2,
            "Kev Space call: HTTP 503",
            True,
            False,
            id="503-exhausts-retries",
        ),
        pytest.param(
            [_submitted(), _reply(500, text="boom")] * 2,
            4,
            "Kev Space result: HTTP 500",
            True,
            False,
            id="result-500-exhausts-retries",
        ),
        pytest.param([_connect_error] * 2, 2, "transport error", True, False, id="transport-error"),
        pytest.param(
            [_reply(200, json={"no": "id"})],
            1,
            "no usable event id",
            False,
            False,
            id="no-event-id",
        ),
        pytest.param(
            [_reply(200, json={"event_id": "../x"})],
            1,
            "no usable event id",
            False,
            False,
            id="unsafe-event-id",
        ),
        pytest.param(
            [_reply(200, text="<html>")] * 2,
            2,
            "not JSON",
            True,
            False,
            id="call-body-not-json",
        ),
        pytest.param(
            [_reply(503, text=SPACE_ASLEEP_PAGE, headers=_HTML)] * 2,
            2,
            _ASLEEP,
            True,
            False,
            id="call-on-a-sleeping-space",
        ),
    ],
)
async def test_decide_errors(
    make_space: SpaceFactory,
    steps: list[Step],
    calls: int,
    message: str,
    retryable: bool,
    fatal: bool,
) -> None:
    script = Script(*steps)
    with pytest.raises(KevSpaceError, match=message) as info:
        await _decide(make_space(script), token=_TOKEN)
    assert (info.value.retryable, info.value.fatal) == (retryable, fatal)
    assert _TOKEN not in str(info.value)
    assert "<" not in str(info.value)
    assert len(script.requests) == calls


async def test_decide_timeout_is_an_email_error_and_not_retried(make_space: SpaceFactory) -> None:
    requests: list[httpx2.Request] = []

    async def stuck(request: httpx2.Request) -> httpx2.Response:
        requests.append(request)
        await asyncio.sleep(1)
        return httpx2.Response(200, json={"event_id": "late"})

    with pytest.raises(KevSpaceError, match=r"no answer within 0\.02 s") as info:
        await _decide(make_space(stuck), timeout_s=0.02)
    assert (info.value.retryable, info.value.fatal, len(requests)) == (False, False, 1)


async def test_info_reads_the_api_schema(make_space: SpaceFactory) -> None:
    script = Script(_reply(200, json=space_info(("Kev-4B",))))
    info = await make_space(script).info(_URL, token=_TOKEN)
    assert model_choices(info, "decide") == ("Kev-4B", "Both")
    (request,) = script.requests
    assert (str(request.url), request.headers["authorization"]) == (
        "https://kev.test/gradio_api/info",
        f"Bearer {_TOKEN}",
    )


@pytest.mark.parametrize(
    ("steps", "calls", "message"),
    [
        pytest.param(
            [_reply(503, text="<html>sleeping</html>")] * 2, 2, "HTTP 503", id="space-asleep"
        ),
        pytest.param([_reply(200, text="<html>")] * 2, 2, "not JSON", id="html-body"),
        pytest.param(
            [_reply(503, text=SPACE_ASLEEP_PAGE, headers=_HTML)] * 2,
            2,
            _ASLEEP,
            id="realistic-sleeping-space-page",
        ),
        pytest.param([_reply(200, text="oops")] * 2, 2, "not JSON: 'oops'", id="plain-text-body"),
        pytest.param([_reply(200, json=[1])], 1, "not an object", id="non-object-body"),
        pytest.param([_connect_error] * 2, 2, "transport error", id="transport-error"),
    ],
)
async def test_info_errors(
    make_space: SpaceFactory, steps: list[Step], calls: int, message: str
) -> None:
    script = Script(*steps)
    with pytest.raises(KevSpaceError, match=message) as info:
        await make_space(script).info(_URL, token=None)
    assert "<" not in str(info.value)
    assert len(script.requests) == calls


@pytest.mark.parametrize(
    ("info", "expected"),
    [
        pytest.param(
            space_info(("Kev-4B", "Kev-0.8B")), ("Kev-4B", "Kev-0.8B", "Both"), id="live-shape"
        ),
        pytest.param({"named_endpoints": {}}, None, id="no-endpoint"),
        pytest.param({}, None, id="no-endpoints"),
        pytest.param(
            {"named_endpoints": {"/decide": {"parameters": []}}}, (), id="no-model-parameter"
        ),
        pytest.param(
            {"named_endpoints": {"/decide": {"parameters": [{"parameter_name": "model_choice"}]}}},
            (),
            id="no-enum",
        ),
        pytest.param(
            {
                "named_endpoints": {
                    "/decide": {
                        "parameters": [
                            {"parameter_name": "model_choice", "type": {"enum": ["a", 1]}}
                        ]
                    }
                }
            },
            ("a",),
            id="non-string-choices-dropped",
        ),
    ],
)
def test_model_choices(info: dict[str, Any], expected: tuple[str, ...] | None) -> None:
    assert model_choices(info, "decide") == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        pytest.param("bad hf_abcdefgh1234 here", "bad hf_…1234 here", id="token"),
        pytest.param("hf_short", "hf_short", id="too-short-to-be-a-token"),
        pytest.param("no token", "no token", id="plain"),
    ],
)
def test_mask_hf_tokens(text: str, expected: str) -> None:
    assert mask_hf_tokens(text) == expected
