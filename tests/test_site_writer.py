"""Tests for jev_bench.site.writer."""

import asyncio
from pathlib import Path

import httpx2
import pytest

from jev_bench.site.errors import ExportError
from jev_bench.site.plan import SiteRequest
from jev_bench.site.writer import SiteWriter

BODY = b'{"ok": true}'


def _client(seen: list[str], status: int = 200, delay_s: float = 0.0) -> httpx2.AsyncClient:
    async def handler(request: httpx2.Request) -> httpx2.Response:
        seen.append(str(request.url))
        await asyncio.sleep(delay_s)
        return httpx2.Response(status, content=BODY)

    transport = httpx2.MockTransport(handler)
    return httpx2.AsyncClient(transport=transport, base_url="http://snapshot.invalid")


async def test_write_fetches_under_api_and_stores_the_body(tmp_path: Path) -> None:
    seen: list[str] = []
    request = SiteRequest(
        path="/compare", params={"runs": "r1,r2"}, file="api/compare/v1/t-default.json"
    )
    async with _client(seen) as client:
        response = await SiteWriter(client, tmp_path).write(request)
    assert response.json() == {"ok": True}
    assert seen == ["http://snapshot.invalid/api/compare?runs=r1%2Cr2"]
    assert (tmp_path / "api" / "compare" / "v1" / "t-default.json").read_bytes() == BODY


@pytest.mark.parametrize(
    ("status", "delay_s", "file", "message"),
    [
        pytest.param(404, 0.0, "api/x.json", "answered 404", id="not-200"),
        pytest.param(200, 1.0, "api/x.json", "timed out", id="timeout"),
        pytest.param(200, 0.0, "../escape.json", "outside the snapshot", id="path-escape"),
    ],
)
async def test_write_refuses(
    tmp_path: Path, status: int, delay_s: float, file: str, message: str
) -> None:
    async with _client([], status, delay_s) as client:
        writer = SiteWriter(client, tmp_path / "out", timeout_s=0.01)
        with pytest.raises(ExportError, match=message):
            await writer.write(SiteRequest(path="/x", file=file))
    assert not (tmp_path / "escape.json").exists()
