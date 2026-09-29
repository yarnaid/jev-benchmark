"""Writer of the static snapshot: fetch one planned GET from the in-process app and store its body.

Classes:
    SiteWriter: `write` fetches a SiteRequest under /api within a timeout and stores the response
        body unchanged under the request's file (any status but 200 is an ExportError); `put`
        stores bytes, refusing any path outside the snapshot.
"""

import asyncio
from pathlib import Path

import httpx2

from jev_bench.site.errors import ExportError
from jev_bench.site.plan import SiteRequest

__all__ = [
    "SiteWriter",
]


class SiteWriter:
    def __init__(self, client: httpx2.AsyncClient, out: Path, *, timeout_s: float = 60.0) -> None:
        self._client = client
        self._out = out.resolve()
        self._timeout_s = timeout_s

    async def write(self, request: SiteRequest) -> httpx2.Response:
        response = await self._get(request)
        if response.status_code != 200:
            detail = f"GET /api{request.path} {request.params} answered {response.status_code}"
            raise ExportError(detail)
        self.put(request.file, response.content)
        return response

    async def _get(self, request: SiteRequest) -> httpx2.Response:
        try:
            async with asyncio.timeout(self._timeout_s):
                return await self._client.get(f"/api{request.path}", params=request.params)
        except TimeoutError as exc:
            detail = f"GET /api{request.path} timed out after {self._timeout_s} s"
            raise ExportError(detail) from exc

    def put(self, file: str, content: bytes) -> None:
        target = (self._out / file).resolve()
        if not target.is_relative_to(self._out):
            raise ExportError(f"refusing to write outside the snapshot: {file}")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
