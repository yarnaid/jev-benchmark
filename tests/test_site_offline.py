"""Tests for jev_bench.site.offline: no network, no key, no OpenRouter model list."""

from pathlib import Path

import pytest
from tests.factories import mini_settings

from jev_bench.site.errors import ExportError
from jev_bench.site.offline import OfflineCatalog, offline_services, refusing_client


async def test_offline_services_hold_no_key_and_list_no_models(tmp_path: Path) -> None:
    async with refusing_client() as http:
        services = offline_services(mini_settings(tmp_path, api_key="sk-or-v1-SENTINEL"), http)
        column = services.benchmark_config().columns[0]
        assert isinstance(services.catalog, OfflineCatalog)
        assert await services.catalog.for_column(column) == []
    assert services.settings.server_api_key() is None
    assert services.settings.max_retries == 0


async def test_refusing_client_raises_on_any_request() -> None:
    async with refusing_client() as http:
        with pytest.raises(ExportError, match="must not reach the network"):
            await http.get("https://openrouter.ai/api/v1/models")
