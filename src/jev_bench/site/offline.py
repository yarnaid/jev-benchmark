"""Offline services for the snapshot export: nothing reaches the network and no key is held.

Classes:
    OfflineCatalog: a Catalog listing no OpenRouter models; the catalog route then offers each
        column's default model only.
Functions:
    refusing_client: an httpx2 client whose transport raises ExportError on any request.
    offline_settings: the given Settings without an OpenRouter key and without retries.
    offline_services: Services over offline_settings and the given client, with OfflineCatalog.
"""

import httpx2

from jev_bench.benchmark_config import ColumnConfig
from jev_bench.catalog import Catalog, ModelInfo
from jev_bench.services import Services
from jev_bench.settings import Settings
from jev_bench.site.errors import ExportError

__all__ = [
    "OfflineCatalog",
    "offline_services",
    "offline_settings",
    "refusing_client",
]


class OfflineCatalog(Catalog):
    async def for_column(self, column: ColumnConfig) -> list[ModelInfo]:
        return []


def _refuse(request: httpx2.Request) -> httpx2.Response:
    raise ExportError(f"the export must not reach the network: {request.url}")


def refusing_client() -> httpx2.AsyncClient:
    return httpx2.AsyncClient(transport=httpx2.MockTransport(_refuse))


def offline_settings(settings: Settings) -> Settings:
    return settings.model_copy(update={"openrouter_api_key": None, "max_retries": 0})


def offline_services(settings: Settings, http: httpx2.AsyncClient) -> Services:
    services = Services(offline_settings(settings), http)
    services.catalog = OfflineCatalog(services.client)
    return services
