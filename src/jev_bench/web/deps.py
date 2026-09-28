"""FastAPI dependencies: the shared Services and the API key a job should use.

Constants:
    NO_KEY_DETAIL, HF_FORMAT_DETAIL
Types:
    ServicesDep, ApiKeyDep, OptionalApiKeyDep, HfTokenDep
    ThresholdQuery: optional `?threshold=` in (0, 1] (422 otherwise, NaN and infinity included).
Functions:
    get_services: Services stored on the app by the lifespan.
    optional_api_key: the X-OpenRouter-Key header, else the server key, else None.
    require_api_key: optional_api_key, or HTTP 400.
    hf_token: the X-HF-Token header, else the server HF_TOKEN, else None; HTTP 400 when it is not
        an `hf_…` token (so another secret pasted into the field never reaches the Space).
    split_ids: comma-separated id list, trimmed, de-duplicated, order kept.
"""

import re
from typing import Annotated

from fastapi import Depends, Header, HTTPException, Query, Request

from jev_bench.services import Services

__all__ = [
    "HF_FORMAT_DETAIL",
    "NO_KEY_DETAIL",
    "ApiKeyDep",
    "HfTokenDep",
    "OptionalApiKeyDep",
    "ServicesDep",
    "ThresholdQuery",
    "get_services",
    "hf_token",
    "optional_api_key",
    "require_api_key",
    "split_ids",
]

NO_KEY_DETAIL = (
    "OpenRouter API key is not configured: "
    "set OPENROUTER_API_KEY on the server or enter a key in the UI."
)

HF_FORMAT_DETAIL = (
    "The Hugging Face token must look like hf_…: check the key dialog or HF_TOKEN on the server."
)
_HF_TOKEN = re.compile(r"hf_[A-Za-z0-9]+")


def get_services(request: Request) -> Services:
    return request.app.state.services


ServicesDep = Annotated[Services, Depends(get_services)]


def optional_api_key(
    services: ServicesDep, x_openrouter_key: Annotated[str | None, Header()] = None
) -> str | None:
    return services.api_key(x_openrouter_key)


def require_api_key(key: Annotated[str | None, Depends(optional_api_key)]) -> str:
    if key is None:
        raise HTTPException(status_code=400, detail=NO_KEY_DETAIL)
    return key


def hf_token(
    services: ServicesDep, x_hf_token: Annotated[str | None, Header()] = None
) -> str | None:
    token = services.hf_token(x_hf_token)
    if token is not None and _HF_TOKEN.fullmatch(token) is None:
        raise HTTPException(status_code=400, detail=HF_FORMAT_DETAIL)
    return token


OptionalApiKeyDep = Annotated[str | None, Depends(optional_api_key)]
ApiKeyDep = Annotated[str, Depends(require_api_key)]
HfTokenDep = Annotated[str | None, Depends(hf_token)]
ThresholdQuery = Annotated[float | None, Query(gt=0.0, le=1.0, allow_inf_nan=False)]


def split_ids(value: str | None) -> list[str]:
    parts = (part.strip() for part in (value or "").split(","))
    return list(dict.fromkeys(part for part in parts if part))
