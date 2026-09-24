"""FastAPI dependencies: the shared Services and the API key a job should use.

Constants:
    NO_KEY_DETAIL
Types:
    ServicesDep, ApiKeyDep
Functions:
    get_services: Services stored on the app by the lifespan.
    require_api_key: server key, else the X-OpenRouter-Key header, else HTTP 400.
"""

from typing import Annotated

from fastapi import Depends, Header, HTTPException, Request

from jev_bench.services import Services

NO_KEY_DETAIL = (
    "OpenRouter API key is not configured: "
    "set OPENROUTER_API_KEY on the server or enter a key in the UI."
)


def get_services(request: Request) -> Services:
    return request.app.state.services


ServicesDep = Annotated[Services, Depends(get_services)]


def require_api_key(
    services: ServicesDep, x_openrouter_key: Annotated[str | None, Header()] = None
) -> str:
    key = services.api_key(x_openrouter_key)
    if key is None:
        raise HTTPException(status_code=400, detail=NO_KEY_DETAIL)
    return key


ApiKeyDep = Annotated[str, Depends(require_api_key)]
