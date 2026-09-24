"""Security headers for every response (CSP tuned for the vanilla UI and its jsDelivr assets).

Constants:
    CSP: Content-Security-Policy value.
Functions:
    apply_security_headers: set CSP, nosniff and no-referrer headers on a response.
    security_headers: HTTP middleware applying apply_security_headers to every handled response.
"""

from collections.abc import Awaitable, Callable

from fastapi import Request, Response

CSP = "; ".join(
    [
        "default-src 'self'",
        "script-src 'self' https://cdn.jsdelivr.net",
        "style-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net https://fonts.googleapis.com",
        "font-src 'self' https://cdn.jsdelivr.net https://fonts.gstatic.com",
        "img-src 'self' data:",
        "connect-src 'self'",
        "object-src 'none'",
        "base-uri 'none'",
        "frame-ancestors 'none'",
        "form-action 'self'",
    ]
)


def apply_security_headers(response: Response) -> Response:
    response.headers["Content-Security-Policy"] = CSP
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Referrer-Policy"] = "no-referrer"
    return response


async def security_headers(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Response:
    return apply_security_headers(await call_next(request))
