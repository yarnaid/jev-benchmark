"""Human-readable failure text persisted for failed background jobs.

Functions:
    failure_text: unwraps an ExceptionGroup to its first leaf and describes it.
"""

from jev_bench.jobs import describe_error
from jev_bench.openrouter import OpenRouterError


def failure_text(exc: BaseException) -> str:
    leaf = exc
    while isinstance(leaf, BaseExceptionGroup) and leaf.exceptions:
        leaf = leaf.exceptions[0]
    if isinstance(leaf, OpenRouterError):
        return describe_error(exc)
    name = type(leaf).__name__
    message = str(leaf)
    return f"{name}: {message}" if message else name
