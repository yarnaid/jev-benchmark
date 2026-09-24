"""Human-readable failure text persisted for failed background jobs.

Functions:
    unwrap_leaf: first non-group exception inside a (possibly nested) ExceptionGroup.
    failure_text: unwraps an ExceptionGroup to its first leaf and describes it.
"""

from jev_bench.jobs import describe_error
from jev_bench.openrouter import OpenRouterError


def unwrap_leaf(exc: BaseException) -> BaseException:
    leaf = exc
    while isinstance(leaf, BaseExceptionGroup) and leaf.exceptions:
        leaf = leaf.exceptions[0]
    return leaf


def failure_text(exc: BaseException) -> str:
    leaf = unwrap_leaf(exc)
    if isinstance(leaf, OpenRouterError):
        return describe_error(leaf)
    name = type(leaf).__name__
    message = str(leaf)
    return f"{name}: {message}" if message else name
