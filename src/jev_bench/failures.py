"""Human-readable failure text and logging for failed background jobs.

Functions:
    unwrap_leaf: first non-group exception inside a (possibly nested) ExceptionGroup
        (re-exported from `jev_bench.jobs`, which owns the implementation to avoid an
        import cycle).
    failure_text: unwraps an ExceptionGroup to its first leaf and describes it.
    log_job_failure: logs a job failure at ERROR; a provider OpenRouterError leaf logs
        the message alone, anything else logs with its traceback.
"""

from loguru import logger

from jev_bench.jobs import describe_error, unwrap_leaf
from jev_bench.openrouter import OpenRouterError


def failure_text(exc: BaseException) -> str:
    leaf = unwrap_leaf(exc)
    if isinstance(leaf, OpenRouterError):
        return describe_error(leaf)
    name = type(leaf).__name__
    message = str(leaf)
    return f"{name}: {message}" if message else name


def log_job_failure(exc: BaseException, text: str, **context: str) -> None:
    bound = logger.bind(**context)
    if isinstance(unwrap_leaf(exc), OpenRouterError):
        bound.error(text)
    else:
        bound.opt(exception=exc).error(text)
