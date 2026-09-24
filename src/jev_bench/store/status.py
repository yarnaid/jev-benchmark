"""Lifecycle status shared by runs and generations.

Types:
    JobStatus: running -> completed | cancelled | failed | interrupted.
"""

from typing import Literal

__all__ = [
    "JobStatus",
]

type JobStatus = Literal["running", "completed", "cancelled", "failed", "interrupted"]
