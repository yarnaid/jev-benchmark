"""Lifecycle status shared by runs and generations.

Types:
    JobStatus: running -> completed | cancelled | failed | interrupted.
"""

from typing import Literal

type JobStatus = Literal["running", "completed", "cancelled", "failed", "interrupted"]
