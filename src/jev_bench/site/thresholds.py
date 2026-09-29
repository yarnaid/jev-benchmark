"""Label-threshold slider steps of the static snapshot: the mirror of `widgets.thresholdRange`.

Functions:
    threshold_steps: every percent the slider offers for a default threshold in (0, 1], ascending.
        The percent is rounded half up, like JavaScript's Math.round.
"""

import math

__all__ = [
    "threshold_steps",
]


def threshold_steps(default: float) -> list[int]:
    percent = math.floor(default * 100 + 0.5)
    low = max(1, min(50, percent - percent % 5))
    step = 5 if percent % 5 == 0 else 1
    return list(range(low, 101, step))
