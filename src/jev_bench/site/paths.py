"""File layout of the static snapshot: the file that answers each GET of the JSON API.

The JS half of this contract is `js/static-api.js`; both are checked against the shared cases in
`tests/fixtures/site_paths.json`.

Types:
    ThresholdEndpoint: the view-dependent endpoints that take a label threshold.
Functions:
    base_file: `api<path>.json`, for a request that does not depend on a view.
    threshold_file: the compare or email-list file of a view at a threshold percent (None: default).
    email_file: the email-detail file of a view.
"""

from typing import Literal

__all__ = [
    "ThresholdEndpoint",
    "base_file",
    "email_file",
    "threshold_file",
]

type ThresholdEndpoint = Literal["compare", "emails"]


def base_file(path: str) -> str:
    return f"api{path}.json"


def threshold_file(endpoint: ThresholdEndpoint, view_id: str, percent: int | None) -> str:
    name = "t-default" if percent is None else f"t{percent}"
    return f"api/{endpoint}/{view_id}/{name}.json"


def email_file(view_id: str, email_id: str) -> str:
    return f"api/email/{view_id}/{email_id}.json"
