"""Validation of `string.Template` prompt templates used in the TOML configs.

Functions:
    check_template: return the template if it is valid and uses only allowed placeholders.
"""

from collections.abc import Collection
from string import Template

__all__ = [
    "check_template",
]


def check_template(template: str, allowed: Collection[str]) -> str:
    parsed = Template(template)
    unknown = sorted(set(parsed.get_identifiers()) - set(allowed))
    if not parsed.is_valid() or unknown:
        msg = (
            f"template may only use placeholders {sorted(allowed)} "
            f"(write $$ for a literal $); unknown: {unknown}"
        )
        raise ValueError(msg)
    return template
