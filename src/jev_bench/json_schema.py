"""Builders for strict structured-output JSON schemas.

Functions:
    strict_object: object schema with `additionalProperties: false` and every property required.
"""

from typing import Any

__all__ = [
    "JsonSchema",
    "strict_object",
]

type JsonSchema = dict[str, Any]


def strict_object(properties: dict[str, JsonSchema], description: str | None = None) -> JsonSchema:
    schema: JsonSchema = {
        "type": "object",
        "additionalProperties": False,
        "required": list(properties),
        "properties": properties,
    }
    if description:
        schema["description"] = description
    return schema
