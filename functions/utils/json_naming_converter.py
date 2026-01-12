from __future__ import annotations

import re
from typing import Any, Iterable, Optional


_SNAKE_RE = re.compile(r"_([a-zA-Z0-9])")


def snake_to_camel(s: str) -> str:
    """
    Convert snake_case string to camelCase.

    - Leaves strings without '_' unchanged
    - Preserves leading/trailing underscores
    """
    if "_" not in s:
        return s

    # preserve leading/trailing underscores
    leading = len(s) - len(s.lstrip("_"))
    trailing = len(s) - len(s.rstrip("_"))
    core = s.strip("_")

    if not core:
        return s  # e.g. "___"

    parts = [p for p in core.split("_") if p]
    if not parts:
        return s

    first = parts[0]
    rest = [p[:1].upper() + p[1:] if p else p for p in parts[1:]]
    camel = first + "".join(rest)

    return ("_" * leading) + camel + ("_" * trailing)


def convert_keys_snake_to_camel(
    obj: Any,
    *,
    preserve_container_keys: Optional[Iterable[str]] = None,
) -> Any:
    """
    Recursively convert dict keys from snake_case to camelCase.

    IMPORTANT:
    Some fields are free-form containers whose INNER KEYS must remain unchanged.

    Args:
        obj:
            Any JSON-like object (dict / list / primitive)
        preserve_container_keys:
            Iterable of keys (snake_case OR camelCase) for which:
            - the container key is converted to camelCase
            - BUT its child dict keys are preserved exactly

    Returns:
        New object with converted keys (input is not mutated)
    """
    preserve = set(preserve_container_keys or [])

    if isinstance(obj, list):
        return [
            convert_keys_snake_to_camel(x, preserve_container_keys=preserve)
            for x in obj
        ]

    if isinstance(obj, dict):
        out: dict[str, Any] = {}

        for key, value in obj.items():
            if not isinstance(key, str):
                out[key] = value
                continue

            camel_key = snake_to_camel(key)
            preserve_children = key in preserve or camel_key in preserve

            if preserve_children and isinstance(value, dict):
                out[camel_key] = value
            else:
                out[camel_key] = convert_keys_snake_to_camel(
                    value, preserve_container_keys=preserve
                )

        return out

    return obj
