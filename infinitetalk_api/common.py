from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from typing import Any


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def deep_merge(base: dict[str, Any] | None, override: dict[str, Any] | None) -> dict[str, Any]:
    result = deepcopy(base or {})
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result

