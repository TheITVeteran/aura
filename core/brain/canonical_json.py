"""Pure canonical JSON encoding shared by signed brain evidence.

Keep this module free of runtime state, filesystem access, and model imports so
symbolic verifier identities can include it without acquiring those powers.
"""

from __future__ import annotations

import json
from typing import Any


def canonical_json_bytes(value: Any) -> bytes:
    """Encode a JSON-compatible value deterministically for hashes/signatures."""

    return json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


__all__ = ["canonical_json_bytes"]
