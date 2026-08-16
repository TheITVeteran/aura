"""What is actually known about who produced a piece of generated text.

``provider_verified: True`` used to be set because the SDK returned
without raising. That is evidence the call did not throw, and nothing
else — no request digest, no response digest, no account identity, no
transport record — and local results omitted the field entirely, so a
reader could not tell an unverified cloud answer from a local one (CP126
``b337de3a``).

There is no signing chain for provider responses, so nothing here claims
one. The receipt records what can be checked: which SDK path ran, what
went out, what came back, and whether a system instruction travelled
natively. ``attestation`` names that bound so no reader mistakes it for
provider-side proof.
"""

from __future__ import annotations

import hashlib
import time
from typing import Any

__all__ = ["ATTESTATION_LEVEL", "digest", "provider_receipt", "reported_token_count"]

#: The honest name for "we watched our own SDK call return".
ATTESTATION_LEVEL = "sdk_return_observed_locally"


def digest(text: str) -> str:
    return hashlib.sha256(str(text or "").encode("utf-8", "replace")).hexdigest()


def reported_token_count(response: Any) -> int | None:
    """The provider's own usage count, when it gives one."""
    usage = getattr(response, "usage_metadata", None)
    for attr in ("total_token_count", "total_tokens"):
        value = getattr(usage, attr, None)
        if isinstance(value, int) and value >= 0:
            return value
    return None


def provider_receipt(
    *,
    provider: str,
    model: str,
    prompt: str,
    response: str,
    system_instruction: str | None,
    transport: str,
) -> dict[str, Any]:
    """Build the receipt for one completed provider call."""
    return {
        "provider": provider,
        "model": model,
        "transport": transport,
        "request_sha256": digest(prompt),
        "response_sha256": digest(response),
        "system_instruction_sha256": digest(system_instruction) if system_instruction else "",
        "role_separation": "native" if system_instruction else "none",
        "observed_at": time.time(),
        "attestation": ATTESTATION_LEVEL,
    }
