"""Typed state for continuing an interrupted user-facing answer."""

# Leave room beneath the resident prefill ceiling for the system contract,
# original request, and recent conversation. The complete draft remains in
# request state for validation and deterministic merging; only the suffix is
# placed back in the model's context when an unusually large answer continues.
CONTINUATION_PROMPT_PREFIX_MAX_CHARS = 32_000


def continuation_state_text(value: object) -> str:
    """Return the authored partial answer without flattening its structure."""

    return str(value or "").strip()


def continuation_prompt_prefix(value: object) -> str:
    """Return the exact suffix ending at the interrupted decoding boundary."""

    text = continuation_state_text(value)
    if len(text) <= CONTINUATION_PROMPT_PREFIX_MAX_CHARS:
        return text
    return text[-CONTINUATION_PROMPT_PREFIX_MAX_CHARS:]
