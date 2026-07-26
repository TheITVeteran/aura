"""The steering hook runs on every block of every token, so it must be cheap.

`AffectiveSteeringHook` patches the forward pass of all 64 transformer blocks.
Anything it does is therefore multiplied by 64 and by every token — including
every token of the prompt during prefill, before a first token can exist.

Two costs sat on that path:

1. `_completion_position_mask` rebuilt a NumPy array sized by the sequence
   length and re-uploaded it to the GPU on EVERY call. The mask is a constant
   for a given shape — zeros with a 1.0 at the final position — so this was
   thousands of host allocations and uploads per turn, scaling with prompt
   length.

2. The composite steering vector was fetched and dtype-cast before checking
   whether `effective_alpha` was even non-zero, so a turn where steering had
   stood down still paid for a vector it then multiplied by zero.

Live on the desktop surface 2026-07-26: short prompts answered fine while
longer ones hit "First-token HARD CEILING exceeded (livelocked: heartbeats but
zero tokens)" at ~110s. Cost scaling with sequence length on the pre-first-token
path is exactly that signature.
"""
from __future__ import annotations

import re
from pathlib import Path

SOURCE = Path("core/consciousness/affective_steering.py")


def _mask_body() -> str:
    src = SOURCE.read_text(encoding="utf-8")
    start = src.index("def _completion_position_mask")
    return src[start : src.index("def ", start + 10)]


def test_the_completion_mask_is_cached_not_rebuilt() -> None:
    body = _mask_body()
    assert "_COMPLETION_MASK_CACHE.get(cache_key)" in body, (
        "the mask must be looked up before it is built"
    )
    assert "return cached" in body
    # The allocation must sit AFTER the cache hit returns.
    assert body.index("_COMPLETION_MASK_CACHE.get") < body.index("np.zeros")


def test_the_cache_key_covers_shape_and_dtype() -> None:
    """A mask of the wrong width or dtype would corrupt the residual stream."""
    body = _mask_body()
    assert "cache_key = (shape, str(h.dtype))" in body


def test_the_cache_is_bounded() -> None:
    src = SOURCE.read_text(encoding="utf-8")
    assert "_COMPLETION_MASK_CACHE_MAX" in src
    body = _mask_body()
    assert "_COMPLETION_MASK_CACHE.clear()" in body, (
        "an unbounded cache on a per-token path is its own leak"
    )


def test_alpha_is_checked_before_the_composite_is_fetched() -> None:
    """Steering that has stood down must cost nothing on the token path."""
    src = SOURCE.read_text(encoding="utf-8")
    start = src.index("def steered_call")
    body = src[start : src.index("hook._inject_count += 1", start)]
    alpha_at = body.index("effective_alpha = hook._effective_alpha()")
    composite_at = body.index("hook.compute_composite_vector_mx")
    assert alpha_at < composite_at, (
        "the composite must not be fetched and cast before alpha is known"
    )
    assert re.search(r"if effective_alpha > 0\.0\s*\n\s*else None", body), (
        "a zero alpha must skip the composite entirely"
    )


def test_the_mask_still_marks_only_the_final_position() -> None:
    """The optimisation must not change what the mask means."""
    body = _mask_body()
    assert "mask_np[-1, 0] = 1.0" in body
    assert "mask_np[:, -1, 0] = 1.0" in body
    assert "np.zeros" in body
