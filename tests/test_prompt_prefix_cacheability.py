"""Turn-volatile prompt sections must come LAST, or KV reuse is worthless.

A prompt-cache entry is KV for a byte-identical prefix, so a single volatile
byte early in the system prompt destroys the reuse of everything after it.
Measured live on the desktop lane: a conversation turn scored a real cache hit
and still reused only 325 of 2,105 tokens (15%), because mood/tone/unity sat
~325 tokens in. The divergence diagnostic named it exactly:
" empathy\\nTone: inquisitive_engaged\\n\\n## UNITY\\nLevel: coherent".
"""
from __future__ import annotations

from core.brain.inference_gate import InferenceGate

STABLE_HEADERS = (
    "## USER-FACING CONVERSATION RELIABILITY CONTRACT",
    "## LIVE DESKTOP RESPONSE CONTRACT",
)
PER_TURN_HEADERS = (
    "## LIVE TONE",
    "## UNITY",
    "## SOMATIC STATE",
    "## STATE",
    "## FUNCTIONAL STATE SIGNALS",
    "## DERIVED RUNTIME SIGNALS",
)

_CONTENT = (
    "## LIVE TONE\nMood: curious\nTone: inquisitive_engaged\n\n"
    "## UNITY\nLevel: coherent | Unity: 0.81\n\n"
    "## USER-FACING CONVERSATION RELIABILITY CONTRACT\nAnswer the person directly.\n\n"
    "## CONTINUITY SUMMARY\nWe were discussing forgetting.\n\n"
    "## SOMATIC STATE\nvitality=0.73 thermal=0.4\n\n"
    "## GOALS\nFinish the soak.\n\n"
    "## LIVE DESKTOP RESPONSE CONTRACT\n- Be natural, not a telemetry card.\n"
)


def _emitted_headers(rendered: str) -> list[str]:
    return [
        line.strip()
        for line in rendered.splitlines()
        if line.startswith("## ") or line.startswith("[")
    ]


def test_stable_contracts_lead_and_per_turn_state_trails():
    headers = _emitted_headers(
        InferenceGate._critical_foreground_system_excerpt(_CONTENT, budget=4000)
    )
    stable_positions = [headers.index(h) for h in STABLE_HEADERS if h in headers]
    volatile_positions = [headers.index(h) for h in PER_TURN_HEADERS if h in headers]

    assert stable_positions, "the stable contract sections must still be emitted"
    assert volatile_positions, "the volatile sections must still be emitted"
    assert max(stable_positions) < min(volatile_positions), (
        f"volatile state emitted before stable text, so the cacheable prefix "
        f"ends at the first mood change: {headers}"
    )


def test_no_section_is_dropped_by_the_reordering():
    rendered = InferenceGate._critical_foreground_system_excerpt(_CONTENT, budget=4000)
    for header in (
        "## LIVE TONE",
        "## UNITY",
        "## USER-FACING CONVERSATION RELIABILITY CONTRACT",
        "## CONTINUITY SUMMARY",
        "## SOMATIC STATE",
        "## GOALS",
        "## LIVE DESKTOP RESPONSE CONTRACT",
    ):
        assert header in rendered, f"reordering dropped {header}"


def test_volatility_ranking_covers_every_selected_header():
    """A header the ranking does not know lands in the middle by default.

    That is safe, but a PER-TURN section defaulting to the middle would quietly
    re-break the prefix, so every known volatile header must be ranked.
    """
    ranked = {
        header for header, _rank in InferenceGate._FOREGROUND_SECTION_VOLATILITY
    }
    for header in PER_TURN_HEADERS + STABLE_HEADERS:
        assert header in ranked, f"{header} has no volatility rank"

    for header in PER_TURN_HEADERS:
        assert InferenceGate._foreground_section_volatility(f"{header}\nx") == 2
    for header in STABLE_HEADERS:
        assert InferenceGate._foreground_section_volatility(f"{header}\nx") == 0
