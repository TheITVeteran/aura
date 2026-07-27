"""One noisy degradation was refusing every build Bryan asked for.

The chain, measured live 2026-07-27 on a host at 4% memory pressure:

    [DEGRADATION] personality_engine (critical): RuntimeError: user-facing
      guard returned no usable text
    Incident INC-...-0002 updated (count=6, severity=emergency)
    CRITICAL EXISTENTIAL STAKES: threat=1.00 (mem_threat=0.04, deg_threat=1.00)
    -> ulysses_covenant: No heavy compute while survival is threatened
    -> "I didn't get 2048 rebuilt"

Every link is behaving as designed. ``personality_engine`` is on the
fail-closed list, so its warnings escalate to CRITICAL. ``existential_threat``
is ``max(memory_threat, degradation_threat)``, deliberately, because a
degradation cascade is a real survival signal. The Ulysses covenant refuses
heavy building work above 0.6, for good reason — it was seeded from a real
duplicate-runtime cascade.

The defect is at the start. Shaping an empty draft produces empty output, the
honesty guard passes it through unchanged, and that was recorded as a failure
to shape. Blank in, blank out, filed as an emergency, once per turn — enough to
pin deg_threat at 1.00 and keep a healthy runtime permanently "under threat".

Nothing to shape is not a failure to shape. When the draft is genuinely lost by
shaping, that is still recorded, and it now says which stage lost it.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from core.brain.personality_engine import PersonalityEngine

SOURCE = Path("core/brain/personality_engine.py")


def _shape(engine, text: str) -> str:
    return engine.filter_response(text)


@pytest.fixture()
def engine():
    return PersonalityEngine.__new__(PersonalityEngine)


@pytest.mark.parametrize("blank", ["", "   ", "\n", "\t\n "])
def test_a_blank_draft_records_nothing(engine, blank: str) -> None:
    """This fired once per turn and kept the runtime "under threat"."""
    with patch("core.brain.personality_engine._record_personality_degradation") as rec:
        result = _shape(engine, blank)
    assert result == blank
    assert not rec.called, "an empty draft is not a failure to shape"


def test_ordinary_text_is_shaped_and_records_nothing(engine) -> None:
    with patch("core.brain.personality_engine._record_personality_degradation") as rec:
        result = _shape(engine, "391.")
    assert result.strip()
    assert not rec.called


def test_assistant_voice_is_still_cured(engine) -> None:
    """The shaping this guards must keep working."""
    result = _shape(engine, "As an AI assistant, I can help.")
    assert "AI assistant" not in result


def test_text_genuinely_lost_by_shaping_is_still_recorded(engine) -> None:
    """Silence here would hide a real defect behind the fix for a noisy one."""
    with patch("core.synthesis.cure_personality_leak", return_value=""), patch(
        "core.brain.personality_engine._record_personality_degradation"
    ) as rec:
        result = _shape(engine, "a real answer that should not vanish")
    assert result == "a real answer that should not vanish"
    assert rec.called, "losing a non-empty reply is a real degradation"


def test_the_record_says_which_stage_lost_the_text(engine) -> None:
    with patch("core.synthesis.cure_personality_leak", return_value=""), patch(
        "core.brain.personality_engine._record_personality_degradation"
    ) as rec:
        _shape(engine, "a real answer")
    message = str(rec.call_args[0][0])
    assert "emptied a non-empty reply" in message


def test_the_guard_clause_precedes_any_shaping() -> None:
    src = SOURCE.read_text(encoding="utf-8")
    guard_at = src.index('if not str(text or "").strip():')
    shape_at = src.index("from core.synthesis import cure_personality_leak")
    assert guard_at < shape_at
