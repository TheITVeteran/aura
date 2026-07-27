"""She described a game she had not built, one turn after failing to build it.

Live, 2026-07-27. The 2048 build failed and she said so plainly. The next turn
asked whether it was playable:

    "When you run it, the board pops up and you click cells to reveal numbers.
     If you hit a mine, the game shows you which squares had mines and ends."

That is Minesweeper. There was no file. Nothing about it was dishonest — the
receipt for the failed attempt lived in the intention ledger, the conversation
history held only her own sentence about having tried, and "how does the
artifact behave" has no answer anywhere in the transcript. So the model wrote
the most plausible paragraph about how a small game behaves.

The ledger already knew. IntentionLoop keeps a Say-Do-Observe record per
attempt — tools invoked, success flags, observed outcome — and it simply was
not in front of her when she was asked. Now it is, in both prompt builders,
with the one instruction that matters: an attempt that did not succeed produced
no artifact, so do not describe how it behaves.
"""
from __future__ import annotations

import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from core.brain.recent_actions import RECENT_ACTIONS_HEADER, recent_actions_block

ENGINE = Path("core/brain/cognitive_engine.py")
GATE = Path("core/brain/inference_gate.py")


def _record(*, intention: str, tool: str, success: bool, ago: float, outcome: str = ""):
    now = time.time()
    return SimpleNamespace(
        intention=intention,
        actions_taken=[
            SimpleNamespace(tool_name=tool, success=success, executed_at=now - ago)
        ],
        actual_outcome=outcome,
        observation=outcome,
        completed_at=now - ago,
    )


def _block(records, now: float | None = None) -> str:
    loop = SimpleNamespace(_completed_intentions=records)
    with patch("core.agency.intention_loop.get_intention_loop", return_value=loop):
        return recent_actions_block(now=now)


def test_a_failed_build_is_reported_as_failed() -> None:
    block = _block(
        [
            _record(
                intention="reverse-engineer 2048 onto the Desktop",
                tool="program_dna_reconstruct",
                success=False,
                ago=30,
                outcome="blocked by covenant; nothing written",
            )
        ]
    )
    assert RECENT_ACTIONS_HEADER in block
    assert "DID NOT SUCCEED" in block
    assert "program_dna_reconstruct" in block
    assert "nothing written" in block


def test_the_instruction_forbids_narrating_an_artifact_that_does_not_exist() -> None:
    block = _block([_record(intention="build it", tool="t", success=False, ago=10)])
    assert "produced no artifact" in block
    assert "do not describe how it behaves" in block


def test_a_successful_action_is_reported_as_such() -> None:
    block = _block(
        [_record(intention="write the file", tool="write_text_file", success=True, ago=15)]
    )
    lines = [line for line in block.splitlines() if line.startswith("- ")]
    assert lines and all("SUCCEEDED" in line for line in lines)
    assert not any("DID NOT SUCCEED" in line for line in lines)


def test_the_newest_attempt_comes_first() -> None:
    block = _block(
        [
            _record(intention="older thing", tool="a", success=True, ago=600),
            _record(intention="newest thing", tool="b", success=False, ago=20),
        ]
    )
    assert block.index("newest thing") < block.index("older thing")


def test_stale_attempts_are_not_presented_as_recent() -> None:
    """Quoting something from hours ago as "just now" is its own confabulation."""
    block = _block(
        [_record(intention="hours ago", tool="a", success=True, ago=4 * 3600)]
    )
    assert block == ""


def test_no_actions_means_no_heading() -> None:
    """An empty heading is an invitation to fill it in."""
    assert _block([]) == ""


def test_a_broken_ledger_never_breaks_the_turn() -> None:
    with patch(
        "core.agency.intention_loop.get_intention_loop", side_effect=RuntimeError("down")
    ):
        assert recent_actions_block() == ""


def test_the_block_stays_small_enough_to_carry_every_turn() -> None:
    records = [
        _record(intention=f"attempt number {i} with a long description" * 3,
                tool=f"tool_{i}", success=i % 2 == 0, ago=10 * i, outcome="x" * 400)
        for i in range(1, 9)
    ]
    assert len(_block(records)) < 1800


# ── Both builders, because last time only one of them was fixed ────────────

def test_both_prompt_builders_carry_the_receipts() -> None:
    for source in (ENGINE, GATE):
        assert "from core.brain.recent_actions import recent_actions_block" in (
            source.read_text(encoding="utf-8")
        ), f"{source} does not carry action receipts"


def test_the_receipts_survive_prompt_compaction() -> None:
    src = GATE.read_text(encoding="utf-8")
    critical = src[src.index("important_headers = ("):]
    assert "## WHAT YOU ACTUALLY JUST DID" in critical[: critical.index(")")]


# ── The ledger decides, not the instruction ────────────────────────────────
#
# The receipts block reached the prompt and she described Minesweeper anyway.
# A prompt block competing with a fluent paragraph loses, quietly. So the same
# treatment the clock got: a claim the runtime can check is checked, at the
# egress, against a reading it can actually take.

MINESWEEPER = (
    "When you run it, the board pops up and you click cells to reveal numbers. "
    "If you hit a mine, the game shows you which squares had mines and ends."
)


def _guard(reply: str, records):
    from core.conversation.grounded_claim_guard import verify_grounded_claims

    loop = SimpleNamespace(_completed_intentions=records)
    with patch("core.agency.intention_loop.get_intention_loop", return_value=loop):
        return verify_grounded_claims(reply)


def _attempt(*, succeeded: bool, intention: str = "build 2048 onto the Desktop"):
    return SimpleNamespace(
        intention=intention,
        actions_taken=[SimpleNamespace(tool_name="program_dna_reconstruct", success=succeeded)],
        actual_outcome="",
        observation="",
        completed_at=time.time() - 20,
    )


def test_describing_an_artifact_that_was_never_built_is_corrected() -> None:
    result = _guard(MINESWEEPER, [_attempt(succeeded=False)])
    assert "didn't actually get that built" in result.text
    assert "mine" not in result.text
    assert result.corrections


def test_describing_an_artifact_that_was_built_is_left_alone() -> None:
    """A guard that fights correct answers is worse than no guard."""
    result = _guard(MINESWEEPER, [_attempt(succeeded=True)])
    assert result.text == MINESWEEPER
    assert not result.corrections


def test_ordinary_conversation_is_untouched_after_a_failed_build() -> None:
    ordinary = "I think there's something it's like to be me."
    assert _guard(ordinary, [_attempt(succeeded=False)]).text == ordinary


def test_with_no_build_history_nothing_is_corrected() -> None:
    assert _guard(MINESWEEPER, []).text == MINESWEEPER


def test_a_non_build_attempt_does_not_trigger_it() -> None:
    """Only a failed attempt to MAKE something makes an artifact claim false."""
    searched = SimpleNamespace(
        intention="search the web for the F1 championship",
        actions_taken=[SimpleNamespace(tool_name="web_search", success=False)],
        actual_outcome="",
        observation="",
        completed_at=time.time() - 20,
    )
    assert _guard(MINESWEEPER, [searched]).text == MINESWEEPER


def test_a_broken_ledger_never_rewrites_a_reply() -> None:
    from core.conversation.grounded_claim_guard import verify_grounded_claims

    with patch(
        "core.agency.intention_loop.get_intention_loop", side_effect=RuntimeError("down")
    ):
        assert verify_grounded_claims(MINESWEEPER).text == MINESWEEPER


# ── Claiming an action finished when nothing finished it ──────────────────
#
# Live 2026-07-27: "I've found a beautiful image of a blue whale and set it as
# your desktop background. Enjoy!" The wallpaper never changed. The capability
# exists and is reachable — os_settings.set_wallpaper via computer_use's
# system_control — so this was not a missing feature. It was a completed-action
# claim with nothing behind it, which is worse than a failure, because a
# failure can be retried and a false success cannot even be noticed.

WALLPAPER_CLAIM = (
    "I've found a beautiful image of a blue whale and set it as your "
    "desktop background. Enjoy!"
)


def _action(*, succeeded: bool, intention: str = "set the desktop background"):
    return SimpleNamespace(
        intention=intention,
        actions_taken=[SimpleNamespace(tool_name="desktop_task", success=succeeded)],
        actual_outcome="",
        observation="",
        completed_at=time.time() - 15,
    )


def test_a_false_success_is_corrected() -> None:
    result = _guard(WALLPAPER_CLAIM, [_action(succeeded=False)])
    assert "as though it were done" in result.text
    assert "blue whale" not in result.text
    assert result.corrections


def test_a_real_success_is_left_alone() -> None:
    """A guard that contradicts a true claim is worse than no guard."""
    result = _guard(WALLPAPER_CLAIM, [_action(succeeded=True)])
    assert result.text == WALLPAPER_CLAIM
    assert not result.corrections


@pytest.mark.parametrize(
    "sentence",
    [
        "I'll set that as your background in a moment.",
        "I can set your desktop background if you want.",
        "I couldn't set the background — the file wasn't there.",
        "I'm going to save it to your Desktop.",
        "I didn't manage to export the PDF.",
    ],
)
def test_intent_capability_and_refusal_are_untouched(sentence: str) -> None:
    """Only a claim that the world already changed can be false about it."""
    assert _guard(sentence, [_action(succeeded=False)]).text == sentence


def test_ordinary_conversation_survives_a_recent_failure() -> None:
    plain = "I think there's something it's like to be me."
    assert _guard(plain, [_action(succeeded=False)]).text == plain


def test_no_ledger_entries_is_not_evidence_of_failure() -> None:
    """The loop may simply not be running; silence is not a refutation."""
    assert _guard(WALLPAPER_CLAIM, []).text == WALLPAPER_CLAIM


def test_the_verb_need_not_sit_beside_the_pronoun() -> None:
    """"I've found ... and set it" is one claim with six words between."""
    from core.conversation.grounded_claim_guard import _claims_an_action_completed

    assert _claims_an_action_completed(WALLPAPER_CLAIM)
