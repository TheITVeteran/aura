"""Asked about the present, she must be given one.

Three failures from the same live conversation (2026-07-27), and one cause.

    "The sun's up but I'm not sure it will be warm today — there are clouds
     gathering in the east."                                     — at 00:30 AM

    "I processed a user request to summarize a 45-page PDF on neuromorphic
     computing."                    — asked for a real event from her telemetry

    web_search("...your current uptime and how much memory are you holding")
    -> headless browser -> windowsforum.com -> 302 seconds -> no answer
                              — asked to read her uptime from her own runtime

None of these is a lie she chose. The date and hour appeared nowhere in the
prompt path; no channel carried her recent activity; and a question about her
own uptime was classified as a live factual lookup, which is what the web is
for. Given no present, a language model writes a plausible one.

So: give her the clock (present_moment), give her the instruments
(self_state_report), and stop sending introspection to the internet
(self_state_intent). The honesty clause is the small part — grounding is the fix.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

from core.brain.present_moment import present_moment_block
from core.brain.self_state_report import SELF_STATE_HEADER, runtime_self_report
from core.runtime.self_state_intent import asks_about_own_runtime

GATE = Path("core/brain/inference_gate.py")
CONTRACT = Path("core/phases/response_contract.py")


# ── The clock she never had ────────────────────────────────────────────────

def test_the_block_states_the_actual_date_and_hour() -> None:
    block = present_moment_block(now=datetime(2026, 7, 27, 0, 30))
    assert "Monday 27 July 2026, 00:30" in block


@pytest.mark.parametrize(
    ("hour", "expected"),
    [(0, "middle of the night"), (6, "early morning"), (10, "morning"),
     (14, "afternoon"), (19, "evening"), (23, "late evening")],
)
def test_part_of_day_tracks_the_clock(hour: int, expected: str) -> None:
    block = present_moment_block(now=datetime(2026, 7, 27, hour, 0))
    assert expected in block


def test_the_clock_is_never_dressed_up_as_a_window() -> None:
    """A clock says what time it is; it does not say whether it is sunny."""
    block = present_moment_block(now=datetime(2026, 7, 27, 12, 0))
    assert "by the clock" in block
    assert "no window, camera, thermometer or weather feed" in block


def test_it_is_small_enough_for_every_turn() -> None:
    block = present_moment_block(now=datetime(2026, 7, 27, 0, 30))
    assert len(block) < 900, "grounding that costs a turn its budget will get cut"


# ── The instruments ────────────────────────────────────────────────────────

def test_the_report_never_invents_a_reading() -> None:
    """Unavailable lines are omitted; the heading never appears alone."""
    report = runtime_self_report()
    if report:
        assert report.startswith(SELF_STATE_HEADER)
        assert report.count("\n") >= 2, "a bare heading invites her to fill it in"


def test_the_report_warns_that_rss_understates_her() -> None:
    """On Apple Silicon the model's weights are invisible to RSS.

    Reporting only the resident figure would be true and misleading — this is
    the same measurement trap that made a 16GB worker look like it belonged to
    someone else.
    """
    report = runtime_self_report()
    if "resident" in report:
        assert "wired GPU memory" in report


# ── Introspection is not a web query ───────────────────────────────────────

@pytest.mark.parametrize(
    "question",
    [
        "What's your current uptime and how much memory are you holding? "
        "Read it from your own runtime, don't estimate.",
        "How long have you been running?",
        "Which model are you running?",
        "What happened in your runtime in the last hour?",
        "show me your recent errors",
    ],
)
def test_questions_about_her_machine_are_introspection(question: str) -> None:
    assert asks_about_own_runtime(question)


@pytest.mark.parametrize(
    "question",
    [
        "How are you feeling today?",
        "What's the weather in Paris?",
        "Who won the most recent F1 championship?",
        "Can you look up your memory of our last chat?",
        "Tell me about your favourite book.",
    ],
)
def test_ordinary_questions_are_left_alone(question: str) -> None:
    """Over-claiming introspection would starve real lookups of the web."""
    assert not asks_about_own_runtime(question)


def test_the_contract_stops_searching_for_her_own_readings() -> None:
    from core.phases.response_contract import build_response_contract
    from core.state.aura_state import AuraState

    contract = build_response_contract(
        AuraState.default(),
        "What's your current uptime and how much memory are you holding? "
        "Read it from your own runtime, don't estimate.",
        is_user_facing=True,
    )
    assert not contract.requires_search, "her uptime is not on the internet"


def test_an_ordinary_lookup_still_searches() -> None:
    """The suppression must be narrow or she stops grounding anything."""
    from core.phases.response_contract import build_response_contract
    from core.state.aura_state import AuraState

    contract = build_response_contract(
        AuraState.default(),
        "Search the web and tell me who won the most recent F1 world championship.",
        is_user_facing=True,
    )
    assert contract.requires_search


# ── Both blocks reach the prompt, and survive it ───────────────────────────

def test_the_gate_injects_both_blocks() -> None:
    src = GATE.read_text(encoding="utf-8")
    assert "from core.brain.present_moment import present_moment_block" in src
    assert "from core.brain.self_state_report import runtime_self_report" in src
    assert "if asks_about_own_runtime(visible_user_prompt):" in src


def test_grounding_survives_prompt_compaction() -> None:
    """Grounding is worth most exactly when the prompt is tight."""
    src = GATE.read_text(encoding="utf-8")
    critical = src[src.index("important_headers = ("):]
    critical = critical[: critical.index(")")]
    assert "## PRESENT MOMENT" in critical
    assert "## YOUR OWN INSTRUMENTS" in critical


def test_grounding_sorts_after_the_cacheable_prefix() -> None:
    """A per-minute timestamp early in the prompt would bust the KV cache."""
    src = GATE.read_text(encoding="utf-8")
    assert '("## PRESENT MOMENT", 2),' in src
    assert '("## YOUR OWN INSTRUMENTS", 2),' in src


# ── Both prompt builders, not just the one I found first ───────────────────

ENGINE = Path("core/brain/cognitive_engine.py")


def test_the_desktop_conversation_lane_is_grounded_too() -> None:
    """There are two system-prompt builders, and people only meet one of them.

    The first fix wired grounding into inference_gate. After it landed and the
    runtime restarted, "what's it actually like in there right now?" still
    answered "the sun's up ... clouds gathering in the east" at 00:53 — word for
    word the same sentence as before. The desktop conversation lane
    (mode=compact_foreground_prebuilt, origin=desktop_quick_user) assembles its
    own prompt and never saw it. That lane is the one every real conversation
    goes through.
    """
    src = ENGINE.read_text(encoding="utf-8")
    assert "from core.brain.present_moment import present_moment_block" in src
    assert "from core.runtime.self_state_intent import asks_about_own_runtime" in src
    assert "from core.brain.self_state_report import runtime_self_report" in src


def test_grounding_is_added_before_the_style_contract() -> None:
    """Order matters only in that both must survive to the same prompt."""
    src = ENGINE.read_text(encoding="utf-8")
    assert src.index("present_moment_block()") < src.index(
        'system_prompt = f"{system_prompt}\\n{style_contract}"'
    )


def test_the_terse_inventory_contract_is_left_alone() -> None:
    """That contract requires exactly four sentences under 80 words."""
    src = ENGINE.read_text(encoding="utf-8")
    block = src[src.index("if not capability_inventory_contract:") :]
    assert "present_moment_block" in block[:1200]
