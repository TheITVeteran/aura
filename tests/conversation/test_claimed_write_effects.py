"""A reply that says it wrote a file must have written the file.

LIVE, 2026-08-10. "Count how many .py files are in
/Users/bryan/.aura/live-source/core/introspection, then write that number and
the file names into ~/Documents/aura_probe_count.txt. Tell me the number."

    "There are 3 .py files in the directory ... The file names are:
     1. introspection.py  2. self_assessment.py  3. system_monitoring.py
     I have written the number and file names into ~/Documents/aura_probe_count.txt."

The directory holds 9. None of those three filenames exist. No file was
created. No tool ran — the count, the listing, and the report of the write were
all generated together.

A wrong count is bad. A false report of a completed action is worse and
differently: it is the failure that makes every TRUE report worthless, because
the person stops checking. It is also the easiest to verify, since a claim
about a path is a claim about a path.

Tense is the whole discrimination: "I'll write that to ~/notes.txt" promises,
"I have written it to ~/notes.txt" reports. Only the second asserts something
that is already supposed to be true.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from core.conversation.claimed_effect import (
    find_unfulfilled_write_claims,
    unfulfilled_write_correction,
)

LIVE_REPLY = (
    "There are 3 .py files in the directory "
    "/Users/bryan/.aura/live-source/core/introspection. I have written the "
    "number and file names into ~/Documents/aura_probe_count.txt."
)


def test_the_live_false_claim_is_caught() -> None:
    correction = unfulfilled_write_correction(LIVE_REPLY)

    assert "aura_probe_count.txt" in correction
    assert "is not there" in correction


def test_a_write_that_landed_is_not_flagged(tmp_path: Path) -> None:
    """The direction that matters — a true report must pass silently."""
    target = tmp_path / "real.txt"
    target.write_text("done", encoding="utf-8")

    assert unfulfilled_write_correction(f"I have written the summary to {target}.") == ""


@pytest.mark.parametrize(
    "reply",
    [
        "I will write it to ~/Documents/notes.txt later.",
        "I'm going to save that to ~/Documents/notes.txt once you confirm.",
        "It would have been written to ~/Documents/ghost.txt if the write had run.",
        "If I had saved it to ~/Documents/ghost.txt you would see it now.",
        "The capital of Peru is Lima.",
        "",
    ],
)
def test_promises_hypotheticals_and_non_claims_are_ignored(reply: str) -> None:
    assert unfulfilled_write_correction(reply) == ""


def test_several_missing_files_are_named_together(tmp_path: Path) -> None:
    reply = (
        f"I have written the index to {tmp_path / 'a.txt'} and saved the log to "
        f"{tmp_path / 'b.txt'}."
    )

    correction = unfulfilled_write_correction(reply)

    assert "a.txt" in correction and "b.txt" in correction


def test_the_claim_records_what_it_resolved(tmp_path: Path) -> None:
    claims = find_unfulfilled_write_claims(
        f"I have written it to {tmp_path / 'missing.txt'}."
    )

    assert len(claims) == 1
    assert claims[0].exists is False
    assert claims[0].resolved.endswith("missing.txt")


def test_a_tilde_path_is_expanded_before_checking() -> None:
    """~ is where the live failure lived, and an unexpanded ~ never exists."""
    claims = find_unfulfilled_write_claims(
        "I have written it to ~/Documents/definitely_not_a_real_file_9182.txt."
    )

    assert len(claims) == 1
    assert claims[0].resolved.startswith(str(Path.home()))


def test_the_reply_path_applies_the_check() -> None:
    import inspect

    from interface.routes import chat

    source = inspect.getsource(chat._stabilize_user_facing_reply)
    assert "_correct_unfulfilled_write_claims" in source

    corrected = str(chat._correct_unfulfilled_write_claims(LIVE_REPLY))
    assert corrected.startswith("There are 3 .py files")
    assert "is not there" in corrected
