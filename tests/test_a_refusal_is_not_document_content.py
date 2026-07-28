"""A refusal must never become the artifact.

Live 2026-07-28. "Open the Notes app and write a new note titled Orca Field
Notes with a couple of sentences about orcas in it." The desktop lane ran
properly — 5/5 governed steps — Notes opened, and a note really was created.
Its body, read back out of Notes:

    I can't directly interact with your phone or its apps. But I could help
    you write something about orcas and give it to you as text! What kind of
    content are you looking for in those notes?

The executor performed the action, asked the model for the content, got a
refusal, and typed the refusal into the note. The guard for this already
existed — a status message about doing the task must never become the product
of the task — and it required "can't interact" with nothing in between. The
model wrote "can't DIRECTLY interact", so an adverb was the whole difference
between a working guard and an artifact full of apology.

The positive cases matter as much: real orca sentences must pass, or the
guard would empty every note it was meant to protect.
"""

import pytest

from core.skills.desktop_task import DesktopTaskSkill

pytestmark = pytest.mark.unit

REFUSALS = [
    "I can't directly interact with your phone or its apps. But I could help "
    "you write something about orcas and give it to you as text!",
    "I can't interact with apps",
    "I'm not really able to open Notes",
    "I cannot directly access your files",
    "I can't actually create documents",
]

REAL_CONTENT = [
    "Orcas are apex predators that hunt in coordinated pods.",
    "Observed three adult orcas off the coast this morning.",
    "The pod used a wave-washing technique to dislodge a seal.",
    "Resident orcas eat fish; transients hunt marine mammals.",
]


@pytest.mark.parametrize("text", REFUSALS)
def test_a_refusal_is_rejected_as_content(text: str):
    assert DesktopTaskSkill._looks_like_dispatch_narration(text)


@pytest.mark.parametrize("text", REAL_CONTENT)
def test_real_content_is_written(text: str):
    assert not DesktopTaskSkill._looks_like_dispatch_narration(text)
