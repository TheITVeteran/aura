"""Save where the person said, not where the code prefers.

Live 2026-07-28: "Find a picture of a grizzly bear online, download it to my
Desktop, and set it as my wallpaper."

Everything worked except the one explicit instruction. A real 1.3MB grizzly
PNG was fetched through the governed image gateway and the wallpaper was set
— and the file landed in ~/Documents, because the destination was hardcoded.
That is the difference between following an instruction and approximating
one, and it is the kind of thing a person notices immediately when they go
looking for the file.

The default stays ~/Documents for requests that name no folder, so nothing
that worked before changes.
"""

import pytest

from core.skills.desktop_task import DesktopTaskSkill

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("objective,expected", [
    ("download it to my Desktop and set it as my wallpaper", "~/Desktop"),
    ("save the picture in my Downloads and use it as wallpaper", "~/Downloads"),
    ("put it in my Pictures folder and set the wallpaper", "~/Pictures"),
    ("save it to my Documents and make it the background", "~/Documents"),
])
def test_a_named_folder_is_honoured(objective: str, expected: str):
    assert DesktopTaskSkill._requested_image_folder(objective) == expected


def test_no_named_folder_keeps_the_old_default():
    assert (
        DesktopTaskSkill._requested_image_folder("set my wallpaper to a grizzly bear")
        == "~/Documents"
    )


def test_the_plan_targets_the_requested_folder():
    steps = DesktopTaskSkill()._derive_steps_from_objective(
        "Find a picture of a grizzly bear online, download it to my Desktop, "
        "and set it as my wallpaper.",
        {},
    )
    targets = [str(getattr(step, "target", "")) for step in steps]
    assert any("~/Desktop/grizzly_bear_wallpaper.png" in t for t in targets), targets
    assert not any("~/Documents/grizzly" in t for t in targets), targets
