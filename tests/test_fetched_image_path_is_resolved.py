"""The download worked; the next step looked for a file that never existed.

Live 2026-07-29:

    system_control failed: [Errno 2] No such file or directory:
    '/Users/bryan/Desktop/orca_wallpaper.png'

while a real 234KB orca_wallpaper.**jpg** sat on the Desktop. The planner
names the file before the fetch runs, so the extension is a guess; the
gateway saves what it was actually served and renames to the sniffed type,
which is right — and left the wallpaper step pointing at a filename nobody
was ever going to write.

The source URL already had this shape and already had a sentinel. The path
needs the same one: both are facts only the receipt knows.
"""

from __future__ import annotations

from core.skills.desktop_task import (
    FETCHED_IMAGE_PATH_SENTINEL,
    FETCHED_IMAGE_SOURCE_SENTINEL,
    DesktopTaskSkill,
)


def test_the_plan_does_not_name_the_downloaded_file():
    steps = DesktopTaskSkill()._derive_single_objective_steps(
        "Find a picture of an orca online, download it to my Desktop, and set "
        "it as my wallpaper. Show me where you got the image from.",
        {},
    )
    actions = [step.action for step in steps]
    assert "fetch_topic_image" in actions
    assert "system_control" in actions

    control = next(step for step in steps if step.action == "system_control")
    assert control.target["value"] == FETCHED_IMAGE_PATH_SENTINEL, (
        "the wallpaper step must reference the fetch receipt, not a guessed "
        f"filename: {control.target}"
    )


def test_the_fetch_still_saves_where_the_person_asked():
    """The sentinel replaces the guessed EXTENSION, not the folder — a
    grizzly once landed in ~/Documents because the destination was
    hardcoded."""
    steps = DesktopTaskSkill()._derive_single_objective_steps(
        "Find a picture of an orca online, download it to my Desktop, and set "
        "it as my wallpaper.",
        {},
    )
    fetch = next(step for step in steps if step.action == "fetch_topic_image")
    assert "Desktop" in str(fetch.target["path"])


def test_the_source_page_keeps_its_own_sentinel():
    steps = DesktopTaskSkill()._derive_single_objective_steps(
        "Find a picture of an orca online, download it to my Desktop, set it "
        "as my wallpaper, and show me where you got the image from.",
        {},
    )
    opens = [step for step in steps if step.action == "open_url"]
    assert any(
        FETCHED_IMAGE_SOURCE_SENTINEL in str(step.target) for step in opens
    ), [step.target for step in opens]


def test_the_two_sentinels_are_distinct():
    assert FETCHED_IMAGE_PATH_SENTINEL != FETCHED_IMAGE_SOURCE_SENTINEL


def test_pdf_uses_the_fetch_receipt_not_a_guessed_image_extension():
    steps = DesktopTaskSkill()._derive_single_objective_steps(
        "Create a folder called Orca Demo in my Documents folder. Find an "
        "orca image online and write a synthesis into a PDF saved there.",
        {},
    )
    render = next(step for step in steps if step.action == "render_text_pdf")
    assert render.target["image_path"] == FETCHED_IMAGE_PATH_SENTINEL
