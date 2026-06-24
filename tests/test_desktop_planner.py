"""Tests for the desktop planner (goal → verified computer-use steps)."""
from __future__ import annotations

import pytest

from core.agency.desktop_planner import DesktopAdapter, DesktopPlanner, is_desktop_goal


class _RecorderAdapter(DesktopAdapter):
    def __init__(self):
        self.calls = []

    async def open_app(self, name):
        self.calls.append(("open_app", name))

    async def open_url(self, url):
        self.calls.append(("open_url", url))

    async def type_text(self, text):
        self.calls.append(("type_text", text))

    async def make_folder(self, path):
        self.calls.append(("make_folder", path))

    async def set_wallpaper(self, target):
        self.calls.append(("set_wallpaper", target))


# ── detection ─────────────────────────────────────────────────────────────

@pytest.mark.parametrize("g", [
    "open Notes",
    "make a folder called Journals on the desktop",
    "set the wallpaper to a mountain",
    "open https://docs.google.com in Chrome",
])
def test_recognizes_desktop_goals(g):
    assert is_desktop_goal(g)


def test_ignores_non_desktop():
    assert not is_desktop_goal("what is the meaning of life")


# ── planning: correct action + verifier predicate ─────────────────────────

@pytest.mark.asyncio
async def test_open_app_step_verified_with_app_frontmost():
    rec = _RecorderAdapter()
    steps = await DesktopPlanner(adapter=rec).plan("open Notes")
    assert len(steps) == 1
    s = steps[0]
    assert s.name == "open_app" and s.verify == "app_frontmost" and s.verify_args == {"name": "Notes"}
    await s.action()
    assert rec.calls == [("open_app", "Notes")]


@pytest.mark.asyncio
async def test_make_folder_step_verified_with_folder_exists():
    rec = _RecorderAdapter()
    steps = await DesktopPlanner(adapter=rec).plan("create a folder called Journals on the desktop")
    s = steps[0]
    assert s.name == "make_folder" and s.verify == "folder_exists"
    assert s.verify_args["path"].endswith("/Journals")
    await s.action()
    assert rec.calls[0][0] == "make_folder"


@pytest.mark.asyncio
async def test_open_url_step_verified_with_browser_tabs():
    rec = _RecorderAdapter()
    steps = await DesktopPlanner(adapter=rec).plan("open https://docs.google.com in Chrome")
    s = steps[0]
    assert s.name == "open_url" and s.verify == "browser_tabs" and s.verify_args == {"min_count": 1}
    await s.action()
    assert rec.calls == [("open_url", "https://docs.google.com")]


@pytest.mark.asyncio
async def test_set_wallpaper_step():
    rec = _RecorderAdapter()
    steps = await DesktopPlanner(adapter=rec).plan("set the wallpaper to an eagle")
    assert steps[0].name == "set_wallpaper"
    await steps[0].action()
    assert rec.calls[0][0] == "set_wallpaper"


# ── runs through the executor with verification ───────────────────────────

@pytest.mark.asyncio
async def test_desktop_step_runs_through_fluid_executor():
    from core.skills.fluid_executor import FluidExecutor

    class _OkVerifier:
        async def verify(self, predicate, args=None):
            from types import SimpleNamespace

            return SimpleNamespace(success=True, detail=predicate)

    rec = _RecorderAdapter()
    steps = await DesktopPlanner(adapter=rec).plan("open Notes")
    ex = FluidExecutor(verifier=_OkVerifier())
    receipt = await ex.run("open notes", steps)
    assert receipt.completed and receipt.verified_progress == 1
    assert rec.calls == [("open_app", "Notes")]


# ── GoalPlanner routes desktop goals here ─────────────────────────────────

@pytest.mark.asyncio
async def test_goal_planner_routes_desktop_intent():
    from core.agency.goal_planner import GoalPlanner

    gp = GoalPlanner()
    assert gp.classify("open Notes") == "desktop"
    steps = await gp.plan("open Notes")
    assert steps and steps[0].name == "open_app"
