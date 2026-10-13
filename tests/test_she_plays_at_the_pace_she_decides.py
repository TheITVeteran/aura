"""She plans only what she can do at the pace she decides, and knows the ground she has gone over between pictures.

Played whole from its title screen here (tools/measure_whole_play_on_worlds.py), her pictures came three or four a
second, and the paint world's own code (tools/oracles/her_choices_on_a_world.py) at that pace showed three faults:
plans that changed keys a sixth of a second ahead, when her next decision was a third of a second ahead, so that she
went right and left on the spot while a bug came to her; cells she walked over between two pictures, never seen under
her, that she went back over again and again; and a cell her steps kept carrying her past, gone for one way and then
the other for fifteen seconds. Measured on seeds 1 to 6, each decision a decision late and the waits between them
uneven by up to 30%, with the floor's marks seen: at three decisions a second she won all six, the slowest in 69 s,
and lost six lives between them; at two, she won two. With plans that change before her next decision she won five
at three a second, losing eight lives, and at two she won none and lost every life.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from core.agency.ground_her_code_names import GroundInHerPicture
from core.agency.imagining_ahead import CHANGES_AT, plans_to_imagine

pytestmark = [pytest.mark.unit]


def _ground() -> GroundInHerPicture:
    return GroundInHerPicture(origin=(0.0, 0.0), cell=(10.0, 10.0), cells=(10, 6))


def _at(x: float, y: float, seen: float = 0.0):
    return SimpleNamespace(x=x, y=y, seen=seen)


def test_no_plan_changes_her_keys_before_her_next_decision():
    every = 1 / 3
    plans = plans_to_imagine(["", "left", "right"], ["up"], "left", every=every)
    changes = [when for plan in plans for when, _key in plan.held[1:]] + [when for plan in plans for when, _ in plan.presses]
    assert changes and all(when == 0.0 or when >= every for when in changes)
    # Deciding faster than any change comes, the plans are as they were.
    quick = plans_to_imagine(["", "left", "right"], ["up"], "left", every=1 / 30)
    assert sorted({when for plan in quick for when, _ in plan.held[1:]}) == sorted(CHANGES_AT)


def test_the_cells_she_went_along_between_two_pictures_are_gone_over():
    ground = _ground()
    ground.marked_by(_at(15.0, 25.0))
    ground.marked_by(_at(55.0, 25.0))                       # along row 2, from column 1 to column 5
    assert ground.marked == {(1, 2), (2, 2), (3, 2), (4, 2), (5, 2)}
    ground.marked_by(_at(55.0, 55.0))                       # down column 5, rows 2 to 5
    assert {(5, 3), (5, 4), (5, 5)} <= ground.marked


def test_what_she_cannot_be_sure_she_went_over_is_not_taken_as_gone_over():
    ground = _ground()
    ground.marked_by(_at(15.0, 15.0))
    ground.marked_by(_at(45.0, 45.0))                       # across, not along a row or a column
    assert ground.marked == {(1, 1), (4, 4)}
    ground = _ground()
    ground.marked_by(_at(30.0, 5.0))
    ground.marked_by(_at(30.0, 55.0))                       # down the edge between columns 2 and 3: only where seen
    assert ground.marked == {(3, 0), (3, 5)}
    ground = _ground()
    ground.marked_by(_at(5.0, 5.0))
    ground.marked_by(_at(95.0, 5.0))                        # further than she walks between two pictures
    assert ground.marked == {(0, 0), (9, 0)}


def test_a_cell_she_keeps_going_for_and_missing_is_left_for_a_while():
    from core.agency.ground_her_code_names import LEFT_FOR_S, MISSED_AFTER_S

    ground = _ground()
    ground.marked = {(c, r) for c in range(10) for r in range(6)} - {(3, 3), (6, 3), (9, 5)}
    at = 0.0
    while at <= MISSED_AFTER_S + 0.5:
        ground.now = at
        ground.going_to((3, 3) if int(at * 4) % 2 else (6, 3), near=True)    # one, then the other, picture by picture
        at += 0.25
    assert ground.left_to_do() == [(9, 5)]
    ground.now = at + LEFT_FOR_S
    assert set(ground.left_to_do()) == {(3, 3), (6, 3), (9, 5)}
    # Left with nothing else to go for, she goes for them again.
    ground.marked.add((9, 5))
    ground.left_until = {(3, 3): ground.now + 5.0, (6, 3): ground.now + 5.0}
    assert set(ground.left_to_do()) == {(3, 3), (6, 3)}


def test_what_the_picture_shows_of_the_ground_is_what_is_marked_and_what_is_hidden_stays():
    ground = _ground()
    ground.marked = {(0, 0), (1, 0)}
    ground.saw(marked={(2, 0), (3, 0)}, looked_at={(0, 0), (2, 0), (3, 0)})
    assert ground.marked == {(1, 0), (2, 0), (3, 0)}            # (1, 0) was hidden: as it was


def test_the_marks_on_the_paint_world_s_canvas_are_read_by_the_colour_its_code_paints_with():
    import asyncio
    import base64
    import io
    import re
    from pathlib import Path

    import numpy as np

    playwright_api = pytest.importorskip("playwright.async_api")
    from PIL import Image

    from core.agency.ground_her_code_names import ground_in_her_picture
    from core.perception.reading_a_world_s_code import read_a_world_s_code

    worlds = Path(__file__).parent / "fixtures" / "worlds_that_move"
    page_text = (worlds / "paint.html").read_text()
    paint = read_a_world_s_code([(worlds / "world.js").read_text(), *re.findall(r"<script>([\s\S]*?)</script>", page_text)])
    painted = {(c, r) for r in range(14) for c in range(24) if (3 * c + 7 * r) % 5 < 2}

    async def drawn():
        async with playwright_api.async_playwright() as playwright:
            try:
                browser = await playwright.chromium.launch(headless=True)
            except Exception as why:  # noqa: BLE001 - no engine installed here
                pytest.skip(f"no browser engine: {why}")
            page = await browser.new_page(viewport={"width": 480, "height": 400})
            await page.goto(f"file://{worlds / 'paint.html'}?seed=3&start=1")
            cells = [r * 24 + c for c, r in painted]
            await page.evaluate(f"""(() => {{ const g = __world.game; __world.lives = 1e9;
                g.painted.fill(0); for (const k of {cells}) g.painted[k] = 1; g.me.x = 300; g.me.y = 200; return 0; }})()""")
            await asyncio.sleep(0.3)
            # Read in the same frame as where everything stands.
            url, state = await page.evaluate("""(() => { const g = __world.game;
                return [document.querySelector('canvas').toDataURL('image/png'),
                        {me: [g.me.x, g.me.y, 16, 16], bugs: g.bugs.map((b) => [b.x, b.y, b.w, b.h]),
                         painted: Array.from(g.painted)}]; })()""")
            await browser.close()
            return np.asarray(Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1]))).convert("RGB")), state

    picture, state = asyncio.run(drawn())
    shape = (160, 240)
    scale = shape[1] / 480
    ground = ground_in_her_picture(paint, shape)
    things = [SimpleNamespace(x=(x + w / 2) * scale, y=(y + h / 2) * scale, w=w * scale, h=h * scale, colour=(200, 200, 200))
              for x, y, w, h in [state["me"], *state["bugs"]]]
    ground.seen_in(picture, shape, things)
    truth = {(k % 24, k // 24) for k, v in enumerate(state["painted"]) if v}
    hidden = truth ^ ground.marked
    # Every cell is read as the world has it, but those under her and the bugs, which are not read at all.
    assert len(ground.marked) >= len(truth) - 8 and ground.marked <= truth, (len(truth), len(ground.marked))
    for column, row in hidden:
        x0, y0 = ground.origin[0] + column * ground.cell[0], ground.origin[1] + row * ground.cell[1]
        assert any(x0 < t.x + t.w / 2 and t.x - t.w / 2 < x0 + ground.cell[0]
                   and y0 < t.y + t.h / 2 and t.y - t.h / 2 < y0 + ground.cell[1] for t in things), (column, row)


def _played(rate: float, *, seen: bool = True) -> list[dict]:
    from tools.oracles.her_choices_on_a_world import HerChoicesOnAWorld

    out = []
    for seed in range(1, 7):
        with HerChoicesOnAWorld("paint", seed=seed, decide_every=1 / rate, late_by=1, jitter=0.3,
                                swerves_from_code=True, marks_seen=seen) as play:
            out.append(play.run(150.0))
    return out


def _node():
    from tools.oracles.a_world_run_from_its_code import node_is_here

    if not node_is_here():
        pytest.skip("node is not installed")


@pytest.mark.slow
def test_at_three_decisions_a_second_she_paints_the_floor_on_every_seed():
    _node()
    played = _played(3.0)
    assert all(came["ended"] and came["ended"][-1]["why"] == "won" for came in played), played
    assert sum(3 - came["lives"] for came in played) <= 6, played


@pytest.mark.slow
def test_with_plans_that_change_before_her_next_decision_she_is_caught(monkeypatch):
    import core.agency.imagining_ahead as imagining

    _node()
    at_her_pace = _played(2.0)
    original = imagining.plans_to_imagine
    monkeypatch.setattr(imagining, "plans_to_imagine",
                        lambda choices, presses, held, every=0.0: original(choices, presses, held, every=0.0))
    too_soon = _played(2.0)

    def won(played):
        return sum(1 for came in played if came["ended"] and came["ended"][-1]["why"] == "won")

    assert won(at_her_pace) >= 1, at_her_pace
    assert won(too_soon) == 0 and all(came["lives"] == 0 for came in too_soon), too_soon
