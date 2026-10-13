"""What her reading of a world's code says is what the code does when it runs.

core/perception/reading_a_world_s_code.py reads a world's scripts without running them: how fast each key moves her
and where she is kept, the grid a floor is painted in and how much of it wins, the clock, the lives, what each kind
of thing does when it touches her and how fast it goes, a jump's lift and the pull after it, a lander's pushes, a
car's turning and driving. Each of those is a claim about the running world, and here each is put to it: the world
is run from its own code (tools/oracles/a_world_run_from_its_code.js, the same world a browser runs), set up so the
claim can fail, and what happens is compared with what was read. Every world in tests/fixtures/worlds_that_move is
read, and a finding no world bears out fails here with the world's own numbers.
"""
from __future__ import annotations

import math
import re
from pathlib import Path

import pytest

from core.perception.reading_a_world_s_code import read_a_world_s_code
from tools.oracles.a_world_run_from_its_code import FRAME_S, WorldRunFromItsCode, node_is_here

pytestmark = [pytest.mark.unit, pytest.mark.skipif(not node_is_here(), reason="node is not installed")]

WORLDS = Path(__file__).parent / "fixtures" / "worlds_that_move"
EVERY_WORLD = sorted(path.stem for path in WORLDS.glob("*.html"))
KEYS = {"left": "ArrowLeft", "right": "ArrowRight", "up": "ArrowUp", "down": "ArrowDown", "space": " "}


def _read(world: str):
    page = (WORLDS / f"{world}.html").read_text()
    return read_a_world_s_code([(WORLDS / "world.js").read_text(), *re.findall(r"<script>([\s\S]*?)</script>", page)])


READ = {world: _read(world) for world in EVERY_WORLD}


def _with(what) -> list[str]:
    return [world for world in EVERY_WORLD if what(READ[world])]


def _her(w: WorldRunFromItsCode, body: str) -> str:
    """Her body in the running world, as an expression: one of a list where the code moves one of several."""
    if not body.endswith("[]"):
        return f"__world.game.{body}"
    many = body[:-2]
    before = w.value(f"__world.game.{many}.map((b) => [b.x, b.y])")
    w.hold("ArrowRight")
    w.run(3 * FRAME_S)
    w.hold()
    after = w.value(f"__world.game.{many}.map((b) => [b.x, b.y])")
    moved = next(i for i, (a, b) in enumerate(zip(before, after, strict=True)) if a != b)
    return f"__world.game.{many}[{moved}]"


def _place(w: WorldRunFromItsCode, her: str, x: float, y: float) -> None:
    w.value(f"({her}.x = {x}, {her}.y = {y}, 0)")


def _where(w: WorldRunFromItsCode, her: str) -> tuple[float, float]:
    return tuple(w.value(f"[{her}.x, {her}.y]"))


def _spots(hers) -> list[tuple[float, float] | None]:
    """Places to measure from, across where she is kept: the middle first. A guard that sees her or a rock that hits
    her puts her back and spoils a measurement, so another place is tried."""
    if hers.kept_inside is None:
        return [None] * 4
    left, top, right, bottom = hers.kept_inside
    shares = (0.5, 0.3, 0.7, 0.1, 0.9)
    return [(left + a * (right - left), top + b * (bottom - top)) for b in shares for a in shares]


def _middle(hers) -> tuple[float, float] | None:
    if hers.kept_inside is None:
        return None
    left, top, right, bottom = hers.kept_inside
    return (left + right) / 2, (top + bottom) / 2


@pytest.mark.parametrize("world", _with(lambda r: r.hers is not None and r.hers.paces))
def test_each_key_moves_her_as_fast_and_the_way_her_reading_says(world):
    hers = READ[world].hers
    with WorldRunFromItsCode(world, seed=5) as w:
        w.value("(__world.lives = 1e9, 0)")
        her = _her(w, hers.name)
        for key, (vx, vy) in hers.paces.items():
            clean = False
            for spot in _spots(hers):
                if spot is not None:
                    _place(w, her, *spot)
                lives, (x0, y0) = w.value("__world.lives"), _where(w, her)
                w.hold(KEYS.get(key, key))
                w.run(0.1)
                w.hold()
                x1, y1 = _where(w, her)
                if w.value("__world.lives") == lives:
                    clean = True                # nothing touched her and put her back meanwhile
                    break
            assert clean, (world, key, "every measurement was spoilt")
            assert math.isclose((x1 - x0) / 0.1, vx, abs_tol=0.01), (world, key, (x1 - x0) / 0.1, vx)
            assert math.isclose((y1 - y0) / 0.1, vy, abs_tol=0.01), (world, key, (y1 - y0) / 0.1, vy)


@pytest.mark.parametrize("world", _with(lambda r: r.hers is not None and r.hers.kept_inside is not None))
def test_she_is_kept_where_her_reading_says_and_no_further(world):
    """Pushed against each side from a few places along it, she never goes past where the reading keeps her, and from
    one at least she gets there: the bound is the code's, not a guess inside it."""
    hers = READ[world].hers
    left, top, right, bottom = hers.kept_inside
    sides = {"left": (0, left, min), "right": (0, right, max), "up": (1, top, min), "down": (1, bottom, max)}
    for key, (axis, bound, _most) in sides.items():
        if key not in hers.paces:
            continue
        reached = False
        for share in (0.15, 0.5, 0.85):
            with WorldRunFromItsCode(world, seed=5) as w:
                w.value("(__world.lives = 1e9, 0)")
                her = _her(w, hers.name)
                spot = [0.0, 0.0]
                spot[axis] = bound + (-20 if key in ("right", "down") else 20)
                spot[1 - axis] = (left + share * (right - left)) if axis else (top + share * (bottom - top))
                _place(w, her, *spot)
                w.hold(KEYS[key])
                for _ in range(10):
                    w.run(0.1)
                    at = _where(w, her)[axis]
                    assert (at >= bound - 1e-6) if key in ("left", "up") else (at <= bound + 1e-6), (world, key, at, bound)
                w.hold()
                reached = reached or math.isclose(_where(w, her)[axis], bound, abs_tol=1e-6)
        assert reached, (world, key, bound)


@pytest.mark.parametrize("world", _with(lambda r: r.ground is not None))
def test_the_cell_painted_under_her_is_the_cell_her_reading_says(world):
    ground, hers = READ[world].ground, READ[world].hers
    columns, rows = ground.cells
    grid, count = f"__world.game.{ground.kept_in}", f"__world.game.{ground.counted_by}"
    left, top, right, bottom = hers.kept_inside
    with WorldRunFromItsCode(world, seed=5) as w:
        w.value("(__world.lives = 1e9, 0)")
        her = _her(w, hers.name)
        for i in range(240):
            x = left + (right - left) * ((i * 0.6180339887) % 1.0)
            y = top + (bottom - top) * ((i * 0.7548776662) % 1.0)
            if i < 4:
                x, y = (left, right)[i % 2], (top, bottom)[i // 2]
            w.value(f"({grid}.fill(0), {count} = 0, 0)")
            _place(w, her, x, y)
            w.run(FRAME_S)
            marked = [k for k, v in enumerate(w.value(f"Array.from({grid})")) if v]
            said = ground.cell_of(x + ground.marks_at[0], y + ground.marks_at[1])
            assert marked == ([said[1] * columns + said[0]] if said else []), (world, (x, y), marked, said)
        assert len(w.value(f"Array.from({grid})")) == columns * rows


@pytest.mark.parametrize("world", _with(lambda r: r.ground is not None and r.ground.wins_at))
def test_the_share_painted_that_wins_is_the_share_her_reading_says(world):
    ground, hers = READ[world].ground, READ[world].hers
    need = math.ceil(ground.wins_at * ground.count - 1e-9)
    grid, count = f"__world.game.{ground.kept_in}", f"__world.game.{ground.counted_by}"
    for before, wins in ((need - 2, False), (need - 1, True)):
        with WorldRunFromItsCode(world, seed=5) as w:
            w.value("(__world.lives = 1e9, 0)")
            her = _her(w, hers.name)
            w.value(f"({grid}.fill(0), {count} = {before}, 0)")
            _place(w, her, *_middle(hers))
            w.run(FRAME_S)
            state = w.state()["world"]
            assert (state["state"] == "over" and state["ended"][-1]["why"] == "won") is wins, (world, before, state["state"])


@pytest.mark.parametrize("world", _with(lambda r: r.clock_s))
def test_the_game_ends_when_her_reading_says_its_clock_runs_out(world):
    clock = READ[world].clock_s
    with WorldRunFromItsCode(world, seed=5) as w:
        w.value("(__world.lives = 1e9, 0)")
        w.run(clock - 0.5)
        assert w.value("__world.state") == "play", world
        w.run(1.0)
        ended = w.value("__world.ended")
        assert ended and math.isclose(ended[-1]["seconds"], clock, abs_tol=2 * FRAME_S), (world, ended, clock)


@pytest.mark.parametrize("world", _with(lambda r: r.lives is not None))
def test_she_has_as_many_lives_as_her_reading_says(world):
    with WorldRunFromItsCode(world, seed=5) as w:
        assert w.value("__world.lives") == READ[world].lives


@pytest.mark.parametrize(("world", "kind"), [(world, t.name) for world in EVERY_WORLD for t in READ[world].things
                                             if t.touching and READ[world].hers is not None])
def test_touching_her_does_what_her_reading_says(world, kind):
    """A thing of the kind put on her: a life goes, or the score goes up, as read; where the reading says either, each
    touch does one of the two and between them both are seen."""
    thing = READ[world].thing(kind)
    said = set(thing.touching.split(" or "))
    seen: set[str] = set()
    with WorldRunFromItsCode(world, seed=5) as w:
        w.value("(__world.lives = 1e9, 0)")
        her = _her(w, READ[world].hers.name)
        many = f"__world.game.{kind}"
        for _ in range(40):
            for _wait in range(80):
                if w.value(f"{many}.length"):
                    break
                w.run(0.1)
            if not w.value(f"{many}.length") and thing.size:
                # None has come by itself (a star behind a door she has not opened): one is put there, of its size.
                w.value(f"({many}.push({{x: 0, y: 0, w: {thing.size[0]}, h: {thing.size[1]}}}), 0)")
            x, y = _where(w, her)
            lives, score = w.value("__world.lives"), w.value("__world.score")
            w.value(f"({many}[0].x = {x}, {many}[0].y = {y}, {many}.length)")
            w.run(FRAME_S)
            if w.value("__world.lives") < lives:
                seen.add("costs a life")
            elif w.value("__world.score") > score:
                seen.add("scores")
            w.run(1.5)
            if seen == said:
                break
    assert seen == said, (world, kind, seen, said)


@pytest.mark.parametrize(("world", "kind"), [(world, t.name) for world in EVERY_WORLD for t in READ[world].things
                                             if t.speed])
def test_nothing_of_a_kind_goes_faster_than_her_reading_says_and_some_go_near_it(world, kind):
    speed = READ[world].thing(kind).speed
    # What only comes of a key of hers (a shot) is made by pressing it now and then.
    makers = [KEYS.get(k, k) for k, does in READ[world].keys.items() if does == "acts"]
    fastest = 0.0
    with WorldRunFromItsCode(world, seed=7) as w:
        w.value("(__world.lives = 1e9, 0)")
        before = None
        for frame in range(int(30 / FRAME_S)):
            if makers and frame % 20 == 0:
                w.hold(*makers) if frame % 40 == 0 else w.hold()
            now = w.value(f"(__world.game.{kind} || []).map((t) => [t.x, t.y || 0])")
            if before:
                for x, y in now:
                    near = min((math.hypot(x - a, y - b) for a, b in before), default=math.inf)
                    if near < 3 * speed * FRAME_S:
                        fastest = max(fastest, near / FRAME_S)
            before = now
            w.run(FRAME_S)
    assert fastest <= speed * 1.001 + 1e-6, (world, kind, fastest, speed)
    assert fastest >= 0.8 * speed, (world, kind, fastest, speed)


@pytest.mark.parametrize("world", _with(lambda r: r.hers is not None and r.hers.lifts))
def test_a_jump_lifts_her_and_she_falls_as_her_reading_says(world):
    hers = READ[world].hers
    with WorldRunFromItsCode(world, seed=5) as w:
        her = _her(w, hers.name)
        w.run(0.5)
        for key, lift in hers.lifts.items():
            w.hold(KEYS.get(key, key))
            w.run(FRAME_S)
            w.hold()
            first = w.value(f"{her}.vy")
            assert math.isclose(first, -lift + hers.pull * FRAME_S, abs_tol=1e-6), (world, key, first, lift)
            w.run(FRAME_S)
            assert math.isclose(w.value(f"{her}.vy") - first, hers.pull * FRAME_S, abs_tol=1e-6), world


@pytest.mark.parametrize("world", _with(lambda r: r.hers is not None and r.hers.pushes))
def test_each_key_pushes_her_as_her_reading_says(world):
    hers = READ[world].hers
    with WorldRunFromItsCode(world, seed=5) as w:
        her = _her(w, hers.name)
        for key, (ax, ay) in hers.pushes.items():
            w.value(f"({her}.x = 200, {her}.y = 60, {her}.vx = 0, {her}.vy = 0, 0)")
            w.hold(*([KEYS.get(key, key)] if key else []))
            w.run(0.2)
            w.hold()
            vx, vy = w.value(f"[{her}.vx, {her}.vy]")
            assert math.isclose(vx / 0.2, ax, abs_tol=0.01) and math.isclose(vy / 0.2, ay, abs_tol=0.01), (world, key, vx, vy)


@pytest.mark.parametrize("world", _with(lambda r: r.hers is not None and r.hers.turns))
def test_each_key_turns_and_drives_her_as_her_reading_says(world):
    hers = READ[world].hers
    with WorldRunFromItsCode(world, seed=5) as w:
        her = _her(w, hers.name)
        for key, rate in hers.turns.items():
            a0 = w.value(f"{her}.a")
            w.hold(KEYS[key])
            w.run(0.5)
            w.hold()
            assert math.isclose((w.value(f"{her}.a") - a0) / 0.5, rate, rel_tol=1e-6), (world, key)
        backward, forward = hers.speeds
        for key, gathers in hers.drives.items():
            w.value(f"({her}.v = 0, {her}.x = 240, {her}.y = 180, 0)")
            w.hold(KEYS[key])
            w.run(0.1)
            assert math.isclose(w.value(f"{her}.v") / 0.1, gathers, rel_tol=1e-6), (world, key)
            w.run(3.0)
            w.hold()
            assert math.isclose(w.value(f"{her}.v"), forward if gathers > 0 else backward, abs_tol=1e-6), (world, key)


def test_every_world_her_body_s_physics_are_in_is_read():
    """The worlds where keys move a body of hers are each read for how: paces, pushes, a lift or turns. A world played
    by cards, a pointer, a box to type in or presses in time has no such body, and is not counted here."""
    bodied = [w for w in EVERY_WORLD if READ[w].hers is not None and (READ[w].hers.paces or READ[w].hers.pushes
                                                                      or READ[w].hers.lifts or READ[w].hers.turns)]
    assert set(bodied) >= {"brawl", "bump", "catch", "charge", "collect", "dodge", "doors", "guard", "held", "jump",
                           "lander", "paddle", "paint", "rooms", "shooter", "steps", "storm", "switch"}, bodied
