"""Her own choices, made by her own code, on a world run from its code: the piece of her that chooses, put to the
world exactly, with nothing else of her in the way.

What she would have measured is handed to her choosing as if she had measured it perfectly: each thing where the
world's code has it, going at what its last two places say, and her keys' paces as her learner reports them once it
knows. So whether she wins comes down to the piece being tested (where she chooses to go, what she holds) and to
what she was given to choose with (the ground her code names, or cells of her own size). Two runs differing only in
that are the test of what it is worth.

    with HerChoicesOnAWorld("paint", seed=3, ground_from_code=True) as play:
        came_to = play.run(150.0)
"""
from __future__ import annotations

import math
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from tools.oracles.a_world_run_from_its_code import FRAME_S, WorldRunFromItsCode

WORLDS = Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "worlds_that_move"
#: Her working picture of a 480 x 320 world: 240 x 160.
SHAPE = (160, 240)
KEYS = {"left": "ArrowLeft", "right": "ArrowRight", "up": "ArrowUp", "down": "ArrowDown", "space": " "}


def reading_of(world: str) -> Any:
    from core.perception.reading_a_world_s_code import read_a_world_s_code

    page = (WORLDS / f"{world}.html").read_text()
    return read_a_world_s_code([(WORLDS / "world.js").read_text(), *re.findall(r"<script>([\s\S]*?)</script>", page)])


class _HerKeys:
    """What her learner reports of her body once it knows: which thing is hers and each key's pace, exactly."""

    number, kind, carried, follows_pointer, follows_along = 1, 0, False, False, (False, False)

    def __init__(self, paces: dict[str, tuple[float, float]], scale: float) -> None:
        self.paces = {k: (v[0] * scale, v[1] * scale) for k, v in paces.items()}
        self.makes: dict[str, Any] = {}
        self.last_velocity = (0.0, 0.0)
        self.lowest, self.highest = [0.0, 0.0], [float(SHAPE[1]), float(SHAPE[0])]

    def thing(self, moves: Any) -> Any:
        return moves.things.get(1)

    def keys_that_move_her(self, keys: list[str]) -> dict[str, tuple[float, float]]:
        return {k: v for k, v in self.paces.items() if k in keys}

    def axes(self, keys: list[str]) -> tuple[bool, bool]:
        ways = self.keys_that_move_her(keys).values()
        return any(abs(vx) > 0 for vx, _ in ways), any(abs(vy) > 0 for _, vy in ways)

    def push_of(self, key: str) -> None:
        return None


def _a_thing(number: int, kind: int, box: tuple[float, float, float, float], scale: float) -> Any:
    from core.perception.what_moves_in_the_picture import Thing

    x, y, w, h = box
    return Thing(number, (x + w / 2) * scale, (y + h / 2) * scale, w * scale, h * scale, np.zeros(64), (200, 200, 200),
                 np.zeros((1, 1, 3), dtype=np.uint8), 0.0, 0.0, kind=kind, moved=True)


class HerChoicesOnAWorld:
    """One world, run from its code, played by her choosing (see the module's words)."""

    def __init__(self, world: str, *, seed: int = 1, ground_from_code: bool = True, decide_every: float = 1 / 30,
                 reading: Any = None, late_by: int = 0) -> None:
        self.world, self.seed = world, seed
        self.reading = reading if reading is not None else reading_of(world)
        self.ground_from_code = ground_from_code
        self.decide_every = decide_every
        #: How many decisions late what she chooses reaches the world: the trip a key takes through her hands.
        self.late_by = late_by
        self.scale = SHAPE[1] / self.reading.stage[0]
        self._oracle = WorldRunFromItsCode(world, seed=seed)
        self.trace: list[dict[str, Any]] = []

    def __enter__(self) -> HerChoicesOnAWorld:
        self._oracle.__enter__()
        return self

    def __exit__(self, *exc: Any) -> None:
        self._oracle.__exit__(*exc)

    @property
    def oracle(self) -> WorldRunFromItsCode:
        return self._oracle

    def _things(self, state: dict[str, Any], before: dict[int, tuple[float, float]]) -> dict[int, Any]:
        """Each thing where the world's code has it, numbered for the run, going at what its last two places say."""
        hers = self.reading.hers
        game = state["game"]
        body = game[hers.name]
        w, h = body.get("w", hers.size[0]), body.get("h", hers.size[1])
        things = {1: _a_thing(1, 0, (body["x"], body["y"], w, h), self.scale)}
        number = 10
        for kind, of in enumerate(self.reading.things, start=1):
            for item in game.get(of.name) or []:
                if not isinstance(item, dict) or "x" not in item or "y" not in item:
                    continue
                size = (item.get("w") or (of.size or (12, 12))[0], item.get("h") or (of.size or (12, 12))[1])
                thing = _a_thing(number, kind, (item["x"], item["y"], *size), self.scale)
                near = min(before.items(), key=lambda kv: math.dist(kv[1], (thing.x, thing.y)), default=None)
                if near is not None and math.dist(near[1], (thing.x, thing.y)) < 12:
                    thing.vx = (thing.x - near[1][0]) / self.decide_every
                    thing.vy = (thing.y - near[1][1]) / self.decide_every
                things[number] = thing
                number += 1
        return things

    def run(self, seconds: float, *, each: Any = None) -> dict[str, Any]:
        """Played for ``seconds`` or until the world's game ends; ``each`` is called at every choice with the ground
        she is going over (None where she has none of the code's) and the world's state, for a check that runs along."""
        from core.agency.choosing_where_to_be import _Choosing
        from core.agency.ground_her_code_names import ground_in_her_picture
        from core.agency.what_meeting_things_does import AVOID, MEET, WhatMeetingDoes

        hers = _HerKeys(self.reading.hers.paces, self.scale)
        keys = list(hers.paces)
        meeting = WhatMeetingDoes()
        for kind, of in enumerate(self.reading.things, start=1):
            if of.touching == "costs a life":
                meeting.told[kind] = AVOID
            elif of.touching == "scores":
                meeting.told[kind] = MEET
        ground = ground_in_her_picture(self.reading, SHAPE) if self.ground_from_code else None
        covered: set[tuple[int, int]] = set()
        cell = None
        held = ""
        on_the_way: list[str] = []
        state = self._oracle.state()
        before: dict[int, tuple[float, float]] = {}
        at = 0.0
        while at < seconds and state["world"]["state"] == "play":
            things = self._things(state, before)
            before = {n: (t.x, t.y) for n, t in things.items() if n != 1}
            moves = SimpleNamespace(shape=SHAPE, things=things, kinds=[], out_of_sight={})
            mine = things[1]
            if ground is not None:
                ground.marked_by(mine)
                if each is not None:
                    each(ground, state)
            else:
                cell = cell or max(4.0, mine.w, mine.h)
                covered.add((int(mine.x // cell), int(mine.y // cell)))
            # How late her keys reach the world, as live play measures it and hands it to her choosing.
            choosing = _Choosing(moves, hers, meeting, keys, None, next_picture_s=self.decide_every,
                                 response_s=self.late_by * self.decide_every,
                                 covered=(covered, cell) if ground is None else None)
            choosing.ground = ground
            key, why, _aim = choosing.key(held)
            on_the_way.append(key)
            arriving = on_the_way.pop(0) if len(on_the_way) > self.late_by else held
            if arriving != held:
                self._oracle.hold(*([KEYS.get(arriving, arriving)] if arriving else []))
                held = arriving
            hers.last_velocity = hers.paces.get(held, (0.0, 0.0)) if held else (0.0, 0.0)
            if len(self.trace) < 100000:
                self.trace.append({"at": round(at, 3), "key": key, "why": why, "x": mine.x, "y": mine.y})
            state = self._oracle.step(self.decide_every, frame=FRAME_S)
            at += self.decide_every
        tally = state["world"]
        return {"world": self.world, "seed": self.seed, "ground_from_code": self.ground_from_code, "seconds": round(at, 2),
                "state": tally["state"], "ended": tally.get("ended") or [], "score": tally.get("score"),
                "lives": tally.get("lives"), "painted": tally.get("painted")}
