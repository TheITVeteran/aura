"""A test world's own code, run on a clock this side drives: the ground truth anything said of the world is checked by.

``tools/oracles/a_world_run_from_its_code.js`` runs a world's two scripts as a page runs them, with nothing drawn and a
clock that moves only when told to. This hands it keys, the pointer and time, and reads back the world's tally and the
game's own state. It is for tests and measurements; Aura never runs it, and nothing she reads is run.

    with WorldRunFromItsCode("paint", seed=3) as world:
        world.hold("ArrowRight")
        after = world.step(1.0)
        after["game"]["me"]["x"]
"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
#: The world's frames: sixty a second, as a browser on a desktop draws.
FRAME_S = 1 / 60


def node_is_here() -> bool:
    return shutil.which("node") is not None


class WorldRunFromItsCode:
    """One world, loaded from its page and running on a driven clock."""

    def __init__(self, world: str, *, seed: int = 1, start: bool = True, query: str = "") -> None:
        self.world = world
        parts = [f"seed={seed}"] + (["start=1"] if start else []) + ([query] if query else [])
        self.query = "&".join(parts)
        self._process: subprocess.Popen[str] | None = None
        self.held: set[str] = set()
        self.pointer = (240.0, 160.0)

    def __enter__(self) -> WorldRunFromItsCode:
        node = shutil.which("node")
        if node is None:
            raise RuntimeError("node is not installed")
        self._process = subprocess.Popen([node, str(HERE / "a_world_run_from_its_code.js")], stdin=subprocess.PIPE,
                                         stdout=subprocess.PIPE, text=True, bufsize=1)
        self._ask({"op": "load", "world": self.world, "query": self.query})
        return self

    def __exit__(self, *_exc: Any) -> None:
        if self._process is not None:
            self._process.stdin.close()
            self._process.wait(timeout=10)
            self._process = None

    def _ask(self, asked: dict[str, Any]) -> dict[str, Any]:
        assert self._process is not None and self._process.stdin is not None and self._process.stdout is not None
        self._process.stdin.write(json.dumps(asked) + "\n")
        self._process.stdin.flush()
        answer = json.loads(self._process.stdout.readline())
        if not answer.get("ok"):
            raise RuntimeError(f"the world's code answered with an error: {answer.get('error')}")
        return answer

    def press(self, key: str) -> None:
        if key not in self.held:
            self.held.add(key)
            self._ask({"op": "key", "type": "keydown", "key": key})

    def release(self, key: str) -> None:
        if key in self.held:
            self.held.discard(key)
            self._ask({"op": "key", "type": "keyup", "key": key})

    def hold(self, *keys: str) -> None:
        """Exactly these keys held from now: the rest let go."""
        for key in sorted(self.held - set(keys)):
            self.release(key)
        for key in keys:
            self.press(key)

    def tap(self, key: str) -> None:
        self.press(key)
        self.step(FRAME_S)
        self.release(key)

    def point(self, x: float, y: float) -> None:
        self.pointer = (float(x), float(y))
        self._ask({"op": "mouse", "type": "mousemove", "x": x, "y": y})

    def mouse_down(self, x: float, y: float) -> None:
        self.point(x, y)
        self._ask({"op": "mouse", "type": "mousedown", "x": x, "y": y})

    def mouse_up(self) -> None:
        self._ask({"op": "mouse", "type": "mouseup", "x": self.pointer[0], "y": self.pointer[1]})

    def click(self, x: float, y: float) -> None:
        self.mouse_down(x, y)
        self.mouse_up()

    def step(self, seconds: float = FRAME_S, *, frame: float = FRAME_S) -> dict[str, Any]:
        """The world run on for ``seconds`` in frames of ``frame`` seconds; its state after."""
        frames = max(1, int(round(seconds / frame)))
        return self._ask({"op": "step", "dt": frame, "n": frames})["state"]

    def run(self, seconds: float, *, frame: float = FRAME_S) -> None:
        """The world run on, its state not read back: quicker where only the end matters."""
        self._ask({"op": "step", "dt": frame, "n": max(1, int(round(seconds / frame))), "quiet": True})

    def state(self) -> dict[str, Any]:
        return self._ask({"op": "state"})["state"]

    def value(self, expression: str) -> Any:
        """One value of the world, by an expression on its globals: ``__world.game.painted.length``."""
        return self._ask({"op": "eval", "js": expression})["value"]
