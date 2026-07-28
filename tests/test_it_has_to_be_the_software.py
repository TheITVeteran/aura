""""Pieces didnt move. Nothing was polished. Looked/felt horrible."

That was the verdict on a reconstruction that passed every gate this repository
had: the held-out battery said the rules were right, the play check said the
program ran, and what came out was a text grid driven with w/a/s/d. Every check
asked whether the program was *correct*. None asked whether it was the software.

For anything with a published surface that is not a detail. 2048 without its
tile palette, its arrow keys and its settling animation is a matrix
transformation with a printout — and the palette and the controls are public,
readable off a screenshot, which is exactly what makes them clean-room material.

So presentation is now a contract, graded like the rules: in another process,
against a recording stand-in for the toolkit, so nothing opens a window and the
real ``tkinter`` is never reachable by the candidate. A program that fails it is
not written to disk.
"""
from __future__ import annotations

import asyncio
import textwrap
from pathlib import Path

import pytest

from core.self_improvement.presentation_contract import (
    PresentationContract,
    colour_distance,
    grade_presentation,
    hex_colours_in,
    match_palette,
)
from core.self_improvement.program_materialization import (
    TWENTY_FORTY_EIGHT,
    materialize_program,
)

PALETTE = dict(TWENTY_FORTY_EIGHT.presentation.palette)


FAITHFUL = textwrap.dedent(
    '''
    import random
    import tkinter as tk

    SIZE = 4
    BG = "#bbada0"
    EMPTY = "#cdc1b4"
    TILE_COLOURS = {
        2: "#eee4da", 4: "#ede0c8", 8: "#f2b179", 16: "#f59563",
        32: "#f67c5f", 64: "#f65e3b", 128: "#edcf72", 256: "#edcc61",
        512: "#edc850", 1024: "#edc53f", 2048: "#edc22e",
    }


    def _slide(row):
        tiles = [v for v in row if v]
        out, gained, i = [], 0, 0
        while i < len(tiles):
            if i + 1 < len(tiles) and tiles[i] == tiles[i + 1]:
                out.append(tiles[i] * 2)
                gained += tiles[i] * 2
                i += 2
            else:
                out.append(tiles[i])
                i += 1
        out.extend([0] * (len(row) - len(out)))
        return out, gained


    def move(state):
        board = [list(r) for r in state["board"]]
        d = state["direction"]
        if d == "left":
            rows = board
        elif d == "right":
            rows = [list(reversed(r)) for r in board]
        elif d == "up":
            rows = [[board[r][c] for r in range(SIZE)] for c in range(SIZE)]
        else:
            rows = [[board[r][c] for r in range(SIZE - 1, -1, -1)] for c in range(SIZE)]
        total, moved = 0, []
        for row in rows:
            new, gained = _slide(row)
            total += gained
            moved.append(new)
        if d == "left":
            res = moved
        elif d == "right":
            res = [list(reversed(r)) for r in moved]
        elif d == "up":
            res = [[moved[c][r] for c in range(SIZE)] for r in range(SIZE)]
        else:
            res = [[moved[c][SIZE - 1 - r] for c in range(SIZE)] for r in range(SIZE)]
        return {"board": res, "score": total}


    def spawn_tile(board):
        empty = [(r, c) for r in range(SIZE) for c in range(SIZE) if not board[r][c]]
        if empty:
            r, c = random.choice(empty)
            board[r][c] = 4 if random.random() < 0.1 else 2
        return board


    def is_game_over(board):
        if any(0 in row for row in board):
            return False
        for r in range(SIZE):
            for c in range(SIZE):
                if c + 1 < SIZE and board[r][c] == board[r][c + 1]:
                    return False
                if r + 1 < SIZE and board[r][c] == board[r + 1][c]:
                    return False
        return True


    class Game2048:
        def __init__(self, root):
            self.root = root
            self.root.title("2048")
            self.score = 0
            self.board = [[0] * SIZE for _ in range(SIZE)]
            self.header = tk.Label(root, text="Score: 0")
            self.header.pack()
            self.canvas = tk.Canvas(root, width=440, height=440, bg=BG)
            self.canvas.pack()
            self.cells = []
            for r in range(SIZE):
                row = []
                for c in range(SIZE):
                    x, y = 10 + c * 107, 10 + r * 107
                    rect = self.canvas.create_rectangle(x, y, x + 97, y + 97, fill=EMPTY, width=0)
                    label = self.canvas.create_text(x + 48, y + 48, text="")
                    row.append((rect, label))
                self.cells.append(row)
            spawn_tile(self.board)
            spawn_tile(self.board)
            for seq in ("<Up>", "<Down>", "<Left>", "<Right>"):
                root.bind(seq, self.on_key)
            self.render()

        def render(self):
            for r in range(SIZE):
                for c in range(SIZE):
                    value = self.board[r][c]
                    rect, label = self.cells[r][c]
                    self.canvas.itemconfig(rect, fill=TILE_COLOURS.get(value, EMPTY))
                    self.canvas.itemconfig(label, text=str(value) if value else "")
            self.header.config(text="Score: %d" % self.score)

        def on_key(self, event):
            direction = {
                "Up": "up", "Down": "down", "Left": "left", "Right": "right"
            }.get(event.keysym)
            if not direction:
                return
            before = [list(r) for r in self.board]
            result = move({"board": self.board, "direction": direction})
            if result["board"] == before:
                return
            self.board = result["board"]
            self.score += result["score"]
            self.render()
            self.root.after(90, self.settle)

        def settle(self):
            spawn_tile(self.board)
            self.render()
            if is_game_over(self.board):
                self.header.config(text="Game over - Score: %d" % self.score)


    def main():
        root = tk.Tk()
        return Game2048(root)


    if __name__ == "__main__":
        main().root.mainloop()
    '''
).strip()


def _without(colours: list[str]) -> str:
    text = FAITHFUL
    for colour in colours:
        text = text.replace(f'"{colour}"', '"#888888"')
    return text


# ── The colour arithmetic the whole check rests on ────────────────────────

def test_the_palette_is_too_tightly_spaced_for_a_loose_tolerance() -> None:
    """Why the tolerance is 12 and not 60, measured on the real palette."""
    values = list(PALETTE.values())
    closest = min(
        colour_distance(values[i], values[j])
        for i in range(len(values))
        for j in range(i + 1, len(values))
    )
    assert closest < 12, "if neighbours were far apart, a loose tolerance would be fine"
    # At 60, the empty-cell grey alone claimed five of the eleven tiles.
    assert sum(1 for want in values if colour_distance("#cdc1b4", want) <= 60) >= 4
    assert sum(1 for want in values if colour_distance("#cdc1b4", want) <= 12) == 0


def test_one_colour_cannot_claim_the_whole_palette() -> None:
    """One-to-one assignment is what stops grey mush scoring as faithful."""
    matched, _ = match_palette(["#eee4da"], PALETTE, 30.0)
    assert len(matched) == 1


def test_the_published_colours_are_found_in_a_faithful_module() -> None:
    matched, _ = match_palette(hex_colours_in(FAITHFUL), PALETTE, 12.0)
    assert len(matched) == len(PALETTE)


# ── The grader, on real modules ───────────────────────────────────────────

def test_a_faithful_gui_passes() -> None:
    report = grade_presentation(FAITHFUL, TWENTY_FORTY_EIGHT.presentation)
    assert report.passed, [str(item) for item in report.findings]
    assert any("binds" in item for item in report.evidence)


def test_a_console_program_is_not_the_software() -> None:
    """It passed before: with no widgets built, every check was skipped."""
    report = grade_presentation(
        'def move(state):\n    return state\n\n\ndef main():\n    print("2048")\n',
        TWENTY_FORTY_EIGHT.presentation,
    )
    assert not report.passed
    assert any("no interface was built" in str(item) for item in report.findings)


@pytest.mark.parametrize(
    ("label", "mutate", "expect"),
    [
        ("grey tiles", lambda: _without(list(PALETTE.values())), "palette"),
        (
            "no controls",
            lambda: FAITHFUL.replace("root.bind(seq, self.on_key)", "pass"),
            "controls",
        ),
        (
            "nothing moves",
            lambda: FAITHFUL.replace("self.root.after(90, self.settle)", "self.settle()"),
            "motion",
        ),
        (
            "no score",
            lambda: FAITHFUL.replace("Score: ", "Value: ").replace('text="Score: 0"', 'text=""'),
            "feedback",
        ),
        (
            "flat render",
            lambda: FAITHFUL.replace(
                "fill=TILE_COLOURS.get(value, EMPTY)", "fill=EMPTY"
            ),
            "palette",
        ),
    ],
)
def test_each_missing_property_is_named(label: str, mutate, expect: str) -> None:
    report = grade_presentation(mutate(), TWENTY_FORTY_EIGHT.presentation)
    assert not report.passed, label
    assert any(item.check == expect for item in report.findings), (
        label,
        [str(item) for item in report.findings],
    )


def test_the_real_toolkit_is_never_reachable() -> None:
    """The allowance is for a substituted module, not for tkinter."""
    from core.discovery.reconstruction_sandbox import (
        ReconstructionASTViolation,
        audit_general_ast,
    )

    with pytest.raises(ReconstructionASTViolation):
        audit_general_ast("import tkinter as tk")
    with pytest.raises(ReconstructionASTViolation):
        audit_general_ast("import os", substituted_modules=frozenset({"tkinter"}))
    audit_general_ast("import tkinter as tk", substituted_modules=frozenset({"tkinter"}))


def test_a_module_that_reaches_for_the_filesystem_is_refused() -> None:
    report = grade_presentation(
        "import os\nimport tkinter as tk\n\n\ndef main():\n    return tk.Tk()\n",
        TWENTY_FORTY_EIGHT.presentation,
    )
    assert not report.passed
    assert any(item.check == "containment" for item in report.findings)


# ── End to end: nothing unpolished reaches the disk ───────────────────────

class _StubEngine:
    """Generation stubbed; every gate after it real."""

    def __init__(self, module_source: str) -> None:
        self.module_source = module_source

    async def reconstruct_executable_via_cognition(self, **kwargs) -> dict:
        return {
            "status": "supported",
            "code": self.module_source,
            "held_out_passed": kwargs and len(kwargs.get("held_out") or []),
            "equivalence": 1.0,
        }


def _materialize(source: str, destination: Path) -> dict:
    async def _run() -> dict:
        report = await materialize_program(
            _StubEngine(source), TWENTY_FORTY_EIGHT, destination
        )
        return report

    import core.self_improvement.program_materialization as pm

    original = pm._write_playable_module

    async def _stub(engine, spec, core_code, *, transfer=""):  # noqa: ANN001
        return source

    pm._write_playable_module = _stub
    try:
        return asyncio.run(_run())
    finally:
        pm._write_playable_module = original


def test_a_faithful_program_reaches_the_disk_and_runs(tmp_path: Path) -> None:
    report = _materialize(FAITHFUL, tmp_path)
    assert report["written"], report.get("reason")
    assert report["playable"]
    assert report["polished"]
    written = tmp_path / TWENTY_FORTY_EIGHT.default_filename
    assert written.exists()
    assert "tkinter" in written.read_text(encoding="utf-8")


def test_an_unpolished_program_is_never_written(tmp_path: Path) -> None:
    """It plays. It is still not 2048."""
    report = _materialize(_without(list(PALETTE.values())), tmp_path)
    assert report["playable"], "the rules and the game are fine — that is the point"
    assert not report["polished"]
    assert not report["written"]
    assert "not the software" in report["reason"]
    assert not list(tmp_path.iterdir())


# ── What the gate caught must reach the next build ────────────────────────

def test_a_rejection_is_remembered_for_the_next_attempt(tmp_path, monkeypatch) -> None:
    """The ledger existed and no build path ever wrote to it.

    A rejection is the only record of what she actually gets wrong, which
    makes it worth more to the next reconstruction than a success.
    """
    import core.self_improvement.reconstruction_memory as memory

    monkeypatch.setattr(memory, "_ledger_path", lambda root=None: tmp_path / "ledger.jsonl")

    report = _materialize(_without(list(PALETTE.values())), tmp_path / "out")
    assert not report["written"]

    remembered = memory.load_attempts()
    assert remembered, "the attempt left no trace"
    latest = remembered[-1]
    assert latest.target == "2048"
    assert not latest.succeeded
    assert any("palette" in correction for correction in latest.corrections)


def test_prior_experience_reaches_the_prompt(tmp_path, monkeypatch) -> None:
    import core.self_improvement.reconstruction_memory as memory

    monkeypatch.setattr(memory, "_ledger_path", lambda root=None: tmp_path / "ledger.jsonl")
    asyncio.run(
        memory.remember_attempt(
            memory.PriorAttempt(
                target="2048",
                summary="sliding tile board game",
                corrections=("palette: found 0 of 11 published colours",),
            )
        )
    )
    block = memory.recall_for("2048", summary="sliding tile board game").as_prompt_block()
    assert "WHAT YOU LEARNED" in block
    assert "published colours" in block
