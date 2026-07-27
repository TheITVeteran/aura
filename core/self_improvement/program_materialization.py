"""From a reconstruction blueprint to a program that actually runs.

Asked to reverse-engineer a clean-room 2048 and put it on the Desktop, the
Program DNA lane produced — when it produced anything — a scaffold: a
``ReconstructedProgram`` class whose ``execute()`` returns
``{"status": "planned"}``. Blueprints, genome JSON, a verification plan, and
nothing anyone can play. The gap was never the analysis. It was that no stage
turned the analysis into working code and then checked that it worked.

This module is that stage, and it is built on the machinery that already does
the honest version of this for host binaries: ``reconstruct_executable_via_
cognition`` has the model write an implementation from behaviour alone, then a
sandbox differentially checks it against held-out observations the synthesizer
never saw, and labels the result ``supported`` / ``refuted`` / ``conjecture``.

**On the reference implementations below.** Each spec carries a hidden oracle
used only to *generate expected outputs* — never shown to the synthesizer, never
shipped. This is the same arrangement as the host-binary lane, where the oracle
is ``/usr/bin/base64``: the grader must know the right answer or it cannot
grade. Clean-room means she implements from the published rules without the
original source, which is exactly what happens here — the rules go in as prose,
an independent implementation of those rules decides whether what came out is
faithful, and a program that fails the battery is not written to disk at all.

Two things are verified, because "faithful" and "playable" are different claims:

* the rules, differentially, against held-out positions;
* the program, by importing what was written and playing it headlessly.
"""
from __future__ import annotations

import ast
import copy
import random
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from core.runtime.errors import record_degradation

_RECOVERABLE = (RuntimeError, AttributeError, TypeError, ValueError, OSError, ImportError, KeyError)

Board = list[list[int]]


# ── The hidden graders ─────────────────────────────────────────────────────

def _slide_row_left(row: list[int]) -> tuple[list[int], int]:
    """One row of 2048, moved left. Public rules, implemented independently.

    Compress, merge each pair once left-to-right, compress again. A tile
    created by a merge cannot merge again on the same move — the rule most
    reimplementations get wrong, and therefore the one worth grading on.
    """
    tiles = [value for value in row if value]
    merged: list[int] = []
    gained = 0
    index = 0
    while index < len(tiles):
        if index + 1 < len(tiles) and tiles[index] == tiles[index + 1]:
            value = tiles[index] * 2
            merged.append(value)
            gained += value
            index += 2
        else:
            merged.append(tiles[index])
            index += 1
    merged.extend([0] * (len(row) - len(merged)))
    return merged, gained


def _reference_2048_move(case: dict[str, Any]) -> dict[str, Any]:
    """``{"board", "direction"} -> {"board", "score"}``, deterministic.

    No tile spawns here: spawning is random, and a random step cannot be
    differentially graded. The move itself is fully determined, so that is what
    the battery holds her to.
    """
    board: Board = copy.deepcopy(list(case["board"]))
    direction = str(case["direction"]).strip().lower()
    size = len(board)

    if direction == "left":
        rows = board
    elif direction == "right":
        rows = [list(reversed(row)) for row in board]
    elif direction == "up":
        rows = [[board[r][c] for r in range(size)] for c in range(size)]
    elif direction == "down":
        rows = [[board[r][c] for r in range(size - 1, -1, -1)] for c in range(size)]
    else:
        raise ValueError(f"unknown direction: {direction}")

    total = 0
    moved_rows: list[list[int]] = []
    for row in rows:
        new_row, gained = _slide_row_left(row)
        total += gained
        moved_rows.append(new_row)

    if direction == "left":
        result = moved_rows
    elif direction == "right":
        result = [list(reversed(row)) for row in moved_rows]
    elif direction == "up":
        result = [[moved_rows[c][r] for c in range(size)] for r in range(size)]
    else:
        result = [[moved_rows[c][size - 1 - r] for c in range(size)] for r in range(size)]

    return {"board": result, "score": total}


# ── What a program has to be, to count as reconstructed ────────────────────

@dataclass(frozen=True)
class ProgramSpec:
    """A behaviour published in prose, plus a way to tell if she matched it."""

    name: str
    aliases: tuple[str, ...]
    fn_name: str
    spec_docs: tuple[str, ...]
    oracle: Callable[[dict[str, Any]], Any]
    case_generator: Callable[[random.Random], dict[str, Any]]
    train_case_count: int = 6
    held_out_case_count: int = 14
    playable_module_hint: str = ""
    play_check: Callable[[Any], tuple[bool, str]] | None = None
    default_filename: str = "program.py"
    authorization: str = "public_observation"
    objective: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def matches(self, target: str) -> bool:
        candidate = " ".join(str(target or "").split()).strip().lower()
        if not candidate:
            return False
        return candidate == self.name.lower() or candidate in self.aliases


def _random_2048_case(rng: random.Random) -> dict[str, Any]:
    """A position that separates a faithful implementation from a plausible one.

    Random boards mostly test sliding, which everyone gets right. The rule that
    actually distinguishes implementations is that a merged tile cannot merge
    again on the same move, and it only shows on a run of three or four equal
    tiles *along the direction of travel*: [2,2,2,0] left is [4,2,0,0], not
    [8,0,0,0]. So the run is planted along the axis being moved. Grading the
    classic wrong merge on unplanted boards caught it 2 times in 14; planted, it
    fails almost every case it should.
    """
    values = [0, 0, 2, 2, 4, 4, 8, 16, 32, 64, 128]
    board = [[rng.choice(values) for _ in range(4)] for _ in range(4)]
    direction = rng.choice(["left", "right", "up", "down"])

    run_length = rng.choice([2, 3, 3, 4])
    value = rng.choice([2, 4, 8, 16])
    start = rng.randrange(4 - run_length + 1)
    line = rng.randrange(4)
    for offset in range(run_length):
        if direction in {"left", "right"}:
            board[line][start + offset] = value
        else:
            board[start + offset][line] = value
    return {"board": board, "direction": direction}


def _play_2048_headlessly(module: Any) -> tuple[bool, str]:
    """Import what she wrote and actually play it.

    A verified ``move`` proves the rules. It does not prove there is a game: a
    module can hold a perfect rule engine and no board, no spawn, and no way to
    lose. So this plays one — spawning, moving, and reaching a terminal state.
    """
    move = getattr(module, "move", None)
    if not callable(move):
        return False, "no callable move(state) in the written program"

    board: Board = [[0] * 4 for _ in range(4)]
    spawn = getattr(module, "spawn_tile", None) or getattr(module, "add_random_tile", None)
    if not callable(spawn):
        return False, "no tile spawn function — a 2048 board that never fills is not a game"

    rng = random.Random(20482048)
    for _ in range(2):
        board = _coerce_board(spawn(board)) or board

    game_over = getattr(module, "is_game_over", None) or getattr(module, "game_over", None)
    if not callable(game_over):
        return False, "no game-over test — a game you cannot lose is not the game"

    moves_played = 0
    total_score = 0
    for _ in range(400):
        if game_over(board):
            break
        direction = rng.choice(["left", "right", "up", "down"])
        outcome = move({"board": board, "direction": direction})
        new_board = _coerce_board(
            outcome.get("board") if isinstance(outcome, dict) else outcome
        )
        if new_board is None:
            return False, f"move() returned something that is not a board: {type(outcome)!r}"
        if isinstance(outcome, dict):
            try:
                total_score += int(outcome.get("score") or 0)
            except (TypeError, ValueError):
                pass
        if new_board != board:
            moves_played += 1
            board = _coerce_board(spawn(new_board)) or new_board

    if moves_played < 10:
        return False, f"only {moves_played} legal moves were possible — the board is not live"
    if max(max(row) for row in board) < 8:
        return False, "no tile ever merged past 4 across 400 moves"
    return True, (
        f"played {moves_played} moves headlessly, score {total_score}, "
        f"highest tile {max(max(row) for row in board)}"
    )


def _coerce_board(value: Any) -> Board | None:
    try:
        board = [[int(cell) for cell in row] for row in value]
    except (TypeError, ValueError):
        return None
    if not board or any(len(row) != len(board) for row in board):
        return None
    return board


TWENTY_FORTY_EIGHT = ProgramSpec(
    name="2048",
    aliases=("2048", "the game 2048", "2048 game", "game 2048"),
    fn_name="move",
    spec_docs=(
        "2048 is played on a 4x4 grid of tiles. Every tile holds a power of two; "
        "an empty cell is 0.",
        "A move is one of left, right, up or down. Every tile slides as far as it "
        "can in that direction.",
        "Two tiles of equal value that collide merge into one tile of twice the "
        "value, and the player scores that new value.",
        "A tile produced by a merge cannot merge again during the same move. "
        "Merging resolves in the direction of travel: moving left, [2,2,2,0] "
        "becomes [4,2,0,0], not [2,4,0,0] and not [8,0,0,0].",
        "After a move that changed the board, one new tile appears in a random "
        "empty cell: a 2 with probability 0.9, a 4 with probability 0.1.",
        "The game ends when no cell is empty and no adjacent pair is equal, so no "
        "move can change anything. Reaching a 2048 tile is the win condition.",
        "Implement move(state) taking {'board': 4x4 list of lists of int, "
        "'direction': one of 'left','right','up','down'} and returning "
        "{'board': the new grid, 'score': points gained by this move}. move() "
        "performs no spawning and no randomness.",
        "Also provide spawn_tile(board) returning a new board with one random "
        "tile added, and is_game_over(board) returning True when no move can "
        "change the board.",
    ),
    oracle=_reference_2048_move,
    case_generator=_random_2048_case,
    play_check=_play_2048_headlessly,
    default_filename="game_2048.py",
    objective="clean-room reconstruction of the game 2048 from its published rules",
    playable_module_hint=(
        "a playable command-line game using your verified move(), spawn_tile() "
        "and is_game_over(): print the grid, read w/a/s/d from input(), keep "
        "score, and stop on win or game over"
    ),
)

KNOWN_PROGRAM_SPECS: tuple[ProgramSpec, ...] = (TWENTY_FORTY_EIGHT,)


def resolve_program_spec(target: str) -> ProgramSpec | None:
    for spec in KNOWN_PROGRAM_SPECS:
        if spec.matches(target):
            return spec
    return None


def build_case_sets(spec: ProgramSpec, *, seed: int = 2048) -> tuple[list[dict], list[dict]]:
    """Train and held-out cases, from the hidden oracle, with no overlap.

    Held-out cases are drawn after the training ones from the same stream, so
    the synthesizer is graded on positions it was never shown.
    """
    rng = random.Random(seed)
    seen: set[str] = set()
    train: list[dict[str, Any]] = []
    held_out: list[dict[str, Any]] = []
    for bucket, wanted in ((train, spec.train_case_count), (held_out, spec.held_out_case_count)):
        guard = 0
        while len(bucket) < wanted and guard < wanted * 40:
            guard += 1
            case = spec.case_generator(rng)
            key = repr(case)
            if key in seen:
                continue
            seen.add(key)
            try:
                expected = spec.oracle(case)
            except _RECOVERABLE:
                continue
            bucket.append({"input": case, "expected": expected})
    return train, held_out


# ── Reconstruct, verify, and only then write ───────────────────────────────

_PLAYABLE_WRAPPER_SYSTEM_PROMPT = (
    "You are finishing a clean-room reimplementation you already wrote and "
    "verified. Standard library only. Output one complete Python module and "
    "nothing else."
)


# A lane that cannot be admitted is a lane that is not available.
#
# The un-steered code model exists because the steered persona cortex corrupts
# symbolic tokens, so it is the right first choice. But it is a second set of
# weights, and on a 64GB host with the resident 32B already holding ~25GB it is
# correctly refused: "in_process_model_admission_refused:lane_budget_exceeded:
# cortex request 21.5GB + committed 25.3GB > budget 46.1GB". Measured live
# 2026-07-27, and it took the whole reconstruction down with it.
#
# Absent weights already fall back to the resident cortex. An admission refusal
# is the same situation — the preferred lane cannot serve — and it should reach
# the same fallback rather than failing the work. Refusing to build anything
# because the nicer tool is busy is not a safety property.
async def _generate_code_with_fallback(
    prompt: str,
    *,
    context: dict[str, Any],
) -> str:
    """Generate through the code lane, falling back to the resident cortex."""
    from core.brain.llm.code_generator import LLMCodeGenerator, extract_python_code

    code_router = None
    try:
        from core.brain.llm.local_code_model import get_local_code_model

        code_router = get_local_code_model()
    except (ImportError, RuntimeError, OSError):
        code_router = None

    if code_router is not None:
        try:
            raw = await LLMCodeGenerator(router=code_router).generate_async(
                prompt, context=dict(context)
            )
            extracted = extract_python_code(raw) or str(raw or "")
            if extracted.strip():
                return extracted
        except _RECOVERABLE as exc:
            record_degradation(
                "program_materialization",
                exc,
                severity="info",
                action="fell back to the resident cortex after the code lane was unavailable",
            )

    raw = await LLMCodeGenerator().generate_async(prompt, context=dict(context))
    return extract_python_code(raw) or str(raw or "")


async def materialize_program(
    engine: Any,
    spec: ProgramSpec,
    destination: Path,
    *,
    seed: int = 2048,
    max_repair_attempts: int = 2,
) -> dict[str, Any]:
    """Have her reconstruct the program, prove it, then put it on disk.

    Nothing is written unless the rules survived held-out verification and the
    written module actually played. A file on the Desktop is a claim that the
    thing works; it should not be possible to make that claim by accident.
    """
    train, held_out = build_case_sets(spec, seed=seed)
    report: dict[str, Any] = {
        "target": spec.name,
        "destination": str(destination),
        "written": False,
        "held_out_total": len(held_out),
        "held_out_passed": 0,
        "status": "conjecture",
        "playable": False,
        "play_evidence": "",
        "reason": "",
    }

    outcome = await engine.reconstruct_executable_via_cognition(
        target=spec.name,
        spec_docs=list(spec.spec_docs),
        train_examples=train,
        held_out=held_out,
        fn_name=spec.fn_name,
        authorization=spec.authorization,
        objective=spec.objective or f"clean-room reconstruction of {spec.name}",
        sandbox_profile="general",
        max_repair_attempts=max_repair_attempts,
        max_tokens=2200,
    )
    report["status"] = str(outcome.get("status") or "conjecture")
    report["held_out_passed"] = int(outcome.get("held_out_passed") or 0)
    report["equivalence"] = outcome.get("equivalence", 0.0)
    report["failures"] = outcome.get("failures", [])
    core_code = str(outcome.get("code") or "")

    if report["status"] != "supported" or not core_code.strip():
        report["reason"] = str(outcome.get("reason") or "") or (
            f"held-out verification did not pass: "
            f"{report['held_out_passed']}/{report['held_out_total']}"
        )
        return report

    module_source = await _write_playable_module(engine, spec, core_code)
    if not module_source.strip():
        report["reason"] = "the verified rules were produced but no playable module was"
        return report

    playable, evidence = _verify_playable(spec, module_source)
    report["playable"] = playable
    report["play_evidence"] = evidence
    if not playable:
        report["reason"] = f"the written program did not play: {evidence}"
        return report

    try:
        from core.runtime.file_write_gateway import get_file_write_gateway

        target_path = destination if destination.suffix else destination / spec.default_filename
        await get_file_write_gateway().ensure_directory_async(
            target_path.parent, source="program_materialization"
        )
        await get_file_write_gateway().write_text_async(
            target_path, module_source, source="program_materialization"
        )
    except _RECOVERABLE as exc:
        record_degradation(
            "program_materialization", exc, action="verified program could not be written to disk"
        )
        report["reason"] = f"verified but could not be written: {type(exc).__name__}: {exc}"
        return report

    report["written"] = True
    report["destination"] = str(target_path)
    return report


async def _write_playable_module(engine: Any, spec: ProgramSpec, core_code: str) -> str:
    """Her verified rules, plus the game around them, written by her."""
    prompt = (
        f"You have already written and verified this {spec.name} rule engine "
        f"against held-out positions:\n\n```python\n{core_code}\n```\n\n"
        f"Now produce ONE complete, runnable Python module that keeps these "
        f"functions exactly as they are — do not change their behaviour — and "
        f"adds {spec.playable_module_hint}.\n\n"
        "Requirements:\n"
        f"- keep `{spec.fn_name}` with the same signature and semantics\n"
        "- provide spawn_tile(board) and is_game_over(board)\n"
        "- put the interactive loop under `if __name__ == \"__main__\":` so the "
        "module can be imported without playing\n"
        "- standard library only, no input at import time\n"
        "Return only the module source."
    )
    try:
        return await _generate_code_with_fallback(
            prompt,
            context={
                "prefer_tier": "primary",
                "temperature": 0.1,
                "max_tokens": 2600,
                "origin": "program_materialization",
                "system_prompt": _PLAYABLE_WRAPPER_SYSTEM_PROMPT,
            },
        )
    except _RECOVERABLE as exc:
        record_degradation(
            "program_materialization", exc, severity="warning", action="no playable module produced"
        )
        return ""


def _verify_playable(spec: ProgramSpec, module_source: str) -> tuple[bool, str]:
    """Import the module and play it. Never trust source that was never run."""
    if spec.play_check is None:
        return True, "no playability check declared for this target"
    try:
        ast.parse(module_source)
    except SyntaxError as exc:
        return False, f"the written module does not parse: {exc}"
    namespace: dict[str, Any] = {"__name__": "reconstructed_program"}
    try:
        exec(compile(module_source, "<reconstructed>", "exec"), namespace)  # noqa: S102
    except _RECOVERABLE as exc:
        return False, f"importing the written module raised {type(exc).__name__}: {exc}"
    module = type("ReconstructedModule", (), namespace)
    try:
        return spec.play_check(module)
    except _RECOVERABLE as exc:
        return False, f"playing the written module raised {type(exc).__name__}: {exc}"


__all__ = [
    "KNOWN_PROGRAM_SPECS",
    "TWENTY_FORTY_EIGHT",
    "ProgramSpec",
    "build_case_sets",
    "materialize_program",
    "resolve_program_spec",
]


# ── Checkers: the case that failed, and the bar that matters ───────────────
#
# Bryan asked for checkers and got a board whose pieces did not move. It is the
# right bar precisely because it is not hard — if this lane cannot produce
# checkers, nothing more complex is worth attempting — and it is unforgiving in
# a useful way: mandatory capture, chained jumps, and promotion ending the move
# are three things a plausible-looking implementation routinely gets wrong.

def _random_checkers_case(rng: random.Random) -> dict[str, Any]:
    """A position reached by real play, so the cases are positions that occur.

    Random piece scatters produce boards no game ever visits and grade an
    implementation on situations it will never meet. Playing a short random
    game first gives openings, midgames, forced captures and kings.
    """
    from core.self_improvement import checkers_reference as ref

    board = ref.initial_board()
    side = ref.BLACK
    for _ in range(rng.randrange(0, 40)):
        moves = ref.legal_moves(board, side)
        if not moves:
            break
        board = ref.apply_move(board, rng.choice(moves))
        side = ref.WHITE if side == ref.BLACK else ref.BLACK
    return {"board": board, "side": side}


def _reference_checkers_moves(case: dict[str, Any]) -> list[list[list[int]]]:
    """Every legal move for the side to move, as JSON-shaped square paths."""
    from core.self_improvement import checkers_reference as ref

    moves = ref.legal_moves(list(case["board"]), str(case["side"]))
    return [[[int(r), int(c)] for r, c in path] for path in moves]


def _play_checkers_headlessly(module: Any) -> tuple[bool, str]:
    """The full quality gate, not merely "it ran"."""
    from core.self_improvement import checkers_reference as ref
    from core.self_improvement.artifact_quality import (
        QualityReport,
        check_it_can_end,
        check_it_is_interactive,
        check_it_refuses_the_illegal,
        check_there_is_something_to_look_at,
    )

    namespace = {
        name: getattr(module, name)
        for name in dir(module)
        if not name.startswith("__")
    }
    report = QualityReport()

    def _initial(ns: dict[str, Any]) -> Any:
        factory = ns.get("initial_board") or ns.get("new_board") or ns.get("starting_board")
        if not callable(factory):
            raise ValueError("no initial_board() to start from")
        return {"board": factory(), "side": ref.BLACK}

    def _legal(ns: dict[str, Any], state: Any) -> list[Any]:
        fn = ns.get("legal_moves") or ns.get("get_legal_moves") or ns.get("valid_moves")
        if not callable(fn):
            raise ValueError("no legal_moves(board, side)")
        return list(fn(state["board"], state["side"]) or [])

    def _apply(ns: dict[str, Any], state: Any, action: Any) -> Any:
        fn = ns.get("apply_move") or ns.get("make_move") or ns.get("move")
        if not callable(fn):
            raise ValueError("no apply_move(board, path)")
        board = fn(state["board"], action)
        other = ref.WHITE if state["side"] == ref.BLACK else ref.BLACK
        return {"board": board, "side": other}

    def _describe(state: Any) -> str:
        return f"{state['side']}:" + "".join(
            str(cell) for row in state["board"] for cell in row
        )

    def _illegal(ns: dict[str, Any], state: Any) -> list[Any]:
        # Moving off the board, moving nothing, and moving the other side's
        # piece: three refusals any real rule engine makes.
        return [
            [[0, 0], [9, 9]],
            [[4, 4], [5, 5]],
            [[0, 1], [0, 1]],
        ]

    def _to_completion(ns: dict[str, Any]) -> tuple[bool, str]:
        state = _initial(ns)
        for ply in range(400):
            moves = _legal(ns, state)
            if not moves:
                return True, f"reached a terminal position after {ply} plies"
            state = _apply(ns, state, moves[0])
        return False, "no terminal position within 400 plies"

    for findings, evidence in (
        check_it_is_interactive(
            namespace,
            initial_state=_initial,
            legal_actions=_legal,
            apply_action=_apply,
            describe_state=_describe,
            min_effective_actions=12,
        ),
        check_it_refuses_the_illegal(
            namespace,
            initial_state=_initial,
            illegal_actions=_illegal,
            apply_action=_apply,
            describe_state=_describe,
        ),
        check_it_can_end(namespace, play_to_completion=_to_completion),
        check_there_is_something_to_look_at(
            namespace, initial_state=lambda ns: _initial(ns)["board"]
        ),
    ):
        report.findings.extend(findings)
        report.evidence.extend(evidence)

    return report.passed, report.summary


CHECKERS = ProgramSpec(
    name="checkers",
    aliases=("checkers", "draughts", "english draughts", "the game checkers", "checkers game"),
    fn_name="legal_moves_for",
    spec_docs=(
        "Checkers (English draughts) is played on an 8x8 board using only the "
        "dark squares. Each side starts with twelve men on the three rows "
        "nearest them.",
        "Represent the board as 8 rows top to bottom, each of 8 cells: 0 for an "
        "empty square, 'b' and 'w' for black and white men, 'B' and 'W' for "
        "kings. Black starts on the top rows and moves down the board; white "
        "starts on the bottom rows and moves up.",
        "A man moves one square diagonally forward to an empty square. A king "
        "moves one square diagonally in any direction.",
        "A capture jumps an adjacent enemy piece to the empty square directly "
        "beyond it, and removes the jumped piece.",
        "Capturing is MANDATORY: if any capture is available to the side to "
        "move, every non-capturing move is illegal.",
        "Captures chain: if the piece that just jumped can jump again, it must, "
        "and the whole chain is one move.",
        "A man reaching the far rank is promoted to king, and promotion ENDS "
        "the move — a man crowned mid-chain does not keep jumping as a king.",
        "The side to move loses when it has no pieces or no legal move.",
        "Implement legal_moves_for(state) taking {'board': the 8x8 grid, "
        "'side': 'b' or 'w'} and returning every legal move as a list of square "
        "paths, where a path is a list of [row, col] pairs starting with the "
        "piece's square. Sort the returned list.",
        "Also provide initial_board(), apply_move(board, path) returning the new "
        "board, legal_moves(board, side), winner(board, side_to_move), and "
        "render(board) returning a readable multi-line string.",
    ),
    oracle=_reference_checkers_moves,
    case_generator=_random_checkers_case,
    train_case_count=5,
    held_out_case_count=12,
    play_check=_play_checkers_headlessly,
    default_filename="checkers.py",
    objective="clean-room reconstruction of English draughts from its published rules",
    playable_module_hint=(
        "a playable command-line game on top of your verified rules: render the "
        "board with row and column labels, list the legal moves, read the "
        "player's choice from input(), play a simple opponent reply, and stop "
        "when someone wins"
    ),
)

KNOWN_PROGRAM_SPECS = (TWENTY_FORTY_EIGHT, CHECKERS)
