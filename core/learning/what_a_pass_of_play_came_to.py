"""What a pass of play over many games came to, as evidence her self-improvement can weigh.

Her governed self-improvement (core/learning/recursive_self_improvement.py) promotes a change only where an evaluation
after it is better than before, and plans its next change from the signals of what is going wrong. A pass of play is
both, for every place she plays: she is put in each of a set of games she has never been taught, the trace of each is
kept, and how far she got in each is read off what she did and said there. So a pass is turned here into the two things
that loop takes:

- a scorecard: how far she got, game by game, on a scale anyone watching would use (0 never reached play, 1 played
  without gaining, 2 gained something, 3 did what was asked of the thing, 4 won), the mean over the games she was
  allowed to learn from, and, kept apart, the mean over games held out from all learning, which alone shows whether
  what she learned is general; with the games that went worse than in the pass before, as regressions;
- signals: for each game short of what was asked, what kind of shortfall it was, as a reason to improve.

A change that raises the games she learned from and not the held-out ones has learned those games, not how to play,
and is not a gain: the held-out mean is reported on its own so that no evaluator can average it away.

Nothing here knows a game. The trace of a game is its log; what she was asked is read the way a watcher would read it.
"""
from __future__ import annotations

import ast
import json
import re
from collections.abc import Collection
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

__all__ = ["HowFar", "levels_in", "scorecard", "signals"]

#: How far she got in a game.
NEVER_PLAYED, PLAYED, GAINED, AS_ASKED, WON = 0, 1, 2, 3, 4
SAID = {NEVER_PLAYED: "never reached play", PLAYED: "played without gaining", GAINED: "gained but did not do what was asked",
        AS_ASKED: "did what was asked", WON: "won"}

#: What she says when she has done what was asked of a thing that is not won: told what she made, what it answered,
#: or her best of the runs asked for.
_AS_ASKED = re.compile(r"it's for making things\. What I made shows|it answers what you|best (?:score|run|was run)|"
                       r"That.s \d runs|Run \d of \d: scored [1-9]")
_STRETCH = re.compile(r"a stretch played as it happened: (\{.*?\})(?:\n|$)")
_SAID = re.compile(r"saying out loud: (.*)")


@dataclass(frozen=True)
class HowFar:
    """How far she got in one game, and what showed it."""

    game: int
    level: int
    gains: int = 0
    stretches: int = 0
    runs: tuple[str, ...] = ()
    ended: tuple[str, ...] = field(default_factory=tuple)


def _results(pass_dir: Path) -> dict[int, dict[str, Any]]:
    results: dict[int, dict[str, Any]] = {}
    for log in pass_dir.glob("**/run.log"):
        for line in log.read_text(errors="replace").splitlines():
            if line.startswith('{"game"'):
                try:
                    result = json.loads(line)
                except ValueError:
                    continue
                results[int(result["game"])] = result
    return results


def levels_in(pass_dir: str | Path) -> dict[int, HowFar]:
    """How far she got in every game of a pass, from each game's result and its log."""
    pass_dir = Path(pass_dir)
    out: dict[int, HowFar] = {}
    for game, result in _results(pass_dir).items():
        text = "".join(log.read_text(errors="replace") for log in pass_dir.glob(f"**/game{game}.log"))
        gains, played, ended = 0, False, []
        stretches = _STRETCH.findall(text)
        for stretch in stretches:
            try:
                kept = ast.literal_eval(stretch)
            except (ValueError, SyntaxError):
                continue
            played = played or bool(kept.get("pictures_a_second"))
            gains += int(kept.get("gains") or 0)
            ended.append(str(kept.get("ended") or ""))
        said = " ".join([*(result.get("said") or []), *_SAID.findall(text)])
        runs = tuple(str(run) for run in result.get("runs") or ())
        if result.get("won"):
            level = WON
        elif _AS_ASKED.search(said):
            level = AS_ASKED
        elif gains > 0:
            level = GAINED
        elif played or re.search(r"shots played|a board played", text) or text.count("about to press") > 8:
            level = PLAYED
        else:
            level = NEVER_PLAYED
        out[game] = HowFar(game, level, gains, len(stretches), runs, tuple(ended[-5:]))
    return out


def scorecard(pass_dir: str | Path, held_out: Collection[int], before: str | Path | None = None) -> dict[str, Any]:
    """The pass as a scorecard her self-improvement can take (``ImprovementScorecard`` fields): its score is the mean
    level over the games she may learn from, as a share of winning; the held-out mean is a metric of its own; and every
    game that went worse than in the pass ``before`` is a regression."""
    levels = levels_in(pass_dir)
    learned = [how.level for game, how in levels.items() if game not in held_out]
    kept_apart = [how.level for game, how in levels.items() if game in held_out]
    earlier = levels_in(before) if before is not None else {}
    regressions = [f"game {game}: {SAID[earlier[game].level]} before, {SAID[how.level]} now"
                   for game, how in sorted(levels.items()) if game in earlier and how.level < earlier[game].level]
    mean = sum(learned) / max(1, len(learned))
    return {
        "score": mean / WON,
        "metrics": {
            "learned_from_mean": mean,
            "held_out_mean": sum(kept_apart) / max(1, len(kept_apart)),
            "games": float(len(levels)),
            "won": float(sum(how.level == WON for how in levels.values())),
            "as_asked_or_better": float(sum(how.level >= AS_ASKED for how in levels.values())),
        },
        "regressions": regressions,
        "evidence": {"pass": str(pass_dir), "levels": {str(game): how.level for game, how in sorted(levels.items())}},
    }


def signals(pass_dir: str | Path, held_out: Collection[int]) -> list[dict[str, Any]]:
    """For each game she may learn from that fell short of what was asked, a reason to improve: what kind of shortfall
    it was, how far short, and how its stretches of play ended. Held-out games give no signals: what is learned from
    them would no longer show whether learning is general."""
    out = []
    for game, how in sorted(levels_in(pass_dir).items()):
        if game in held_out or how.level >= AS_ASKED:
            continue
        out.append({"source": "a pass of play", "kind": SAID[how.level], "severity": (AS_ASKED - how.level) / AS_ASKED,
                    "metric": "how far she got", "evidence": {"game": game, "runs": list(how.runs),
                                                             "stretches_ended": list(how.ended)}})
    return out


def evaluator_for(campaign: str | Path, held_out: Collection[int]) -> Any:
    """An evaluator for her self-improvement loop (``RecursiveSelfImprovementLoop(evaluator=...)``): the scorecard of
    the latest pass of play in ``campaign``, against the pass before it."""
    campaign = Path(campaign)

    def evaluate() -> dict[str, Any]:
        passes = sorted((p for p in campaign.iterdir() if p.is_dir() and any(p.glob("**/run.log"))),
                        key=lambda p: p.stat().st_mtime)
        if not passes:
            return {"score": 0.0, "regressions": ["no pass of play to evaluate"]}
        return scorecard(passes[-1], held_out, before=passes[-2] if len(passes) > 1 else None)

    return evaluate


def recorded(loop: Any, pass_dir: str | Path, held_out: Collection[int]) -> int:
    """Each shortfall of a pass recorded with her self-improvement loop as a reason to improve; how many."""
    found = signals(pass_dir, held_out)
    for signal in found:
        loop.record_signal(signal["source"], signal["kind"], severity=signal["severity"], metric=signal["metric"],
                           evidence=signal["evidence"])
    return len(found)
