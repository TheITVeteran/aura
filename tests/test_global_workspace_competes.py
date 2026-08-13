"""The global workspace must actually compete.

The class docstring says the refractory mechanism "prevents the same subsystem
from dominating every cycle and forces genuine competition". It did the
opposite. Every LOSER was inhibited for a tick, and inhibition is checked in
``submit()``, so the sources that had just lost could not bid on the next tick.
The winner then ran unopposed, won again, and inhibited them again.

Measured before the fix: four sources bidding every tick at 0.90 / 0.88 / 0.86
/ 0.84 gave the top source **24 wins out of 24** while the other three had half
their submissions refused. A two-point priority difference bought a permanent
monopoly of the broadcast — which means that in steady state there was no
competition and no global workspace, only "highest-priority source always
wins".

Hard-inhibiting the winner instead was tried and is worse in a subtler way: an
urgent source at 0.99 and an idle one at 0.20 alternated 50/50, because a hard
block ignores how much stronger the bid was, and a source bidding alone won
only half its ticks. The mechanism that works is adaptation — fatigue the
recent winner's effective priority and let it recover.

These four regimes define correct behaviour. Any future tuning has to keep all
of them.
"""

from __future__ import annotations

import asyncio
import collections

import pytest

from core.consciousness.global_workspace import CognitiveCandidate, GlobalWorkspace

TICKS = 24


async def _run(sources: dict[str, float], ticks: int = TICKS):
    workspace = GlobalWorkspace()
    wins: collections.Counter[str] = collections.Counter()
    refused: collections.Counter[str] = collections.Counter()
    for tick in range(ticks):
        for name, priority in sources.items():
            admitted = await workspace.submit(
                CognitiveCandidate(
                    content=f"{name}@{tick}", source=name, priority=priority
                )
            )
            if not admitted:
                refused[name] += 1
        winner = await workspace.run_competition()
        if winner is not None:
            wins[winner.source] += 1
    return wins, refused


def test_a_two_point_gap_does_not_buy_a_monopoly():
    """The regression. This was 24/24 for one source."""
    wins, _ = _sync({"memory": 0.90, "drive": 0.88, "perception": 0.86, "curiosity": 0.84})
    top_share = max(wins.values()) / sum(wins.values())
    assert top_share < 0.75, f"one source still dominates the workspace: {dict(wins)}"
    assert len(wins) >= 2, f"only one source ever reached broadcast: {dict(wins)}"


def test_losing_a_bid_does_not_silence_the_next_one():
    """Being outbid is a reason to bid again, not to be excluded."""
    _, refused = _sync({"a": 0.90, "b": 0.88, "c": 0.86})
    assert not refused, f"sources were refused submission after losing: {dict(refused)}"


def test_priority_still_dominates():
    """Adaptation must not flatten a real difference in urgency."""
    wins, _ = _sync({"urgent": 0.99, "idle": 0.20})
    assert wins.get("urgent", 0) == TICKS, (
        f"an urgent source lost ticks to an idle one: {dict(wins)}"
    )


def test_a_clear_gap_is_respected():
    """0.30 apart is not a near-tie; the stronger source should hold."""
    wins, _ = _sync({"strong": 0.90, "weak": 0.60})
    assert wins.get("strong", 0) == TICKS, dict(wins)


def test_a_lone_source_is_never_silenced():
    """With nothing else to attend to, the only bid must win every tick.

    The hard-refractory attempt failed exactly here: 12 of 24.
    """
    wins, _ = _sync({"only": 0.9})
    assert wins.get("only", 0) == TICKS, dict(wins)


def test_ignition_requires_crossing_the_threshold():
    """Sub-threshold content wins the slot but must not ignite."""

    async def scenario():
        workspace = GlobalWorkspace()
        for name, priority in (("a", 0.2), ("b", 0.3)):
            await workspace.submit(
                CognitiveCandidate(content="x", source=name, priority=priority)
            )
        await workspace.run_competition()
        return workspace.is_ignited(), workspace.get_ignition_level()

    ignited, level = asyncio.run(scenario())
    assert ignited is False
    assert level < GlobalWorkspace._IGNITION_THRESHOLD


def test_ignition_fires_above_the_threshold():
    async def scenario():
        workspace = GlobalWorkspace()
        await workspace.submit(
            CognitiveCandidate(content="x", source="urgent", priority=0.95)
        )
        await workspace.run_competition()
        return workspace.is_ignited()

    assert asyncio.run(scenario()) is True


def _sync(sources: dict[str, float]):
    return asyncio.run(_run(sources))


@pytest.mark.parametrize(
    "attribute", ["_WINNER_FATIGUE", "_MAX_FATIGUE", "_FATIGUE_RECOVERY"]
)
def test_the_adaptation_constants_exist(attribute: str) -> None:
    """Named constants, so the tuning is visible rather than buried."""
    assert isinstance(getattr(GlobalWorkspace, attribute), float)
