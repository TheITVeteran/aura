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

That fix was real but incomplete, and the assertions written with it are the
reason it looked finished. They asked for ``top_share < 0.75`` and two distinct
winners — both of which are true of a perfect **duopoly**, which is what the
mechanism actually settled into: a and b splitting 24 ticks 12/12 in a strict
a-b-a-b alternation while c and d won nothing at all. A monopoly of one had
become a cartel of two and the regression test applauded.

The cause was structural rather than a tuning miss. Adaptation is a leaky
integrator, so a source winning fraction ``f`` of ticks is in equilibrium when
``g·f = r``; a fixed recovery rate therefore pins the sustainable share at
``r/g`` and lets exactly ``g/r`` sources rotate — two, for every possible field
size. ``GlobalWorkspace._fatigue_recovery`` derives ``r = g/n`` instead, making
the equilibrium share ``1/n``.

These regimes define correct behaviour, and they are deliberately in tension:
anything that widens the rotation can flatten real urgency, and anything that
sharpens urgency can starve the field. Both halves are asserted here, via
``core.verify.dynamics``, over a trajectory rather than a single tick.
"""

from __future__ import annotations

import asyncio
import collections

import pytest

from core.consciousness.global_workspace import CognitiveCandidate, GlobalWorkspace
from core.verify.dynamics import competition_health, no_limit_cycle

TICKS = 24


async def _run(sources: dict[str, float], ticks: int = TICKS):
    workspace = GlobalWorkspace()
    wins: collections.Counter[str] = collections.Counter()
    refused: collections.Counter[str] = collections.Counter()
    sequence: list[str] = []
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
            sequence.append(winner.source)
    return wins, refused, sequence


def test_a_two_point_gap_does_not_buy_a_monopoly():
    """The regression, asserted as a property instead of a symptom.

    Checking "the top source is under 75%" is what let the duopoly through.
    The real requirement is the opposing pair: nobody is locked out, *and*
    bidding higher still wins more often. Neither half is sufficient alone —
    ignoring the bids entirely satisfies the first, and a monopoly satisfies
    the second.
    """
    bids = {"memory": 0.90, "drive": 0.88, "perception": 0.86, "curiosity": 0.84}
    wins, _, _ = _sync(bids)
    total = sum(wins.values())
    shares = {name: wins.get(name, 0) / total for name in bids}

    findings = competition_health(
        shares,
        bids,
        subject="global_workspace",
        # Four near-equal sources over 24 ticks: an even rotation gives each
        # 0.25, so a floor of 0.10 fails a source that is being crowded out
        # while tolerating the integer remainder of 24/4.
        min_share=0.10,
        min_normalised_entropy=0.90,
        # One tick of 24 is 0.042; two sources can legitimately differ by that
        # much without arbitration being broken.
        order_tolerance=0.10,
    )
    assert not findings, "workspace competition is unhealthy:\n" + "\n".join(
        f"  {f}" for f in findings
    )


def test_the_field_does_not_collapse_to_a_cartel():
    """Every source that bids competitively must reach broadcast.

    This is the duopoly regression specifically: c and d measured 0.000 while
    the previous assertions passed.
    """
    bids = {"a": 0.90, "b": 0.88, "c": 0.86, "d": 0.84}
    wins, _, _ = _sync(bids)
    silent = sorted(name for name in bids if wins.get(name, 0) == 0)
    assert not silent, f"sources never reached broadcast in {TICKS} ticks: {silent}"


def test_arbitration_does_not_lock_into_a_short_cycle():
    """A fixed repeating order means arbitration stopped reading its inputs.

    Entropy alone cannot see this: a strict a-b-a-b alternation scores a
    perfect rotation between two sources.
    """
    _, _, sequence = _sync({"a": 0.90, "b": 0.88, "c": 0.86, "d": 0.84})
    findings = no_limit_cycle(
        sequence, max_period=2, min_repeats=6, subject="global_workspace"
    )
    assert not findings, "\n".join(str(f) for f in findings)


def test_losing_a_bid_does_not_silence_the_next_one():
    """Being outbid is a reason to bid again, not to be excluded."""
    _, refused, _seq = _sync({"a": 0.90, "b": 0.88, "c": 0.86})
    assert not refused, f"sources were refused submission after losing: {dict(refused)}"


def test_priority_still_dominates():
    """Adaptation must not flatten a real difference in urgency."""
    wins, _, _seq = _sync({"urgent": 0.99, "idle": 0.20})
    assert wins.get("urgent", 0) == TICKS, (
        f"an urgent source lost ticks to an idle one: {dict(wins)}"
    )


def test_a_clear_gap_is_respected():
    """0.30 apart is not a near-tie; the stronger source should hold."""
    wins, _, _seq = _sync({"strong": 0.90, "weak": 0.60})
    assert wins.get("strong", 0) == TICKS, dict(wins)


def test_a_lone_source_is_never_silenced():
    """With nothing else to attend to, the only bid must win every tick.

    The hard-refractory attempt failed exactly here: 12 of 24.
    """
    wins, _, _seq = _sync({"only": 0.9})
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
    "attribute", ["_WINNER_FATIGUE", "_MAX_FATIGUE"]
)
def test_the_adaptation_constants_exist(attribute: str) -> None:
    """Named constants, so the tuning is visible rather than buried."""
    assert isinstance(getattr(GlobalWorkspace, attribute), float)
