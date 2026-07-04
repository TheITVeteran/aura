"""The ablation runner must drive real organs and report honest deltas.

Guards the #1 external-review deliverable (Criticisms.pdf): reviewer-runnable
subsystem deltas with no clamps and no hidden nulls.
"""
from __future__ import annotations

import asyncio

import pytest

from tools.ablation_runner import (
    OFFLINE_CONDITIONS,
    ablate_substrate,
    ablate_system2,
    ablate_verifier,
    run,
)


def test_substrate_is_load_bearing():
    r = ablate_substrate()
    assert r.baseline > r.ablated
    assert r.ablated == 0.0, "a coupler blinded to state must produce one identical policy"
    assert r.detail["distinct_intact_policies"] > r.detail["distinct_blinded_policies"] == 1
    assert r.load_bearing is True


def test_system2_is_load_bearing():
    r = ablate_system2()
    assert r.detail["battery_size"] >= 4, "battery must actually contain solvable prompts"
    assert r.baseline == 1.0 and r.ablated == 0.0
    assert r.load_bearing is True


def test_verifier_rejects_wrong_answers_a_null_verifier_would_accept():
    r = ablate_verifier()
    # Real verifier rejects every wrong distractor; the null verifier rejects none.
    assert r.detail["wrong_rejected_by_real"] == r.detail["battery_size"]
    assert r.detail["wrong_rejected_by_null"] == 0
    assert r.baseline > r.ablated
    assert r.load_bearing is True


def test_runner_reports_every_offline_condition():
    results = asyncio.run(run(list(OFFLINE_CONDITIONS)))
    names = {r.name for r in results}
    assert names == {"without_substrate", "without_system2", "without_verifier"}
    # Honesty: the verdict is a real boolean derived from measured values,
    # never asserted true unconditionally.
    for r in results:
        assert isinstance(r.load_bearing, bool)
        assert r.delta == pytest.approx(r.baseline - r.ablated)


def test_no_clamps_or_hardcoded_statistics_in_source():
    import inspect

    import tools.ablation_runner as mod

    src = inspect.getsource(mod)
    # The causal-agency runner's old sins: floors/clamps that manufacture a result.
    assert "Ensure floor" not in src
    assert "Ensure low divergence" not in src
