"""Compute receipts must be checked against what was actually allocated.

CP126 0673f84a: _receipt_contract_errors compared the worker's spend against
the worker's OWN reported maximum — which any receipt satisfies by reporting
a large enough ceiling. Nothing compared either number with what the facade
allocated, so an episode could spend far more than its budget and still
produce a self-consistent, passing receipt.
"""
from __future__ import annotations

import pytest

from core.brain.latent_cortex_service import LatentCortexService


def _check(receipt_budget, allocated):
    receipt = {"budget": receipt_budget}
    return LatentCortexService._receipt_contract_errors(
        receipt, {}, None, None, ..., "general", allocated_budget=allocated
    )


_ALLOCATED = {"max_layer_apps": 1_000_000, "wall_clock_s": 100.0}


def test_spending_within_the_allocation_is_accepted():
    errors = _check(
        {"spent_layer_apps": 500_000, "max_layer_apps": 1_000_000,
         "exhausted": False, "wall_clock_s": 50.0},
        _ALLOCATED,
    )

    assert "compute_spend_exceeds_allocation" not in errors
    assert "compute_ceiling_exceeds_allocation" not in errors
    assert "wall_clock_exceeds_allocation" not in errors


def test_a_worker_cannot_raise_its_own_ceiling():
    """The defect: report a bigger max and any spend looks compliant."""
    errors = _check(
        {"spent_layer_apps": 9_000_000, "max_layer_apps": 10_000_000,
         "exhausted": False},
        _ALLOCATED,
    )

    assert "compute_ceiling_exceeds_allocation" in errors


def test_overspending_the_allocation_is_caught():
    """Self-consistent (2M <= its own 5M ceiling) but over the 1M allocated."""
    errors = _check(
        {"spent_layer_apps": 2_000_000, "max_layer_apps": 5_000_000,
         "exhausted": False},
        _ALLOCATED,
    )

    assert "compute_spend_exceeds_allocation" in errors
    assert "compute_ceiling_exceeds_allocation" in errors


def test_a_receipt_inconsistent_with_itself_is_still_caught_first():
    errors = _check(
        {"spent_layer_apps": 2_000_000, "max_layer_apps": 1_000_000,
         "exhausted": False},
        _ALLOCATED,
    )

    assert "incomplete_or_exhausted_compute_receipt" in errors


def test_a_large_wall_clock_overrun_is_caught():
    errors = _check(
        {"spent_layer_apps": 100, "max_layer_apps": 1_000_000,
         "exhausted": False, "wall_clock_s": 400.0},
        _ALLOCATED,
    )

    assert "wall_clock_exceeds_allocation" in errors


def test_small_scheduling_noise_is_tolerated():
    """A deadline is not a stopwatch; a slight overrun is not a violation."""
    errors = _check(
        {"spent_layer_apps": 100, "max_layer_apps": 1_000_000,
         "exhausted": False, "wall_clock_s": 105.0},
        _ALLOCATED,
    )

    assert "wall_clock_exceeds_allocation" not in errors


def test_without_an_allocation_the_old_self_consistency_still_applies():
    errors = _check(
        {"spent_layer_apps": 2_000_000, "max_layer_apps": 1_000_000, "exhausted": False},
        None,
    )

    assert "incomplete_or_exhausted_compute_receipt" in errors


def test_an_exhausted_budget_is_still_refused():
    errors = _check(
        {"spent_layer_apps": 10, "max_layer_apps": 1_000, "exhausted": True},
        _ALLOCATED,
    )

    assert "incomplete_or_exhausted_compute_receipt" in errors


def test_a_missing_compute_receipt_is_still_refused():
    assert "missing_compute_receipt" in _check({}, _ALLOCATED)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), "soon", None, True])
def test_an_unusable_wall_clock_is_not_treated_as_an_overrun(bad):
    errors = _check(
        {"spent_layer_apps": 100, "max_layer_apps": 1_000_000,
         "exhausted": False, "wall_clock_s": bad},
        _ALLOCATED,
    )

    assert "wall_clock_exceeds_allocation" not in errors


def test_the_allocated_budget_reaches_the_checker():
    import inspect

    from core.brain import latent_cortex_service

    source = inspect.getsource(latent_cortex_service.LatentCortexService.deep_reason)
    assert "allocated_budget=budget" in source
