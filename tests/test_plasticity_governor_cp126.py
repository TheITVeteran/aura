"""EWC plasticity governor: the defects that made it protect nothing.

The governor's job is to stop new learning from overwriting old learning. Each
test here corresponds to a way it silently failed to do that.
"""
from __future__ import annotations

import numpy as np
import pytest

from core.adaptation import plasticity_governor as pg
from core.adaptation.plasticity_governor import (
    ParameterSnapshot,
    PlasticityConfig,
    PlasticityGovernor,
)

pytestmark = pytest.mark.unit


@pytest.fixture()
def isolated(tmp_path, monkeypatch):
    monkeypatch.setattr(pg, "_DATA_DIR", tmp_path)
    monkeypatch.setattr(pg, "_FISHER_PATH", tmp_path / "fisher_state.npz")
    return tmp_path


def _train(gov: PlasticityGovernor, name: str, n: int = 12, scale: float = 1.0) -> None:
    rng = np.random.default_rng(0)
    for _ in range(n):
        gov.record_gradient(name, rng.normal(scale=scale, size=4))


def test_persisted_fisher_survives_a_restart(isolated):
    """_load iterated self._snapshots, which is EMPTY at construction, so every
    restored Fisher was discarded and protection reset on each boot."""
    first = PlasticityGovernor(PlasticityConfig(min_samples_for_fisher=2))
    first.register_parameters("W", np.ones(4))
    _train(first, "W", n=6)
    first.consolidate()
    saved = first.get_importance_map("W")
    assert float(np.sum(saved)) > 0.0

    second = PlasticityGovernor(PlasticityConfig(min_samples_for_fisher=2))
    second.register_parameters("W", np.ones(4))

    restored = second.get_importance_map("W")
    assert restored is not None
    assert float(np.sum(restored)) > 0.0
    np.testing.assert_allclose(restored, saved)


def test_consolidation_updates_the_anchor(isolated):
    """theta_star stayed at REGISTRATION values, so the penalty pulled toward
    the initial weights forever and consolidation points never existed."""
    gov = PlasticityGovernor(PlasticityConfig(min_samples_for_fisher=2))
    gov.register_parameters("W", np.zeros(4))
    _train(gov, "W", n=4)

    moved = np.array([5.0, 5.0, 5.0, 5.0])
    gov.consolidate(current_params={"W": moved})

    np.testing.assert_allclose(gov._snapshots["W"].theta_star, moved)


def test_min_samples_is_actually_enforced(isolated):
    """Exposed as a safety parameter and never read: Fisher could be estimated
    from a single gradient and treated as an importance map."""
    gov = PlasticityGovernor(PlasticityConfig(min_samples_for_fisher=10))
    gov.register_parameters("W", np.ones(4))
    gov.record_gradient("W", np.ones(4))

    assert gov.consolidate() == []
    assert float(np.sum(gov.get_importance_map("W"))) == 0.0


def test_consolidation_interval_fires_automatically(isolated):
    """Also exposed and never read — nothing consolidated unless an external
    caller happened to remember."""
    gov = PlasticityGovernor(
        PlasticityConfig(consolidation_interval=5, min_samples_for_fisher=2)
    )
    gov.register_parameters("W", np.ones(4))
    _train(gov, "W", n=5)

    assert float(np.sum(gov.get_importance_map("W"))) > 0.0


def test_non_finite_gradients_are_dropped_not_absorbed(isolated):
    """A NaN norm compares False against every threshold, so the clip passed
    non-finite gradients straight into Fisher, poisoning it permanently."""
    gov = PlasticityGovernor(PlasticityConfig(min_samples_for_fisher=1))
    gov.register_parameters("W", np.ones(4))

    gov.record_gradient("W", np.array([np.nan, 1.0, 1.0, 1.0]))
    gov.record_gradient("W", np.array([np.inf, 1.0, 1.0, 1.0]))
    gov.record_gradient("W", np.ones(4))
    gov.consolidate()

    assert np.all(np.isfinite(gov.get_importance_map("W")))


def test_penalty_cannot_reverse_an_update(isolated):
    """The raw EWC gradient was subtracted with no norm cap, so a large lambda
    could produce a penalty bigger than the update — a brake that pushes."""
    gov = PlasticityGovernor(
        PlasticityConfig(ewc_lambda=1e6, min_samples_for_fisher=1)
    )
    gov.register_parameters("W", np.zeros(4))
    for _ in range(3):
        gov.record_gradient("W", np.ones(4))
    gov.consolidate()

    delta = np.array([0.01, 0.01, 0.01, 0.01])
    drifted = np.array([10.0, 10.0, 10.0, 10.0])
    penalized, report = gov.penalize_update("W", drifted, delta)

    assert np.all(np.isfinite(penalized))
    # It may cancel the update, but must never invert and amplify it.
    assert np.linalg.norm(penalized) <= np.linalg.norm(delta) + 1e-9
    assert report.penalized_delta_norm <= report.original_delta_norm + 1e-9


def test_misaligned_gradient_is_rejected_not_padded():
    """Padding hid index-alignment corruption: the Fisher entry for parameter i
    would be estimated from some other parameter's gradient."""
    snap = ParameterSnapshot("W", np.ones(4))

    with pytest.raises(ValueError, match="does not match"):
        snap.accumulate_gradient(np.ones(7))


def test_misaligned_parameters_are_rejected_in_penalty():
    snap = ParameterSnapshot("W", np.ones(4))

    with pytest.raises(ValueError, match="does not match"):
        snap.compute_penalty(np.ones(9), 1.0)


def test_config_rejects_values_that_would_disable_the_governor():
    """lambda/gamma/limits were never validated; a NaN gamma silently made the
    running average meaningless."""
    cfg = PlasticityConfig(
        ewc_lambda=float("nan"),
        fisher_gamma=float("inf"),
        consolidation_interval=0,
        min_samples_for_fisher=-5,
        gradient_clip=float("nan"),
    )

    assert cfg.ewc_lambda == 100.0
    assert 0.0 <= cfg.fisher_gamma < 1.0
    assert cfg.consolidation_interval >= 1
    assert cfg.min_samples_for_fisher >= 1
    assert cfg.gradient_clip > 0.0


def test_save_is_atomic(isolated):
    """A crash mid-write left a truncated .npz that the next boot failed to
    load, silently discarding every consolidation ever made."""
    gov = PlasticityGovernor(PlasticityConfig(min_samples_for_fisher=2))
    gov.register_parameters("W", np.ones(4))
    _train(gov, "W", n=4)
    gov.consolidate()

    assert (isolated / "fisher_state.npz").exists()
    assert not list(isolated.glob("*.tmp"))
