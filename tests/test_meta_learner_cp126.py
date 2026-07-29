"""Meta-learner: noisy steps committed without proof, and gradients off by sigma."""
from __future__ import annotations

import numpy as np
import pytest

from core.adaptation import meta_learner as ml
from core.adaptation.meta_learner import (
    ESMetaOptimizer,
    MetaConfig,
    MetaLearner,
    MetaTask,
)

pytestmark = pytest.mark.unit


@pytest.fixture()
def isolated(tmp_path, monkeypatch):
    monkeypatch.setattr(ml, "_DATA_DIR", tmp_path)
    monkeypatch.setattr(ml, "_STATE_PATH", tmp_path / "meta_state.npz")
    return tmp_path


def test_es_gradient_uses_the_canonical_variance_scaling():
    """eps already carries sigma, so the canonical (Salimans) estimator divides
    by n*sigma^2. Dividing by n*sigma instead cancelled sigma entirely, making
    the estimate sigma-INDEPENDENT — a different estimator, and one whose
    magnitude no longer tracked the configured perturbation scale.

    With reward normalisation the absolute magnitude is set by the shaping, so
    what is asserted here is the relationship the canonical form implies: the
    estimate scales as 1/sigma. The old form produced the same value for both
    sigmas.
    """
    def grad_at(sigma: float) -> float:
        opt = ESMetaOptimizer(MetaConfig(n_perturbations=512,
                                         perturbation_sigma=sigma,
                                         antithetic=True))
        g, _ = opt.estimate_gradient(np.zeros(3), lambda p: float(p[0]))
        return float(g[0])

    coarse, fine = grad_at(0.1), grad_at(0.01)

    assert coarse > 0 and fine > 0, "gradient must point uphill"
    # 1/sigma scaling: a 10x smaller sigma gives a ~10x larger estimate.
    assert 5.0 < fine / coarse < 20.0


def test_non_finite_rewards_do_not_poison_the_gradient():
    """One NaN reward flowed through normalization and poisoned mean/std for
    EVERY perturbation, not just its own."""
    opt = ESMetaOptimizer(MetaConfig(n_perturbations=16))
    calls = {"n": 0}

    def flaky(p):
        calls["n"] += 1
        return float("nan") if calls["n"] % 5 == 0 else float(p[0])

    grad, metrics = opt.estimate_gradient(np.zeros(2), flaky)

    assert np.all(np.isfinite(grad))
    assert metrics["non_finite_rewards"] > 0


def test_all_non_finite_rewards_skip_the_step():
    opt = ESMetaOptimizer(MetaConfig(n_perturbations=8))

    grad, metrics = opt.estimate_gradient(np.zeros(2), lambda p: float("nan"))

    assert np.allclose(grad, 0.0)
    assert metrics["n_evaluations"] == 0


def test_a_worsening_step_is_rejected_and_rolled_back(isolated):
    """Every noisy ES step used to be committed with no baseline comparison and
    nothing to roll back to, so a bad step became the permanent starting point."""
    learner = MetaLearner()
    # An evaluator that punishes any movement away from the origin.
    task = MetaTask(
        name="stay",
        evaluate=lambda p: -float(np.linalg.norm(p)),
        parameter_dim=3,
        baseline_params=np.zeros(3),
    )
    learner.register_task(task)
    before = learner.get_meta_params("stay").copy()

    learner.meta_step("stay")

    after = learner.get_meta_params("stay")
    assert np.linalg.norm(after) <= np.linalg.norm(before) + 1e-9


def test_an_improving_step_is_accepted(isolated):
    """The acceptance gate must not simply freeze learning."""
    learner = MetaLearner()
    task = MetaTask(
        name="climb",
        evaluate=lambda p: float(p[0]),
        parameter_dim=2,
        baseline_params=np.zeros(2),
    )
    learner.register_task(task)

    learner.meta_step("climb")

    assert learner.get_meta_params("climb")[0] > 0.0


def test_declared_dimension_must_match_the_baseline(isolated):
    """parameter_dim was declared and never checked."""
    learner = MetaLearner()
    bad = MetaTask(name="mismatch", evaluate=lambda p: 0.0,
                   parameter_dim=10, baseline_params=np.zeros(3))

    with pytest.raises(ValueError, match="parameter_dim"):
        learner.register_task(bad)


def test_non_finite_persisted_parameters_are_discarded(isolated):
    """Persisted arrays were trusted straight into the parameters."""
    np.savez_compressed(
        isolated / "meta_state.npz",
        cycle_count=np.array([3]),
        meta_poison=np.array([np.nan, 1.0, 2.0]),
    )

    learner = MetaLearner()

    assert learner.get_meta_params("poison") is None


def test_save_is_atomic(isolated):
    learner = MetaLearner()
    learner.register_task(MetaTask(name="t", evaluate=lambda p: float(p[0]),
                                   parameter_dim=2, baseline_params=np.zeros(2)))
    learner.meta_step("t")

    assert (isolated / "meta_state.npz").exists()
    assert not list(isolated.glob("*.tmp"))
