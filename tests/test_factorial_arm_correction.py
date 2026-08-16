"""CP126 ba3ffbac: the more mechanisms you test, the more you get credit for.

Each arm of the factorial ablation was graded on its own, and any arm reaching
PROVEN or SUPPORTED joined the attribution list. Holm correction ran INSIDE a
claim, across that arm's families, and never across the arms themselves. Seven
mechanisms at alpha 0.05 therefore carried roughly a one-in-three chance of
attributing a gain to a mechanism that did nothing — and the caller supplies
the arm tuple, so the error rate was the caller's to inflate.
"""
from __future__ import annotations

from core.brain.llm.latent_cortex.experiments import (
    ExperimentProvenance,
    Task,
    run_factorial_ablations,
)

_PROVENANCE = ExperimentProvenance(
    task_manifest_sha256="a" * 64,
    checkpoint_fingerprint="b" * 64,
    schedule_sha256="c" * 64,
    verifier_version="verifier-1.0.0",
    environment_sha256="d" * 64,
)


def _tasks(count: int = 40) -> dict[str, list[Task]]:
    return {
        "math": [
            Task(
                prompt=f"task {index}",
                answer=str(index),
                depth=2,
                family="math",
                seed=index,
            )
            for index in range(count)
        ]
    }


def _solver(winners: set[str]):
    """Every listed arm wins outright; every other arm matches vanilla."""

    def solve(task: Task, arm: str) -> tuple[bool, int]:
        if arm in winners and arm != "vanilla":
            return True, 100
        return task.seed % 4 == 0, 100

    return solve


class TestArmCorrection:
    def test_the_result_reports_what_it_corrected_against(self):
        result = run_factorial_ablations(
            _solver({"latent_opt"}),
            _tasks(),
            arms=("latent_opt", "fast_weights", "recurrence"),
            provenance=_PROVENANCE,
        )
        assert result["arms_tested"] == 3
        assert set(result["arm_holm_adjusted_p"]) == {
            "latent_opt",
            "fast_weights",
            "recurrence",
        }
        assert result["arm_family_wise_alpha"] == 0.05

    def test_a_real_mechanism_survives_correction(self):
        result = run_factorial_ablations(
            _solver({"latent_opt"}),
            _tasks(),
            arms=("latent_opt", "fast_weights"),
            provenance=_PROVENANCE,
        )
        assert "latent_opt" in result["attribution"]
        assert result["arm_holm_adjusted_p"]["latent_opt"] < 0.05

    def test_an_inert_mechanism_is_not_attributed(self):
        result = run_factorial_ablations(
            _solver(set()),
            _tasks(),
            arms=("latent_opt", "fast_weights", "recurrence"),
            provenance=_PROVENANCE,
        )
        assert result["attribution"] == []

    def test_correction_is_visible_as_a_separate_step(self):
        """Both lists are published, so a reader can see what the correction
        cost rather than only what survived it."""
        result = run_factorial_ablations(
            _solver({"latent_opt"}),
            _tasks(),
            arms=("latent_opt", "fast_weights"),
            provenance=_PROVENANCE,
        )
        assert set(result["attribution"]) <= set(
            result["attribution_before_arm_correction"]
        )

    def test_adding_arms_raises_every_arms_adjusted_p(self):
        """The cost of looking in more places lands on every arm.

        This is the property the old code lacked entirely: the same evidence
        for the same mechanism was judged identically whether it was one of
        two candidates or one of twenty.
        """
        few = run_factorial_ablations(
            _solver({"latent_opt"}),
            _tasks(),
            arms=("latent_opt", "fast_weights"),
            provenance=_PROVENANCE,
        )
        many = run_factorial_ablations(
            _solver({"latent_opt"}),
            _tasks(),
            arms=(
                "latent_opt",
                "fast_weights",
                "recurrence",
                "retrieval",
                "contrastive",
                "speculative",
            ),
            provenance=_PROVENANCE,
        )
        assert (
            many["arm_holm_adjusted_p"]["latent_opt"]
            >= few["arm_holm_adjusted_p"]["latent_opt"]
        )
        assert many["arms_tested"] > few["arms_tested"]
