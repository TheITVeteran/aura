"""The campaign could not admit the tasks its own compiler emitted.

task_depths, train_depths and heldout_depths were three strings typed into
each profile. The semantic-micro migration changed what the frontier compiler
produces — the deepest program went from 10 steps to 28 — and nothing connected
the two, so _validate_task_depth_admission refused a prepared campaign with
compiled_task_depth_not_train_admitted.

The ladder is now read off the compiler. These tests hold every frontier
profile against its own battery, so the next compilation change moves the
ladder with it instead of stranding it.
"""
from __future__ import annotations

import pytest

from core.learning.frontier_process_supervision import frontier_process_task_battery
from core.learning.unified_intrinsic_objective import UnifiedIntrinsicTrainingSpec
from tools.prepare_unified_intrinsic_resident_campaign import (
    PROFILES,
    _profile_training,
    _validate_task_depth_admission,
)

_FRONTIER_PROFILES = sorted(
    profile
    for profile in PROFILES
    if _profile_training(profile).get("task_source") == "frontier_process"
)


def _battery(training):
    families = tuple(training["families"].split(","))
    difficulties = tuple(
        int(value) for value in training["frontier_difficulties"].split(",")
    )
    train_tasks = frontier_process_task_battery(
        families,
        difficulties,
        int(training["per_cell"]),
        seed=int(training["seed"]),
        registry_version=str(training["frontier_registry_version"]),
    )
    holdout_tasks = frontier_process_task_battery(
        families,
        difficulties,
        int(training["holdout_per_cell"]),
        seed=int(training["seed"]) + 9_973,
        registry_version=str(training["frontier_registry_version"]),
        excluded_prompts=tuple(task.prompt for task in train_tasks),
    )
    return (*train_tasks, *holdout_tasks)


def test_there_are_frontier_profiles_to_check():
    """A sweep over an empty list proves nothing."""
    assert len(_FRONTIER_PROFILES) >= 10


@pytest.mark.parametrize("profile", _FRONTIER_PROFILES)
class TestEveryFrontierProfile:
    def test_the_ladder_matches_its_own_compiler(self, profile):
        training = _profile_training(profile)
        compiled = sorted({int(task.depth) for task in _battery(training)})
        assert training["task_depths"] == ",".join(str(depth) for depth in compiled)

    def test_every_compiled_depth_is_admitted(self, profile):
        """The exact check that refused the prepared campaign."""
        training = _profile_training(profile)
        _validate_task_depth_admission(training, _battery(training))

    def test_training_includes_the_anchor(self, profile):
        training = _profile_training(profile)
        admitted = [int(value) for value in training["train_depths"].split(",")]
        assert 1 in admitted

    def test_heldout_extrapolates_beyond_training(self, profile):
        training = _profile_training(profile)
        admitted = [int(value) for value in training["train_depths"].split(",")]
        heldout = [int(value) for value in training["heldout_depths"].split(",")]
        assert min(heldout) > max(admitted)
        assert not set(heldout) & set(admitted)

    def test_the_spec_accepts_the_derived_ladder(self, profile):
        """UnifiedIntrinsicTrainingSpec is the contract the ladder has to
        satisfy; deriving numbers that it then rejects would swap one broken
        campaign for another."""
        training = _profile_training(profile)
        spec = UnifiedIntrinsicTrainingSpec(
            prelude_end=1,
            coda_start=2,
            train_depths=tuple(
                int(value) for value in training["train_depths"].split(",")
            ),
            heldout_depths=tuple(
                int(value) for value in training["heldout_depths"].split(",")
            ),
        )
        assert spec.depths == spec.train_depths + spec.heldout_depths


def test_a_profile_whose_compiler_moved_is_refused():
    """The drift this replaces: a ladder that stopped matching its compiler.

    Pinning the old hardcoded depths against a battery that now runs deeper is
    exactly the state the prepared campaign was in.
    """
    training = _profile_training("process_neural_acquisition")
    stale = {**training, "train_depths": "1,3,4,5,6,9,10"}
    with pytest.raises(RuntimeError, match="compiled_task_depth_not_train_admitted"):
        _validate_task_depth_admission(stale, _battery(training))
