"""A dead worker must not be able to fence the lane forever.

Live 2026-07-25, repeating, with no path back to service:

    [DEGRADATION] mlx_client (critical):
    durable_owner_release_not_confirmed:mlx:98014:<32B model>:token=2253
    -> lane left FENCED: durable owner ... could not be released during
       dead_worker_before_respawn; admission stays blocked until it is

``release_owner_sync`` returned ``False`` for two entirely different
situations — "this owner is already gone" and "a NEWER owner holds the lane" —
and ``mlx_client`` fences the lane on ``False``. So a settled release, the
ordinary case when a worker dies, blocked admission for the life of the
runtime. The cortex could not respawn, and every turn after it failed.

Same family as chip task_0b5865ec (MODEL_LOAD admission leases outliving dead
holders), on the durable-owner path.
"""
from __future__ import annotations

import os

import pytest

from core.runtime.model_lane_control import ModelLaneController

pytestmark = pytest.mark.unit


@pytest.fixture()
def controller(tmp_path, monkeypatch):
    monkeypatch.setenv("AURA_MODEL_LANE_STATE_PATH", str(tmp_path / "lane.json"))
    return ModelLaneController(state_path=tmp_path / "lane.json")


def _seed_owner(controller, owner_id, token, **extra):
    """Register an owner directly in controller state.

    The claim path runs through admission decisions; this test is about the
    RELEASE path, so the claim is seeded rather than driven.
    """
    with controller._thread_lock:
        state = controller._load_locked()
        state.setdefault("owners", {})[owner_id] = {
            "owner_id": owner_id,
            "fencing_token": int(token),
            **extra,
        }
        controller._save_locked(state)
    return owner_id, int(token)


class TestASettledReleaseIsNotAFailure:
    def test_releasing_an_owner_that_is_already_gone_succeeds(self, controller):
        """The live case: the worker died, the claim is gone, release it."""
        assert controller.release_owner_sync(
            "mlx:99999:/models/aura-32b",
            fencing_token=2253,
            reason="dead_worker_before_respawn",
        ) is True, (
            "a missing owner is a settled release; returning False fences the "
            "lane and blocks admission with no way back"
        )

    def test_a_normal_release_still_succeeds(self, controller):
        owner_id, token = _seed_owner(
            controller, f"mlx:{os.getpid()}:/models/aura-32b", 7
        )
        assert controller.release_owner_sync(
            owner_id, fencing_token=token, reason="worker_stopped"
        ) is True


class TestADeadHolderIsReaped:
    def test_a_token_mismatch_from_a_dead_holder_is_reaped(self, controller):
        """A corpse holding the door shut is not a conflict."""
        dead_pid = 999999          # not a live process on this host
        owner_id, token = _seed_owner(
            controller, f"mlx:{dead_pid}:/models/aura-32b", 2253
        )
        # Release with a STALE token: normally a hard conflict.
        assert controller.release_owner_sync(
            owner_id, fencing_token=token + 1, reason="dead_worker_before_respawn"
        ) is True

    def test_a_live_holder_is_never_reaped(self, controller):
        """Reaping on a guess would be the mirror of the bug it fixes."""
        owner_id, token = _seed_owner(
            controller, f"mlx:{os.getpid()}:/models/aura-32b", 11
        )
        assert controller.release_owner_sync(
            owner_id, fencing_token=token + 1, reason="some_other_reason"
        ) is False, "a live holder's claim must still win a token conflict"


class TestHolderLivenessFailsSafe:
    @pytest.mark.parametrize(
        "owner_id",
        [
            "mlx:not-a-pid:/models/x",
            "no-colons-at-all",
            "mlx::/models/x",
            "",
        ],
    )
    def test_an_unreadable_holder_counts_as_alive(self, owner_id):
        assert ModelLaneController._owner_holder_is_alive({}, owner_id=owner_id) is True

    def test_this_process_counts_as_alive(self):
        assert ModelLaneController._owner_holder_is_alive(
            {"holder_pid": os.getpid()}, owner_id="mlx:1:/x"
        ) is True

    def test_a_dead_pid_counts_as_dead(self):
        assert ModelLaneController._owner_holder_is_alive(
            {"holder_pid": 999999}, owner_id="mlx:999999:/x"
        ) is False
