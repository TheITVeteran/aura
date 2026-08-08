"""Unloading the resident Cortex for nothing is the failure to prevent.

CP126 2a2791b1 + 402a99f0 — one defect from two sides.

The nucleus listener acted on `data.status == "success"` alone and
immediately unloaded the live Cortex, roughly 20GB of resident weights.
The event bus carries no publisher identity, so anything able to publish
could evict her mind at will, and a malformed or duplicated event did it
for free.

Then the reload used `self.cortex_path`, captured in `__init__` — so the
newly fused model was never bound. The lane unloaded and reloaded the SAME
weights and every optimization run was a no-op that cost a full reload.

The artifact is the authority: no loadable `fused_model` on disk, no
action. That needs no principal, cannot be forged by a publisher, and
makes the pointless unload impossible.
"""
from __future__ import annotations

import asyncio

import pytest

from core.brain.llm.nucleus_manager import NucleusManager


class _Manager:
    """The two fields the adoption path touches, without a real load."""

    def __init__(self, tmp_path):
        self.cortex_path = str(tmp_path / "old-cortex")
        self._adapter_dir = "some/adapter"
        self.unloaded: list[str] = []

    async def _unload_model_entry(self, name, reason=""):
        self.unloaded.append(f"{name}:{reason}")

    _adopt_promoted_cortex = NucleusManager._adopt_promoted_cortex


@pytest.fixture
def manager(tmp_path):
    return _Manager(tmp_path)


def _adopt(manager, data):
    asyncio.run(manager._adopt_promoted_cortex(data))


# --------------------------------------------- no artifact, no unload


def test_a_success_with_no_artifact_does_not_unload_the_cortex(manager):
    _adopt(manager, {"status": "success"})
    assert manager.unloaded == [], (
        "20GB of resident weights were evicted on an event that promoted nothing"
    )


def test_a_success_naming_a_missing_artifact_does_not_unload(manager, tmp_path):
    _adopt(manager, {"status": "success", "fused_model": str(tmp_path / "nope")})
    assert manager.unloaded == []


def test_the_cortex_path_is_not_moved_by_an_unusable_event(manager):
    before = manager.cortex_path
    _adopt(manager, {"status": "success", "fused_model": ""})
    assert manager.cortex_path == before


def test_an_empty_payload_is_survivable(manager):
    _adopt(manager, {})
    assert manager.unloaded == []


# ------------------------------------------ a real artifact IS adopted


def test_a_real_artifact_rebinds_the_lane_then_unloads(manager, tmp_path):
    promoted = tmp_path / "fused-cortex"
    promoted.mkdir()

    _adopt(manager, {"status": "success", "fused_model": str(promoted)})

    assert manager.cortex_path == str(promoted), (
        "the lane still points at the old weights, so the reload brings back "
        "the same model and the promotion is a no-op"
    )
    assert manager.unloaded == ["cortex:optimizer_promoted_artifact"]


def test_the_stale_adapter_is_dropped_with_the_old_weights(manager, tmp_path):
    promoted = tmp_path / "fused-cortex"
    promoted.mkdir()
    _adopt(manager, {"status": "success", "fused_model": str(promoted)})
    assert manager._adapter_dir == "", (
        "an adapter selected for the previous base model must not be carried "
        "onto newly fused weights"
    )


def test_the_lane_is_rebound_before_the_unload(manager, tmp_path):
    """Rebinding after the unload reproduces the original bug on next load."""
    promoted = tmp_path / "fused-cortex"
    promoted.mkdir()
    seen: list[str] = []

    async def _record_unload(name, reason=""):
        seen.append(manager.cortex_path)

    manager._unload_model_entry = _record_unload
    _adopt(manager, {"status": "success", "fused_model": str(promoted)})

    assert seen == [str(promoted)]
