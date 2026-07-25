"""An immune remedy must report what it did, and a stuck one must be visible.

The 2026-07-25 idle window fired ``reallocate_flow`` 247 times with byte-identical
parameters. 55 logged a transfer; the other 192 failed and produced no log line
at all, because the failure only ever landed in a returned dict nobody read. So
the immune system spent a quiet hour re-issuing a remedy that was not relieving
anything, and nothing in the runtime could tell the difference between "treated"
and "tried and achieved nothing".

Two contracts. The actuator reports the MEASURED delta rather than the requested
amount — the world clips transfers, so the request is a hope and the delta is the
fact. And a failing action is logged, with a degradation once the same action has
failed repeatedly.
"""
from __future__ import annotations

import pytest

from core.actuators.actuator_registry import ReallocateFlowActuator
from core.adaptation.immune_executor import ImmuneHeuristicExecutor

pytestmark = pytest.mark.unit


class FakeNode:
    kind = "node"

    def __init__(self, load: float, capacity: float):
        self.load = float(load)
        self.capacity = float(capacity)

    def enforce_constraints(self) -> None:
        self.load = max(0.0, min(self.load, self.capacity))


class FakeWorld:
    """Applies transfers the way the real world model does — with clipping."""

    def __init__(self, entities, *, apply=True):
        self.entities = entities
        self._apply = apply

    def get_entity(self, entity_id):
        return self.entities.get(entity_id)

    def simulate(self, duration_s, actions=None):
        if not self._apply:
            return {}
        for action in actions or []:
            if action.get("type") != "transfer":
                continue
            src = self.entities[action["entity_id"]]
            dst = self.entities[action["target_id"]]
            qty = min(float(action["amount"]), dst.capacity - dst.load)
            qty = max(0.0, min(qty, src.load))
            src.load -= qty
            dst.load += qty
        return {}


@pytest.fixture()
def patch_world(monkeypatch):
    def _install(world):
        monkeypatch.setattr(
            "core.world.world_model.get_physics_world_model", lambda: world
        )
    return _install


class TestMeasuredEffect:
    def test_a_full_transfer_reports_the_amount_that_moved(self, patch_world):
        world = FakeWorld({"A": FakeNode(1000, 1000), "B": FakeNode(0, 1000)})
        patch_world(world)

        result = ReallocateFlowActuator().execute(
            {"source_id": "A", "target_id": "B", "amount": 384.0}
        )

        assert result.success
        assert result.updates["_measured"]["moved"] == pytest.approx(384.0)
        assert world.entities["B"].load == pytest.approx(384.0)

    def test_a_clipped_transfer_reports_what_actually_moved(self, patch_world):
        """Requested 384, the target could only take 100 — say 100, not 384."""
        world = FakeWorld({"A": FakeNode(1000, 1000), "B": FakeNode(900, 1000)})
        patch_world(world)

        result = ReallocateFlowActuator().execute(
            {"source_id": "A", "target_id": "B", "amount": 384.0}
        )

        assert result.success
        assert result.updates["_measured"]["moved"] == pytest.approx(100.0)
        assert "clipped" in result.message
        assert "384" not in result.message.split("clipped")[0]

    def test_a_transfer_that_moves_nothing_is_a_failure(self, patch_world):
        """The old code called this a success and claimed the full amount."""
        world = FakeWorld(
            {"A": FakeNode(1000, 1000), "B": FakeNode(0, 1000)}, apply=False
        )
        patch_world(world)

        result = ReallocateFlowActuator().execute(
            {"source_id": "A", "target_id": "B", "amount": 384.0}
        )

        assert not result.success
        assert "moved nothing" in result.message

    def test_a_target_at_capacity_is_refused_before_simulating(self, patch_world):
        world = FakeWorld({"A": FakeNode(1000, 1000), "B": FakeNode(1000, 1000)})
        patch_world(world)

        result = ReallocateFlowActuator().execute(
            {"source_id": "A", "target_id": "B", "amount": 384.0}
        )

        assert not result.success
        assert "maximum capacity" in result.message


class Result:
    def __init__(self, success, message="no effect"):
        self.success = success
        self.message = message


class TestStuckRemedyIsVisible:
    @pytest.fixture(autouse=True)
    def clean_ledger(self):
        ImmuneHeuristicExecutor._failure_streaks.clear()
        yield
        ImmuneHeuristicExecutor._failure_streaks.clear()

    def test_the_first_failure_is_logged(self, caplog):
        with caplog.at_level("WARNING"):
            ImmuneHeuristicExecutor._note_action_outcome(
                "reallocate_flow", {"amount": 384.0}, Result(False)
            )
        assert "did not take effect" in caplog.text

    def test_a_repeated_failure_escalates_once(self, monkeypatch):
        recorded: list[dict] = []
        monkeypatch.setattr(
            "core.adaptation.immune_executor.record_degradation",
            lambda *a, **kw: recorded.append(kw),
        )

        for _ in range(12):
            ImmuneHeuristicExecutor._note_action_outcome(
                "reallocate_flow", {"amount": 384.0}, Result(False)
            )

        assert len(recorded) == 1, "escalate once, do not create a degradation storm"
        assert recorded[0]["extra"]["streak"] == 5

    def test_a_success_clears_the_streak(self, monkeypatch):
        recorded: list[dict] = []
        monkeypatch.setattr(
            "core.adaptation.immune_executor.record_degradation",
            lambda *a, **kw: recorded.append(kw),
        )

        for _ in range(4):
            ImmuneHeuristicExecutor._note_action_outcome(
                "reallocate_flow", {"amount": 384.0}, Result(False)
            )
        ImmuneHeuristicExecutor._note_action_outcome(
            "reallocate_flow", {"amount": 384.0}, Result(True, "ok")
        )
        for _ in range(4):
            ImmuneHeuristicExecutor._note_action_outcome(
                "reallocate_flow", {"amount": 384.0}, Result(False)
            )

        assert recorded == [], "a remedy that started working is not stuck"

    def test_different_parameters_are_tracked_separately(self, monkeypatch):
        recorded: list[dict] = []
        monkeypatch.setattr(
            "core.adaptation.immune_executor.record_degradation",
            lambda *a, **kw: recorded.append(kw),
        )

        for amount in range(10):
            ImmuneHeuristicExecutor._note_action_outcome(
                "reallocate_flow", {"amount": float(amount)}, Result(False)
            )

        assert recorded == [], "varying the remedy is not the same as being stuck"

    def test_the_ledger_is_bounded(self):
        for i in range(ImmuneHeuristicExecutor._FAILURE_LEDGER_CAP * 3):
            ImmuneHeuristicExecutor._note_action_outcome(
                "reallocate_flow", {"amount": float(i)}, Result(False)
            )
        assert (
            len(ImmuneHeuristicExecutor._failure_streaks)
            <= ImmuneHeuristicExecutor._FAILURE_LEDGER_CAP
        )
