"""CP126 contract tests for the personality → physics bridge."""
from __future__ import annotations

import math
import threading
from types import SimpleNamespace

import pytest

from core.brain import personality_bridge as pb_module
from core.brain.personality_bridge import PHYSICS_BOUNDS, PersonalityBridge


class _Array(list):
    """Minimal stand-in for a numpy array slice-assign target."""

    def copy(self):
        return _Array(self)

    def __mul__(self, factor):
        return _Array(value * factor for value in self)

    __rmul__ = __mul__

    def __setitem__(self, key, value):
        if isinstance(key, slice):
            values = list(value)
            if len(values) != len(self):
                raise ValueError("shape mismatch")
            for index, item in enumerate(values):
                list.__setitem__(self, index, item)
            return
        list.__setitem__(self, key, value)

    @property
    def shape(self):
        return (len(self),)


class _Model:
    def __init__(self, joints=(), neck_adr=None):
        self.jnt_stiffness = _Array([10.0, 20.0])
        self.dof_damping = _Array([1.0, 2.0])
        self._joints = dict(joints)
        self._neck_adr = neck_adr

    def joint(self, name):
        if name in self._joints:
            return SimpleNamespace(qposadr=self._joints[name])
        raise KeyError(name)


class _Body:
    def __init__(self, model=None, data=None, lock=None):
        self.model = model
        self.data = data
        if lock is not None:
            self.physics_lock = lock


def _affect(**overrides):
    base = {"valence": 0.5, "arousal": 0.5, "curiosity": 0.5}
    base.update(overrides)
    return SimpleNamespace(**base)


@pytest.fixture()
def registry(monkeypatch):
    services: dict = {}
    monkeypatch.setattr(
        pb_module,
        "get_runtime_service",
        lambda name, default=None: services.get(name, default),
    )
    return services


class _Repo:
    def __init__(self, state):
        self._state = state

    async def get_current(self):
        return self._state


def _register_state(registry, affect):
    registry["state_repository"] = _Repo(SimpleNamespace(affect=affect))


# --- 89fa9c28: affect must not become an invalid physics constant ----------


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf"), None, "hot"])
def test_nonfinite_affect_yields_bounded_constants(bad):
    mods = PersonalityBridge().derive_physics_modifiers(
        SimpleNamespace(valence=bad, arousal=bad, curiosity=bad)
    )

    for name, (low, high) in PHYSICS_BOUNDS.items():
        value = mods[name]
        assert math.isfinite(value)
        assert low <= value <= high
    assert mods["input_faults"]


def test_out_of_range_affect_is_clamped_and_reported():
    mods = PersonalityBridge().derive_physics_modifiers(
        SimpleNamespace(valence=-50.0, arousal=99.0, curiosity=-3.0)
    )

    assert PHYSICS_BOUNDS["stiffness_mult"][0] <= mods["stiffness_mult"] <= PHYSICS_BOUNDS["stiffness_mult"][1]
    assert mods["jitter"] <= PHYSICS_BOUNDS["jitter"][1]
    assert len(mods["input_faults"]) == 3


def test_normal_affect_produces_no_faults():
    mods = PersonalityBridge().derive_physics_modifiers(_affect())
    assert mods["input_faults"] == []
    assert mods["damping_mult"] > 0


def test_missing_affect_fields_fall_back_to_defaults():
    mods = PersonalityBridge().derive_physics_modifiers(SimpleNamespace())
    assert math.isfinite(mods["stiffness_mult"])
    assert mods["input_faults"] == []


# --- fe8e6eee: writes must use the body's barrier when it has one ----------


def test_body_lock_is_acquired_for_the_write(registry):
    events = []

    class _Lock:
        def __enter__(self):
            events.append("enter")

        def __exit__(self, *exc):
            events.append("exit")
            return False

    _register_state(registry, _affect())
    body = _Body(model=_Model(), lock=_Lock())

    import asyncio

    receipt = asyncio.run(PersonalityBridge().sync_embodiment(body))

    assert receipt["applied"] is True
    assert receipt["synchronization"] == "physics_lock"
    assert events == ["enter", "exit"]


def test_unsynchronized_body_is_flagged(registry, monkeypatch):
    recorded = []
    monkeypatch.setattr(pb_module, "record_degradation", lambda *a, **k: recorded.append(a))
    _register_state(registry, _affect())
    body = _Body(model=_Model())

    import asyncio

    receipt = asyncio.run(PersonalityBridge().sync_embodiment(body))

    assert receipt["applied"] is True
    assert receipt["synchronization"] == "none"
    assert recorded


def test_fallback_lock_is_a_real_lock():
    assert isinstance(pb_module._FALLBACK_APPLY_LOCK, type(threading.RLock()))


# --- d38c257a: the baseline must be versioned ------------------------------


def test_baseline_is_captured_once_and_reused(registry):
    import asyncio

    _register_state(registry, _affect())
    model = _Model()
    body = _Body(model=model)
    bridge = PersonalityBridge()

    first = asyncio.run(bridge.sync_embodiment(body))
    scaled_once = list(model.jnt_stiffness)
    second = asyncio.run(bridge.sync_embodiment(body))

    assert first["baseline"] == "captured"
    assert second["baseline"] == "reused"
    # No compounding drift on the second application.
    assert list(model.jnt_stiffness) == scaled_once


def test_model_replacement_refreshes_the_baseline(registry):
    import asyncio

    _register_state(registry, _affect())
    body = _Body(model=_Model())
    bridge = PersonalityBridge()
    asyncio.run(bridge.sync_embodiment(body))

    body.model = _Model()
    receipt = asyncio.run(bridge.sync_embodiment(body))

    assert receipt["baseline"] == "recaptured"


def test_shape_change_refreshes_the_baseline(registry):
    import asyncio

    _register_state(registry, _affect())
    model = _Model()
    body = _Body(model=model)
    bridge = PersonalityBridge()
    asyncio.run(bridge.sync_embodiment(body))

    model.jnt_stiffness = _Array([1.0, 2.0, 3.0])
    model.dof_damping = _Array([1.0, 2.0, 3.0])
    receipt = asyncio.run(bridge.sync_embodiment(body))

    assert receipt["baseline"] == "recaptured"
    assert receipt["applied"] is True


def test_reset_baseline_forces_a_recapture(registry):
    import asyncio

    _register_state(registry, _affect())
    body = _Body(model=_Model())
    bridge = PersonalityBridge()
    asyncio.run(bridge.sync_embodiment(body))

    assert bridge.reset_baseline(body) is True
    assert asyncio.run(bridge.sync_embodiment(body))["baseline"] == "captured"
    assert bridge.reset_baseline(body) is True


# --- 047a0b0b: postural control is applied or declared absent --------------


def test_gaze_is_actually_applied_when_a_neck_joint_exists(registry):
    import asyncio

    _register_state(registry, _affect(curiosity=1.0))
    model = _Model(joints={"neck": [1]})
    body = _Body(model=model, data=SimpleNamespace(qpos=_Array([0.0, 0.0, 0.0])))

    receipt = asyncio.run(PersonalityBridge().sync_embodiment(body))

    assert receipt["tilt_applied"] is True
    assert receipt["tilt_reason"] == "applied"
    assert body.data.qpos[1] == pytest.approx(receipt["tilt_bias"])


def test_missing_neck_joint_is_declared_not_silently_skipped(registry):
    import asyncio

    _register_state(registry, _affect())
    body = _Body(model=_Model(), data=SimpleNamespace(qpos=_Array([0.0])))

    receipt = asyncio.run(PersonalityBridge().sync_embodiment(body))

    assert receipt["tilt_applied"] is False
    assert receipt["tilt_reason"] == "no_neck_joint"


def test_explicit_neck_index_is_honoured(registry):
    import asyncio

    _register_state(registry, _affect(curiosity=0.8))
    body = _Body(model=_Model(), data=SimpleNamespace(qpos=_Array([0.0, 0.0])))
    body.neck_qpos_index = 0

    receipt = asyncio.run(PersonalityBridge().sync_embodiment(body))

    assert receipt["tilt_applied"] is True
    assert body.data.qpos[0] == pytest.approx(receipt["tilt_bias"])


def test_out_of_range_neck_index_is_refused(registry):
    import asyncio

    _register_state(registry, _affect())
    body = _Body(model=_Model(), data=SimpleNamespace(qpos=_Array([0.0])))
    body.neck_qpos_index = 12

    receipt = asyncio.run(PersonalityBridge().sync_embodiment(body))

    assert receipt["tilt_applied"] is False
    assert "out_of_range" in receipt["tilt_reason"]


# --- 4995f0ab: every outcome produces a receipt ----------------------------


def test_missing_repository_reports_a_reason(registry):
    import asyncio

    receipt = asyncio.run(PersonalityBridge().sync_embodiment(_Body()))

    assert receipt["applied"] is False
    assert receipt["reason"] == "state_repository_unavailable"


def test_missing_state_reports_a_reason(registry):
    import asyncio

    registry["state_repository"] = _Repo(None)

    receipt = asyncio.run(PersonalityBridge().sync_embodiment(_Body()))

    assert receipt["reason"] == "state_unavailable"


def test_missing_model_still_returns_modifiers(registry):
    import asyncio

    _register_state(registry, _affect())

    receipt = asyncio.run(PersonalityBridge().sync_embodiment(SimpleNamespace(model=None)))

    assert receipt["reason"] == "no_physics_model"
    assert receipt["damping_mult"] > 0
    assert receipt["applied"] is False


def test_last_affect_is_updated(registry):
    import asyncio

    affect = _affect(valence=-0.4)
    _register_state(registry, affect)
    bridge = PersonalityBridge()

    asyncio.run(bridge.sync_embodiment(_Body(model=_Model())))

    assert bridge.last_affect is affect
    assert bridge.status()["applied"] is True


def test_unavailable_repository_records_a_degradation(registry, monkeypatch):
    import asyncio

    recorded = []
    monkeypatch.setattr(pb_module, "record_degradation", lambda *a, **k: recorded.append(a))

    asyncio.run(PersonalityBridge().sync_embodiment(_Body()))

    assert recorded and recorded[0][0] == "personality_bridge"


# --- 288b0d62: ordinary numeric/backend failures stay inside the bridge ----


@pytest.mark.parametrize(
    "exc", [TypeError("t"), ValueError("v"), IndexError("i"), OSError("o"), ZeroDivisionError("z")]
)
def test_backend_failures_are_isolated(registry, exc):
    import asyncio

    class Exploding(_Repo):
        async def get_current(self):
            raise exc

    registry["state_repository"] = Exploding(None)

    receipt = asyncio.run(PersonalityBridge().sync_embodiment(_Body()))

    assert receipt["applied"] is False
    assert type(exc).__name__ in receipt["reason"]


def test_shape_mismatch_does_not_escape(registry):
    import asyncio

    _register_state(registry, _affect())
    model = _Model()
    body = _Body(model=model)
    bridge = PersonalityBridge()
    asyncio.run(bridge.sync_embodiment(body))

    # Baseline now has two entries; the live array shrinks underneath it in a
    # way the fingerprint cannot see (same object, mutated in place).
    stored = body._aura_physics_baseline
    stored["stiffness"] = _Array([1.0, 2.0, 3.0])

    receipt = asyncio.run(bridge.sync_embodiment(body))

    assert receipt["applied"] is False
    assert "ValueError" in receipt["reason"]
