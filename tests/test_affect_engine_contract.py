from __future__ import annotations

import asyncio
import time
from pathlib import Path

import numpy as np
import pytest

from core.affect.damasio_v2 import AffectEngineV2, DamasioMarkers


def _verified_context(event_id: str, *, intensity: float = 1.0) -> dict:
    return {
        "event_id": event_id,
        "source": "contract_test",
        "intensity": intensity,
        "evidence": {"kind": "test_observation"},
        "appraisal": {"v": 0.7, "a": 0.6, "e": 0.8},
    }


def test_public_somatic_output_is_explicitly_unitless() -> None:
    markers = DamasioMarkers()
    wheel = markers.get_wheel()

    assert wheel["physiology"]["classification"] == (
        "simulated_functional_indices_not_biomedical_measurements"
    )
    rendered = repr(wheel)
    assert "bpm" not in rendered
    assert "μS" not in rendered
    assert "μg/dL" not in rendered
    assert set(wheel["somatic_indices"]) == {"activation", "conductance", "stress", "mobilization"}


def test_numeric_artifact_loader_forbids_pickle(monkeypatch) -> None:
    observed: dict[str, object] = {}

    def guarded_load(path, *, allow_pickle):
        observed.update({"path": path, "allow_pickle": allow_pickle})
        raise ValueError("invalid artifact")

    original_exists = Path.exists
    monkeypatch.setattr(
        Path,
        "exists",
        lambda path: True if path.name == "weights.npz" else original_exists(path),
    )
    monkeypatch.setattr(np, "load", guarded_load)

    markers = DamasioMarkers()

    assert observed["allow_pickle"] is False
    assert markers.activation_index == pytest.approx((72.0 - 45.0) / 95.0)


def test_signed_valence_distinguishes_positive_and_negative_affect() -> None:
    engine = AffectEngineV2()
    engine.markers.emotions.update({key: 0.0 for key in engine.markers.emotions})
    engine.markers.emotions["joy"] = 0.8
    positive = engine.get_status()["valence"]
    engine.markers.emotions["joy"] = 0.0
    engine.markers.emotions["fear"] = 0.8
    negative = engine.get_status()["valence"]

    assert positive > 0.0
    assert negative < 0.0


@pytest.mark.asyncio
async def test_duplicate_stimulus_is_idempotent() -> None:
    engine = AffectEngineV2()
    first = await engine.react("novel_stimulus", _verified_context("evt-1"))
    after_first = dict(engine.markers.emotions)
    second = await engine.react("novel_stimulus", _verified_context("evt-1"))

    assert first["stimulus_receipt"]["applied"] is True
    assert first["stimulus_receipt"]["evidence_status"] == "observed"
    assert second["stimulus_receipt"]["duplicate"] is True
    assert second["stimulus_receipt"]["applied"] is False
    assert engine.markers.emotions == after_first


def test_unverified_label_is_bounded_and_cannot_create_complex_social_emotion() -> None:
    markers = DamasioMarkers()
    markers.somatic_update("positive_interaction", 1.0)

    assert markers.emotions["love"] == 0.0
    assert markers.emotions["belonging"] == 0.0
    assert markers.emotions["empathy"] == 0.0
    assert markers.emotions["joy"] == 0.0


def test_untrusted_source_cannot_self_assert_observed_evidence() -> None:
    event_id, source, evidence_status, intensity = AffectEngineV2._stimulus_context(
        "goal_achieved",
        {
            "source": "caller_controlled",
            "event_id": "claimed",
            "evidence": {"verified": True},
            "intensity": 1.0,
        },
    )
    assert (event_id, source, intensity) == ("claimed", "caller_controlled", 1.0)
    assert evidence_status == "unverified_legacy"


def test_silence_cannot_create_relational_emotions() -> None:
    markers = DamasioMarkers()
    markers.last_interaction_time = time.time() - 86_400
    before = {key: markers.emotions[key] for key in ("loneliness", "longing", "belonging")}

    deltas = markers.temporal_pulse(elapsed_s=300.0)

    assert not ({"loneliness", "longing", "belonging"} & set(deltas))
    assert {key: markers.emotions[key] for key in before} == before


def test_temporal_deltas_are_elapsed_time_normalized() -> None:
    once = DamasioMarkers()
    twice = DamasioMarkers()
    once.last_interaction_time = twice.last_interaction_time = time.time() - 600.0

    one_minute = once.temporal_pulse(elapsed_s=60.0)
    first_half = twice.temporal_pulse(elapsed_s=30.0)
    second_half = twice.temporal_pulse(elapsed_s=30.0)

    for key, value in one_minute.items():
        assert first_half[key] + second_half[key] == pytest.approx(value)


@pytest.mark.asyncio
async def test_get_is_a_pure_read() -> None:
    engine = AffectEngineV2()
    before = engine.get_snapshot()

    async def forbidden_pulse():
        raise AssertionError("read path invoked pulse")

    engine.pulse = forbidden_pulse
    first = await engine.get()
    second = await engine.get()

    assert first == second
    assert engine.get_snapshot() == before


def test_all_legacy_views_share_canonical_dimensions() -> None:
    engine = AffectEngineV2()
    engine.markers.emotions["fear"] = 0.7
    engine.markers.emotions["joy"] = 0.1

    snapshot = engine.get_snapshot()
    state = engine._snapshot_state()
    status = engine.get_status()
    current = engine.current

    assert state.valence == pytest.approx(snapshot["valence"])
    assert current.valence == pytest.approx(snapshot["valence"])
    assert status["valence"] == pytest.approx(snapshot["valence"], abs=0.01)
    assert state.arousal == pytest.approx(snapshot["arousal"])
    assert current.arousal == pytest.approx(snapshot["arousal"])


def test_raw_state_is_immutable() -> None:
    engine = AffectEngineV2()
    raw = engine._raw_state

    with pytest.raises(TypeError):
        raw["curiosity_metric"] = 90.0


@pytest.mark.asyncio
async def test_affect_never_increases_risk_or_scheduling_priority() -> None:
    engine = AffectEngineV2()
    engine.markers.emotions.update({key: 1.0 for key in engine.markers.emotions})

    modifiers = await engine.get_behavioral_modifiers()

    assert 0.2 <= modifiers["risk_tolerance"] <= 1.0
    assert modifiers["verification_pressure"] >= 1.0
    assert await engine.get_metabolic_boost() == 1.0


def test_qualia_echo_is_diagnostic_not_circular_feedback() -> None:
    engine = AffectEngineV2()
    engine.markers.emotions["joy"] = 0.6
    before = dict(engine.markers.emotions)

    receipt = engine.receive_qualia_echo(q_norm=999, pri=float("nan"), trend=-999)

    assert engine.markers.emotions == before
    assert receipt["effect"] == "diagnostic_only_no_affect_amplification"
    observation = engine.get_snapshot()["qualia_observation"]
    assert observation["q_norm"] == 1.0
    assert observation["pri"] == 0.0
    assert observation["trend"] == -1.0


def test_distress_detector_preserves_evidence() -> None:
    engine = AffectEngineV2()
    engine.markers.emotions["fear"] = 1.0
    engine.markers.emotions["sadness"] = 1.0
    before = dict(engine.markers.emotions)

    result = engine._check_for_despair_spiral()

    assert result["detected"] is True
    assert engine.markers.emotions == before


@pytest.mark.asyncio
async def test_llm_appraisal_rejects_noncanonical_output(monkeypatch) -> None:
    engine = AffectEngineV2()

    class Gate:
        async def generate(self, *_args, **_kwargs):
            return '```json\n{"v": 0.5, "a": 0.5, "e": 0.5}\n```'

    monkeypatch.setattr("core.container.ServiceContainer.get", lambda name, default=None: Gate())

    with pytest.raises(ValueError, match="parse_failure"):
        await engine._appraise_with_llm("event", {"evidence": {"kind": "test"}})


@pytest.mark.parametrize(
    "payload",
    [
        {"v": float("nan"), "a": 0.5, "e": 0.5},
        {"v": 0.0, "a": 2.0, "e": 0.5},
        {"v": 0.0, "a": 0.5, "e": 0.5, "extra": 1},
    ],
)
def test_appraisal_schema_rejects_nonfinite_out_of_range_and_extra_fields(payload) -> None:
    with pytest.raises(ValueError):
        AffectEngineV2._validate_appraisal(payload)


def test_background_appraisal_admission_fails_closed(monkeypatch) -> None:
    def unavailable(*_args, **_kwargs):
        raise RuntimeError("control plane unavailable")

    monkeypatch.setattr("core.container.ServiceContainer.get", unavailable)
    assert AffectEngineV2._background_llm_should_defer() is True


@pytest.mark.asyncio
async def test_stop_cancels_owned_background_tasks() -> None:
    engine = AffectEngineV2()
    started = asyncio.Event()

    async def pending() -> None:
        started.set()
        await asyncio.Event().wait()

    task = engine._spawn_background_task(pending(), name="affect.contract.pending")
    await started.wait()
    receipt = engine.stop()
    await asyncio.sleep(0)

    assert receipt["cancelled_tasks"] == 1
    assert task.cancelled()
    assert engine.is_ready() is False
