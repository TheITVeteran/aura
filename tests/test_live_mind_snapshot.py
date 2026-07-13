from __future__ import annotations

from types import SimpleNamespace


class GroundedAffectReadout:
    def __init__(self, label: str, intensity: float) -> None:
        self.label = label
        self.intensity = intensity

    def to_dict(self) -> dict[str, object]:
        return {"label": self.label, "intensity": self.intensity}


class GlobalWorkspaceReadout:
    def get_snapshot(self) -> dict[str, object]:
        return {"tick": 7, "last_winner": "curiosity", "ignited": True, "phi": 0.42}


class NociceptionReadout:
    def snapshot(self) -> dict[str, object]:
        return {"nociceptive_pressure": 0.12, "grounded_valence": 0.18}


class AffectGroundingReadout:
    def gather(self) -> AffectGroundingReadout:
        return self

    def assess(self) -> list[GroundedAffectReadout]:
        return [GroundedAffectReadout("curiosity", 0.68)]

    def dominant(self) -> GroundedAffectReadout:
        return GroundedAffectReadout("curiosity", 0.68)


class DriveIntegrationReadout:
    def state(self) -> dict[str, object]:
        return {"drives": {"curiosity": {"activation": 0.71, "suppressed": False}}}


class OutcomeLedgerReadout:
    def stats(self) -> dict[str, object]:
        return {"pending": 2, "resolved_count": 5, "expectation_calibration": 0.13}


class ScientificEngineReadout:
    def stats(self) -> dict[str, object]:
        return {"total": 3, "by_status": {"supported": 1, "testing": 2}}


class UnifiedWorldModelReadout:
    def status(self) -> dict[str, object]:
        return {"module": "UnifiedWorldModel", "facets": {"learned": {"available": True}}}


class PhenomenalEngineReadout:
    last_state = SimpleNamespace(
        t=11,
        valence=0.31,
        arousal=0.24,
        free_energy=0.19,
        integration=0.73,
        self_presence=0.81,
        mineness=0.77,
        curiosity=0.66,
        intentional_object="desktop conversation",
        policy_priors={"answer_directly": 0.8},
        memory_weights={"recent_user_context": 0.9},
    )


class ScreenPerceptionReadout:
    def get_status(self) -> dict[str, object]:
        return {"captures": 4, "last_hash": "screen-hash"}


class PerceptualPumpReadout:
    def get_status(self) -> dict[str, object]:
        return {
            "running": True,
            "frames_produced": 12,
            "latest_frame": {"active_app": "Notes", "window_title": "Aura note"},
        }


class LiquidSubstrateReadout:
    def get_substrate_affect(self) -> dict[str, object]:
        return {"valence": 0.21, "arousal": 0.18}

    def get_status(self) -> dict[str, object]:
        return {"coherence": 0.91, "thermal_pressure": "nominal"}


class PhenomenalKnowingReadout:
    def snapshot(self) -> dict[str, object]:
        return {
            "active": True,
            "latest": {"event_kind": "desktop_context", "control_updates": {"phenomenal_knowing": 0.72}},
        }


class RecursiveSelfKnowingReadout:
    def snapshot(self) -> dict[str, object]:
        return {
            "active": True,
            "latest": {"calibration_delta": 0.08, "confidence": 0.79},
        }


class AutomaticSelfKnowingReadout:
    def snapshot(self) -> dict[str, object]:
        return {
            "active": True,
            "latest": {
                "event_kind": "runtime_tick",
                "controls": {"automatic_self_knowing_active": True},
            },
        }


class RuntimeServices:
    services = {
        "global_workspace": GlobalWorkspaceReadout(),
        "nociception": NociceptionReadout(),
        "affect_grounding": AffectGroundingReadout(),
        "drive_integration": DriveIntegrationReadout(),
        "outcome_ledger": OutcomeLedgerReadout(),
        "scientific_engine": ScientificEngineReadout(),
        "unified_world_model": UnifiedWorldModelReadout(),
        "phenomenal_engine": PhenomenalEngineReadout(),
        "screen_perception": ScreenPerceptionReadout(),
        "perceptual_pump": PerceptualPumpReadout(),
        "liquid_substrate": LiquidSubstrateReadout(),
        "liquid_state": LiquidSubstrateReadout(),
        "phenomenal_knowing": PhenomenalKnowingReadout(),
        "recursive_self_knowing": RecursiveSelfKnowingReadout(),
        "automatic_self_knowing": AutomaticSelfKnowingReadout(),
    }

    @classmethod
    def get(cls, name: str, default=None):
        return cls.services.get(name, default)


def test_live_mind_snapshot_collects_deep_runtime_state(monkeypatch):
    from core.runtime import live_mind_snapshot

    monkeypatch.setattr(
        live_mind_snapshot, "get_runtime_service", RuntimeServices.get
    )
    monkeypatch.setattr(live_mind_snapshot, "_frontmost_app_fast", lambda: "Notes")

    snapshot = live_mind_snapshot.collect_live_mind_snapshot(
        lane={"desired_model": "Cortex (32B)", "conversation_ready": True}
    )

    assert snapshot["schema"] == "aura.live_mind_snapshot.v1"
    assert all(snapshot["services_present"].values())
    assert snapshot["global_workspace"]["last_winner"] == "curiosity"
    assert snapshot["nociception"]["nociceptive_pressure"] == 0.12
    assert snapshot["affect_grounding"]["dominant"]["label"] == "curiosity"
    assert snapshot["drive_integration"]["drives"]["curiosity"]["activation"] == 0.71
    assert snapshot["outcome_ledger"]["pending"] == 2
    assert snapshot["scientific_engine"]["by_status"]["testing"] == 2
    assert snapshot["world_model"]["facets"]["learned"]["available"] is True
    assert snapshot["phenomenal_engine"]["intentional_object"] == "desktop conversation"
    assert snapshot["phenomenal_knowing"]["latest"]["event_kind"] == "desktop_context"
    assert snapshot["recursive_self_knowing"]["latest"]["confidence"] == 0.79
    assert snapshot["automatic_self_knowing"]["latest"]["controls"]["automatic_self_knowing_active"] is True
    assert snapshot["screen_perception"]["last_hash"] == "screen-hash"
    assert snapshot["perceptual_pump"]["latest_frame"]["active_app"] == "Notes"
    assert snapshot["frontmost_app_fast"] == "Notes"


def test_live_mind_runtime_materializes_registered_organs_before_snapshot(monkeypatch):
    from core.runtime import live_mind_runtime, live_mind_snapshot

    class LazyContainer:
        services = dict(RuntimeServices.services)
        materialized: dict[str, object] = {}

        @classmethod
        def get(cls, name: str, default=None):
            value = cls.services.get(name, default)
            if value is not default:
                cls.materialized[name] = value
            return value

    monkeypatch.setattr(
        live_mind_snapshot,
        "get_runtime_service",
        lambda name, default=None: LazyContainer.materialized.get(name, default),
    )
    monkeypatch.setattr(
        live_mind_runtime,
        "get_runtime_service",
        lambda name, default=None: LazyContainer.materialized.get(name, default),
    )
    monkeypatch.setattr(live_mind_snapshot, "_frontmost_app_fast", lambda: "Notes")

    runtime = live_mind_runtime.LiveMindRuntime()
    before = runtime.get_status()
    report = runtime.materialize(LazyContainer)

    assert before["ready"] is False
    assert report["ready"] is True
    assert report["missing_services"] == []
    assert set(live_mind_snapshot.REQUIRED_LIVE_MIND_SERVICES).issubset(
        LazyContainer.materialized
    )


def test_live_mind_runtime_refuses_partial_activation(monkeypatch):
    from core.runtime import live_mind_runtime, live_mind_snapshot

    class PartialContainer:
        @staticmethod
        def get(name: str, default=None):
            if name == "scientific_engine":
                return default
            return RuntimeServices.services.get(name, default)

    resolved: dict[str, object] = {}

    def resolve(name: str, default=None):
        return resolved.get(name, default)

    original_get = PartialContainer.get

    def recording_get(name: str, default=None):
        value = original_get(name, default)
        if value is not default:
            resolved[name] = value
        return value

    monkeypatch.setattr(PartialContainer, "get", recording_get)
    monkeypatch.setattr(live_mind_snapshot, "get_runtime_service", resolve)
    monkeypatch.setattr(live_mind_runtime, "get_runtime_service", resolve)
    monkeypatch.setattr(live_mind_snapshot, "_frontmost_app_fast", lambda: "Notes")

    report = live_mind_runtime.LiveMindRuntime().materialize(PartialContainer)

    assert report["ready"] is False
    assert report["missing_services"] == ["scientific_engine"]
    assert report["activation_errors"]["scientific_engine"] == "registered service resolved to None"


def test_live_mind_snapshot_requires_readout_from_every_required_organ():
    from core.runtime.live_mind_snapshot import (
        REQUIRED_LIVE_MIND_SECTIONS,
        assess_live_mind_snapshot,
    )

    snapshot = {
        "services_present": {
            service_name: True for service_name in REQUIRED_LIVE_MIND_SECTIONS
        },
        **{
            section_name: {"available": True}
            for section_name in REQUIRED_LIVE_MIND_SECTIONS.values()
        },
    }
    snapshot["scientific_engine"] = None

    quality = assess_live_mind_snapshot(snapshot)

    assert quality["ready"] is False
    assert quality["missing_services"] == []
    assert quality["unpopulated_services"] == ["scientific_engine"]
    assert quality["unpopulated_sections"] == ["scientific_engine"]


def test_live_desktop_context_payload_carries_mind_snapshot(monkeypatch):
    from core.runtime import live_mind_snapshot
    from interface.routes import chat as chat_routes

    monkeypatch.setattr(
        live_mind_snapshot, "get_runtime_service", RuntimeServices.get
    )
    monkeypatch.setattr(chat_routes, "ServiceContainer", RuntimeServices)
    monkeypatch.setattr(
        chat_routes,
        "_resolve_live_voice_state",
        lambda *args, **kwargs: {
            "mood": "engaged",
            "dominant_action": "answer",
            "substrate_snapshot": {"coherence": 0.9},
            "voice_profile": {"surface": "desktop"},
        },
    )
    for name in (
        "_runtime_kernel_available",
        "_runtime_cognitive_engine_available",
        "_runtime_memory_available",
        "_runtime_tool_governance_available",
        "_runtime_substrate_voice_available",
    ):
        monkeypatch.setattr(chat_routes, name, lambda: True)
    monkeypatch.setattr(chat_routes, "_runtime_inference_available", lambda *args, **kwargs: True)

    payload = chat_routes._build_live_mind_context_payload(
        user_message="What are you attending to?",
        lane={"desired_model": "Cortex (32B)", "conversation_ready": True},
        recent_conversation_context="Bryan asked about Aura's live path.",
        recent_context_needed=True,
        require_engine=True,
    )

    assert payload["must_answer_from_full_mind_path"] is True
    assert payload["required_subsystems_ok"] is True
    assert payload["mind_snapshot_quality"]["ready"] is True
    assert payload["voice"]["mood"] == "engaged"
    assert payload["substrate"]["affect"]["valence"] == 0.21
    assert payload["mind_snapshot"]["global_workspace"]["last_winner"] == "curiosity"
    assert payload["mind_snapshot"]["phenomenal_engine"]["self_presence"] == 0.81


def test_live_desktop_context_payload_carries_recent_voice_perception(monkeypatch):
    from core.runtime import live_mind_snapshot
    from interface.routes import chat as chat_routes

    class VoiceWorldState:
        last_voice_transcript = "Bryan said the sentence about blue glass."
        last_voice_transcript_at = 1000.0
        voice_activity_detected = True
        last_audio_source_assessment = {
            "source": "near-field_voice",
            "response_authorized": False,
            "attention_mode": "listen",
        }

    RuntimeServices.services["world_state"] = VoiceWorldState()
    try:
        monkeypatch.setattr(
        live_mind_snapshot, "get_runtime_service", RuntimeServices.get
    )
        monkeypatch.setattr(chat_routes, "ServiceContainer", RuntimeServices)
        monkeypatch.setattr(chat_routes.time, "time", lambda: 1030.0)
        monkeypatch.setattr(chat_routes, "_resolve_live_voice_state", lambda *args, **kwargs: {})
        for name in (
            "_runtime_kernel_available",
            "_runtime_cognitive_engine_available",
            "_runtime_memory_available",
            "_runtime_tool_governance_available",
            "_runtime_substrate_voice_available",
        ):
            monkeypatch.setattr(chat_routes, name, lambda: True)
        monkeypatch.setattr(chat_routes, "_runtime_inference_available", lambda *args, **kwargs: True)

        payload = chat_routes._build_live_mind_context_payload(
            user_message="What did I say out loud?",
            lane={"desired_model": "Cortex (32B)", "conversation_ready": True},
            require_engine=True,
        )
    finally:
        RuntimeServices.services.pop("world_state", None)

    perception = payload["voice_perception"]
    assert perception["heard"] is True
    assert perception["recent"] is True
    assert perception["authorized_command"] is False
    assert perception["requires_wake_word_session"] is True
    assert "blue glass" in perception["transcript"]


def test_protected_foreground_prompt_reports_voice_activity_without_transcript(monkeypatch):
    from interface.routes import chat as chat_routes

    class AcousticWorldState:
        last_voice_transcript = ""
        last_voice_transcript_at = 0.0
        voice_activity_detected = True
        last_voice_activity_at = 2000.0
        last_audio_source_assessment = {
            "source": "browser_voice_signal",
            "response_authorized": False,
            "attention_mode": "listen",
            "transcript_available": False,
        }

    RuntimeServices.services["world_state"] = AcousticWorldState()
    try:
        monkeypatch.setattr(chat_routes, "ServiceContainer", RuntimeServices)
        monkeypatch.setattr(chat_routes.time, "time", lambda: 2012.0)
        monkeypatch.setattr(chat_routes, "_resolve_protected_foreground_snapshot", lambda: {})
        monkeypatch.setattr(chat_routes, "_resolve_live_voice_state", lambda *args, **kwargs: {})

        prompt = chat_routes._build_protected_foreground_system_prompt(
            "What did I say out loud?",
            lane={"state": "ready"},
        )
    finally:
        RuntimeServices.services.pop("world_state", None)

    assert "Recent heard speech" in prompt
    assert "recent voice activity detected" in prompt
    assert "no transcript available" in prompt
