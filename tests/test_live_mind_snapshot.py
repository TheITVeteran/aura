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
    def gather(self) -> "AffectGroundingReadout":
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
    }

    @classmethod
    def get(cls, name: str, default=None):
        return cls.services.get(name, default)


def test_live_mind_snapshot_collects_deep_runtime_state(monkeypatch):
    from core.runtime import live_mind_snapshot

    monkeypatch.setattr(live_mind_snapshot, "ServiceContainer", RuntimeServices)
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
    assert snapshot["screen_perception"]["last_hash"] == "screen-hash"
    assert snapshot["perceptual_pump"]["latest_frame"]["active_app"] == "Notes"
    assert snapshot["frontmost_app_fast"] == "Notes"


def test_live_desktop_context_payload_carries_mind_snapshot(monkeypatch):
    from core.runtime import live_mind_snapshot
    from interface.routes import chat as chat_routes

    monkeypatch.setattr(live_mind_snapshot, "ServiceContainer", RuntimeServices)
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
