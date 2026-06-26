from __future__ import annotations

import pytest


def _ready_live_mind_context(*, curiosity: float = 0.72, distress: float = 0.08) -> dict[str, object]:
    return {
        "mind_snapshot_quality": {"ready": True},
        "mind_snapshot": {
            "services_present": {
                "global_workspace": True,
                "nociception": True,
                "affect_grounding": True,
                "drive_integration": True,
                "outcome_ledger": True,
                "scientific_engine": True,
                "unified_world_model": True,
                "phenomenal_engine": True,
            },
            "global_workspace": {"ignited": True},
            "nociception": {"nociceptive_pressure": distress},
            "affect_grounding": {
                "dominant": {"label": "curiosity", "intensity": curiosity}
            },
            "drive_integration": {
                "drives": {"curiosity": {"activation": curiosity}}
            },
            "outcome_ledger": {"expectation_calibration": distress},
            "phenomenal_engine": {
                "integration": 0.74,
                "self_presence": 0.82,
            },
        },
    }


def test_live_mind_generation_controls_convert_snapshot_to_sampling_controls():
    from core.brain.cognitive_engine import _live_mind_generation_controls

    controls = _live_mind_generation_controls(_ready_live_mind_context())

    assert controls["temperature"] > 0.58
    assert controls["top_p"] <= 0.94
    assert controls["clean_user_surface_recurrent_loops"] == 2
    assert controls["clean_user_surface_steering_alpha"] > 0.25


def test_live_mind_generation_controls_reduce_sampling_under_distress():
    from core.brain.cognitive_engine import _live_mind_generation_controls

    calm = _live_mind_generation_controls(_ready_live_mind_context(curiosity=0.7, distress=0.05))
    pressured = _live_mind_generation_controls(
        _ready_live_mind_context(curiosity=0.2, distress=0.65)
    )

    assert pressured["temperature"] < calm["temperature"]
    assert pressured["top_p"] < calm["top_p"]
    assert pressured["clean_user_surface_recurrent_loops"] == 2


@pytest.mark.asyncio
async def test_desktop_quick_reply_passes_live_mind_controls_to_router(monkeypatch):
    from core.brain import cognitive_engine as cognitive_engine_module
    from core.brain.cognitive_engine import CognitiveEngine
    from core.brain.types import ThinkingMode

    calls: list[dict[str, object]] = []

    class Router:
        async def think(self, **kwargs):
            calls.append(dict(kwargs))
            return "I am tracking the current turn through the live desktop mind path."

        def get_last_generation_metadata(self):
            return {
                "surface_control_receipt": {
                    "enabled": True,
                    "live_mind_controls_bound": True,
                    "clean_user_surface_contract": True,
                    "surface_alpha_applied": 0.30,
                    "surface_alpha_applied_ok": True,
                    "recurrent_runtime_loops_applied": 2,
                    "recurrent_runtime_loops_applied_ok": True,
                    "surface_quality_gate_enabled": True,
                    "surface_quality_gate_passed": True,
                    "surface_quality_gate_attempts": 1,
                    "surface_quality_gate_reasons": [],
                    "applied": True,
                }
            }

    class Container:
        def get(self, name, default=None):
            if name == "llm_router":
                return Router()
            return default

    monkeypatch.setattr(cognitive_engine_module, "get_container", lambda: Container())

    engine = CognitiveEngine()
    thought = await engine._direct_desktop_quick_reply(
        "What are you attending to?",
        ThinkingMode.FAST,
        "user",
        {
            "desktop_quick_reply_contract": True,
            "live_mind_context_required": True,
            "cognitive_engine_required": True,
            "desktop_cognitive_engine_required": True,
            "live_mind_context": _ready_live_mind_context(),
            "visible_user_message": "What are you attending to?",
            "max_tokens": 512,
        },
        timeout_s=30.0,
    )

    assert thought is not None
    assert calls
    router_kwargs = calls[0]
    assert router_kwargs["temperature"] > 0.58
    assert router_kwargs["top_p"] <= 0.94
    assert router_kwargs["clean_user_surface_recurrent_loops"] == 2
    assert router_kwargs["clean_user_surface_steering_alpha"] > 0.25
    assert router_kwargs["clean_user_surface_contract"] is True
    assert router_kwargs["live_mind_controls_bound"] is True
    assert router_kwargs["live_mind_snapshot_ready"] is True
    assert router_kwargs["live_mind_required_subsystems_ok"] is False
    assert router_kwargs["live_mind_generation_controls"]["temperature"] > 0.58
    assert thought.metadata["live_mind_controls_bound"] is True
    assert thought.metadata["live_mind_controls_worker_applied"] is True
    assert thought.metadata["live_mind_snapshot_ready"] is True
    assert thought.metadata["live_mind_generation_controls"]["top_p"] <= 0.94
