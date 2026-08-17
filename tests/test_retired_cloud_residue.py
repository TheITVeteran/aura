"""Mechanical invariants for the retired remote-model surface."""

from __future__ import annotations

import asyncio
from pathlib import Path

from core.adapters import provider_tools
from core.brain import composer_node, pii_scrubber


ROOT = Path(__file__).resolve().parents[1]


def test_dead_cloud_local_conversation_tier_formatter_is_removed():
    assert not (ROOT / "core/guardians/conversational_guard.py").exists()


def test_live_local_modules_make_no_gemini_or_cloud_fallback_claim():
    paths = (
        "core/agency/self_play.py",
        "core/brain/composer_node.py",
        "core/brain/latent_bridge.py",
        "core/adapters/provider_tools.py",
        "core/brain/types.py",
        "core/brain/affect_state.py",
    )
    for relative_path in paths:
        source = (ROOT / relative_path).read_text(encoding="utf-8").lower()
        assert "gemini" not in source, relative_path
        assert "cloud fallback" not in source, relative_path


def test_egress_scrubber_is_provider_neutral_and_legacy_calls_remain_safe(monkeypatch):
    monkeypatch.setattr(pii_scrubber, "_cached_names", [])
    source = "contact me at owner@example.com"

    scrubbed, receipt = pii_scrubber.scrub_for_egress_with_receipt(source)

    assert scrubbed == "contact me at [REDACTED_EMAIL]"
    assert receipt["schema"] == "aura.egress.privacy_receipt.v1"
    assert receipt["safe_to_send"] is True
    assert pii_scrubber.scrub_pii_for_cloud(source) == scrubbed
    assert pii_scrubber.scrub_for_cloud_with_receipt(source) == (scrubbed, receipt)


def test_tool_admission_uses_the_live_capability_registry(monkeypatch):
    events = []
    monkeypatch.setattr(provider_tools, "registered_capability_names", lambda: {"web_search"})
    monkeypatch.setattr(provider_tools, "record_degradation", lambda *args, **kwargs: events.append((args, kwargs)))

    admitted = provider_tools.admissible_tools(
        [{"name": "web_search"}, {"name": "unregistered_action"}]
    )

    assert admitted == [{"name": "web_search"}]
    assert events[0][0][0] == "provider_tools"


def test_composer_receipt_never_claims_a_planned_transform_was_applied(monkeypatch):
    class Vision:
        frame_buffer = [object()]

        async def query_visual_context(self, _request, _engine):
            return "A grounded description"

    node = composer_node.ComposerNode()
    node._is_setup = True
    node.vision_buffer = Vision()
    monkeypatch.setattr(composer_node, "get_runtime_service", lambda *_args, **_kwargs: None)

    result = asyncio.run(node.stylize_desktop("ink wash"))

    assert result["ok"] is True
    assert result["effect_applied"] is False
    assert result["requires_image_transform"] is True
    assert "complete" not in result["message"].lower()
    source = (ROOT / "core/brain/composer_node.py").read_text(encoding="utf-8")
    assert "Style Transfer Enabled" not in source
