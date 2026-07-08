"""The cortex model path is a RUNTIME decision, not an import-time constant.

Regression contract for the promotion seam: when the compounding loop
publishes training/fused-model/active.json, every consumer of
get_model_path — most importantly a worker respawn after a stall kill —
must resolve the NEW artifact without a process restart. The old
import-time constant silently resurrected the previous generation's
weights on respawn.
"""
from __future__ import annotations

import json

import pytest

import core.brain.llm.model_registry as model_registry

pytestmark = pytest.mark.unit

CORTEX = model_registry._CORTEX_NAME


@pytest.fixture
def registry_sandbox(tmp_path, monkeypatch):
    monkeypatch.delenv("AURA_LLM__MLX_MODEL_PATH", raising=False)
    monkeypatch.setattr(model_registry, "BASE_DIR", tmp_path)
    monkeypatch.setattr(model_registry, "_cortex_path_cache", None)
    manifest_dir = tmp_path / "training" / "fused-model"
    manifest_dir.mkdir(parents=True)
    yield tmp_path, manifest_dir / "active.json"
    monkeypatch.setattr(model_registry, "_cortex_path_cache", None)


def _publish(manifest_path, model_dir) -> None:
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "model.safetensors").write_bytes(b"weights")
    manifest_path.write_text(
        json.dumps({"active_model_path": str(model_dir)}), encoding="utf-8"
    )


def test_promotion_is_visible_without_restart(registry_sandbox, monkeypatch):
    tmp_path, manifest = registry_sandbox

    before = model_registry.get_model_path(CORTEX)
    assert "gen1" not in before

    _publish(manifest, tmp_path / "fused" / "gen1")
    monkeypatch.setattr(model_registry, "_cortex_path_cache", None)

    after = model_registry.get_model_path(CORTEX)
    assert after.endswith("gen1")


def test_next_generation_supersedes_previous(registry_sandbox, monkeypatch):
    tmp_path, manifest = registry_sandbox
    _publish(manifest, tmp_path / "fused" / "gen1")
    monkeypatch.setattr(model_registry, "_cortex_path_cache", None)
    assert model_registry.get_model_path(CORTEX).endswith("gen1")

    _publish(manifest, tmp_path / "fused" / "gen2")
    monkeypatch.setattr(model_registry, "_cortex_path_cache", None)
    assert model_registry.get_model_path(CORTEX).endswith("gen2")


def test_operator_env_pin_beats_manifest(registry_sandbox, monkeypatch):
    tmp_path, manifest = registry_sandbox
    _publish(manifest, tmp_path / "fused" / "gen1")
    pinned = tmp_path / "pinned-build"
    pinned.mkdir()
    (pinned / "model.safetensors").write_bytes(b"pinned")
    monkeypatch.setenv("AURA_LLM__MLX_MODEL_PATH", str(pinned))
    monkeypatch.setattr(model_registry, "_cortex_path_cache", None)

    assert model_registry.get_model_path(CORTEX).endswith("pinned-build")


def test_ttl_cache_serves_then_expires(registry_sandbox, monkeypatch):
    tmp_path, manifest = registry_sandbox
    _publish(manifest, tmp_path / "fused" / "gen1")
    monkeypatch.setattr(model_registry, "_cortex_path_cache", None)
    assert model_registry.get_model_path(CORTEX).endswith("gen1")

    # Within TTL the cached resolution serves (spawn paths stay cheap) ...
    _publish(manifest, tmp_path / "fused" / "gen2")
    assert model_registry.get_model_path(CORTEX).endswith("gen1")

    # ... and an expired cache picks up the promotion with no restart.
    stamp, path = model_registry._cortex_path_cache
    monkeypatch.setattr(
        model_registry,
        "_cortex_path_cache",
        (stamp - model_registry._CORTEX_MANIFEST_TTL_S - 1.0, path),
    )
    assert model_registry.get_model_path(CORTEX).endswith("gen2")


def test_missing_everything_falls_back_to_hf_repo(registry_sandbox):
    _tmp_path, _manifest = registry_sandbox
    resolved = model_registry.get_model_path(CORTEX)
    assert resolved == "mlx-community/Qwen2.5-32B-Instruct-8bit"
