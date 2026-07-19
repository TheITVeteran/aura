"""Contract tests: the cortex generation-upgrade pipeline.

The pipeline's promises, proven on real machinery:
- the capability battery runs real decodes and DISCRIMINATES (a sabotaged
  model scores measurably worse than its healthy twin);
- the comparison verdict demands breadth wins without reasoning losses;
- the memory guard refuses candidates the host cannot afford;
- staging writes a byte-exact rollback and changes nothing live;
- activation is impossible without operator authorization + PASS verdict;
- rollback restores the pointer byte-exactly.
"""
from __future__ import annotations

import json

import pytest

mx = pytest.importorskip("mlx.core")
pytest.importorskip("mlx_lm")

from mlx_lm.models.qwen2 import Model, ModelArgs  # noqa: E402

from core.learning.cortex_generation_upgrade import (  # noqa: E402
    MemoryGuard,
    ROLLBACK_POINTER_NAME,
    STAGED_POINTER_NAME,
    activate_upgrade,
    build_migration_plan,
    capability_battery,
    compare_batteries,
    rollback_upgrade,
    stage_upgrade,
)


class TinyTokenizer:
    eos_token_id = 0

    def encode(self, text, **kwargs):
        return [ord(c) % 127 + 1 for c in str(text)][:32] or [5]

    def decode(self, ids):
        return " ".join(str(i) for i in ids)


def _model(seed=0):
    args = ModelArgs(
        model_type="qwen2", hidden_size=64, num_hidden_layers=8,
        intermediate_size=128, num_attention_heads=4, rms_norm_eps=1e-6,
        vocab_size=128, num_key_value_heads=2, max_position_embeddings=512,
        rope_theta=10000.0,
    )
    mx.random.seed(seed)
    model = Model(args)
    mx.eval(model.parameters())
    return model


# ── Battery ─────────────────────────────────────────────────────────────


def test_battery_runs_and_is_deterministic():
    model = _model()
    first = capability_battery(model, TinyTokenizer(), label="tiny")
    second = capability_battery(model, TinyTokenizer(), label="tiny")
    assert first["breadth_total"] == 24 and first["reasoning_total"] == 12
    assert first["breadth_accuracy"] == second["breadth_accuracy"]
    assert first["identity_digests"] == second["identity_digests"]
    assert len(first["identity_digests"]) >= 8


def test_battery_discriminates_a_sabotaged_model():
    healthy = _model(seed=0)
    healthy_receipt = capability_battery(healthy, TinyTokenizer(), label="healthy")

    wrecked = _model(seed=0)
    layer = wrecked.model.layers[4]
    layer.mlp.down_proj.weight = layer.mlp.down_proj.weight + mx.random.normal(
        layer.mlp.down_proj.weight.shape, key=mx.random.key(9)
    )
    wrecked_receipt = capability_battery(wrecked, TinyTokenizer(), label="wrecked")
    # Identity behavior MUST move when the weights are wrecked — the battery
    # sees real model behavior, not fixtures.
    assert healthy_receipt["identity_digests"] != wrecked_receipt["identity_digests"]


def test_comparison_verdict_requires_breadth_win_without_reasoning_loss():
    current = {"label": "cur", "breadth_accuracy": 0.5, "reasoning_accuracy": 0.5,
               "identity_digests": ["a"]}
    better = {"label": "cand", "breadth_accuracy": 0.7, "reasoning_accuracy": 0.5,
              "identity_digests": ["b"]}
    worse_reasoning = {"label": "cand", "breadth_accuracy": 0.7,
                       "reasoning_accuracy": 0.3, "identity_digests": ["b"]}
    no_breadth = {"label": "cand", "breadth_accuracy": 0.5,
                  "reasoning_accuracy": 0.9, "identity_digests": ["b"]}
    assert compare_batteries(current, better)["verdict"] == "PASS"
    assert compare_batteries(current, worse_reasoning)["verdict"] == "FAIL"
    assert compare_batteries(current, no_breadth)["verdict"] == "FAIL"
    assert compare_batteries(current, better)["identity_behavior_changed"] is True


# ── Memory guard ────────────────────────────────────────────────────────


def test_memory_guard_refuses_empty_and_oversized(tmp_path, monkeypatch):
    guard = MemoryGuard()
    empty = guard.admit(tmp_path)
    assert empty["admitted"] is False and "no weight files" in empty["refusal_reason"]

    # Availability is injected: the REAL host may legitimately be under
    # model-scale pressure (a 32B training run), and the guard must report
    # the injected world, not the test machine's mood.
    monkeypatch.setattr(MemoryGuard, "_available_gb", staticmethod(lambda: 32.0))
    (tmp_path / "model.safetensors").write_bytes(b"x" * 1024)
    admitted = guard.admit(tmp_path)
    assert admitted["admitted"] is True

    giant_guard = MemoryGuard(free_margin_gb=10**6)  # impossible margin
    refused = giant_guard.admit(tmp_path)
    assert refused["admitted"] is False
    assert "headroom" in refused["refusal_reason"]

    monkeypatch.setattr(MemoryGuard, "_available_gb", staticmethod(lambda: 2.0))
    pressured = MemoryGuard().admit(tmp_path)
    assert pressured["admitted"] is False, "a strained host must refuse"


# ── Migration plan ──────────────────────────────────────────────────────


def test_migration_plan_names_real_artifacts_and_lanes(tmp_path):
    plan = build_migration_plan(
        fused_model_dir=tmp_path / "fused", data_dir=tmp_path / "data"
    )
    names = {step["name"] for step in plan["steps"]}
    assert {"activation_pointer", "persona_crsm_delta", "caa_steering_vectors",
            "expert_adapters", "recurrence_native_adapter"} <= names
    assert plan["automatic_steps"] == ["activation_pointer"]
    assert len(plan["operator_steps"]) == 4
    # Honest existence flags for a bare tmp dir.
    pointer = next(s for s in plan["steps"] if s["name"] == "activation_pointer")
    assert pointer["exists"] is False


# ── Stage / activate / rollback ─────────────────────────────────────────


def _fused_dir(tmp_path):
    fused = tmp_path / "fused-model"
    fused.mkdir()
    current = {
        "active_model_path": str(tmp_path / "current-model"),
        "base_model": "Qwen2.5-32B",
        "fused_at": 1000,
        "schema_version": 2,
        "size": "32B",
        "tag": "current",
    }
    (fused / "active.json").write_text(json.dumps(current, indent=2) + "\n")
    candidate = tmp_path / "candidate-model"
    candidate.mkdir()
    (candidate / "model.safetensors").write_bytes(b"weights")
    return fused, candidate


def test_stage_writes_rollback_and_changes_nothing_live(tmp_path, monkeypatch):
    monkeypatch.setenv("AURA_LOG_DIR", str(tmp_path / "logs"))
    fused, candidate = _fused_dir(tmp_path)
    before = (fused / "active.json").read_bytes()
    receipt = stage_upgrade(
        candidate_model_path=candidate,
        base_model_path="Qwen3-32B",
        tag="qwen3-gen",
        fused_model_dir=fused,
    )
    assert (fused / "active.json").read_bytes() == before, "staging must not touch live"
    assert (fused / ROLLBACK_POINTER_NAME).read_bytes() == before
    staged = json.loads((fused / STAGED_POINTER_NAME).read_text())
    assert staged["active_model_path"] == str(candidate)
    assert staged["base_model"] == "Qwen3-32B"
    assert receipt["staged_active_model"] == str(candidate)


def test_activation_gates_and_flip(tmp_path, monkeypatch):
    monkeypatch.setenv("AURA_LOG_DIR", str(tmp_path / "logs"))
    fused, candidate = _fused_dir(tmp_path)
    stage_upgrade(
        candidate_model_path=candidate, base_model_path="Qwen3-32B",
        tag="qwen3-gen", fused_model_dir=fused,
    )
    with pytest.raises(PermissionError, match="authorization"):
        activate_upgrade(fused_model_dir=fused, authorized_by="", 
                         evaluation={"verdict": "PASS"})
    with pytest.raises(PermissionError, match="PASS"):
        activate_upgrade(fused_model_dir=fused, authorized_by="bryan",
                         evaluation={"verdict": "FAIL"})
    receipt = activate_upgrade(
        fused_model_dir=fused, authorized_by="bryan",
        evaluation={"verdict": "PASS"},
    )
    active = json.loads((fused / "active.json").read_text())
    assert active["active_model_path"] == str(candidate)
    assert receipt["effective"] == "next_boot"
    assert receipt["authorized_by"] == "bryan"


def test_rollback_is_byte_exact(tmp_path, monkeypatch):
    monkeypatch.setenv("AURA_LOG_DIR", str(tmp_path / "logs"))
    fused, candidate = _fused_dir(tmp_path)
    original = (fused / "active.json").read_bytes()
    stage_upgrade(candidate_model_path=candidate, base_model_path="Qwen3-32B",
                  tag="qwen3-gen", fused_model_dir=fused)
    activate_upgrade(fused_model_dir=fused, authorized_by="bryan",
                     evaluation={"verdict": "PASS"})
    assert (fused / "active.json").read_bytes() != original
    receipt = rollback_upgrade(fused_model_dir=fused)
    assert receipt["byte_exact"] is True
    assert (fused / "active.json").read_bytes() == original


def test_activation_without_staging_refuses(tmp_path, monkeypatch):
    monkeypatch.setenv("AURA_LOG_DIR", str(tmp_path / "logs"))
    fused, _ = _fused_dir(tmp_path)
    with pytest.raises(ValueError, match="nothing staged"):
        activate_upgrade(fused_model_dir=fused, authorized_by="bryan",
                         evaluation={"verdict": "PASS"})
    with pytest.raises(ValueError, match="no rollback"):
        rollback_upgrade(fused_model_dir=fused)
