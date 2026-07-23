"""Contract tests: the learned-halting head is ATTACHABLE, not just present.

CP234 put the seam in HaltingController; this closes the loop the wiring
handoff demanded: a config knob loads a TRAINED head from disk, attaches it
to every branch, and the episode receipt answers the honest question —
did the head decide anything, or did every stop come from the residual
floor? A learned run that never fires is the old policy under a new name,
and the receipt must say so.
"""
from __future__ import annotations

import hashlib

import pytest

mx = pytest.importorskip("mlx.core")
pytest.importorskip("mlx_lm")

from mlx_lm.models.qwen2 import Model, ModelArgs  # noqa: E402

from core.brain.llm.latent_cortex.engine import LatentCortexEngine  # noqa: E402
from core.brain.llm.latent_cortex.types import (  # noqa: E402
    BranchConfig,
    ComputeBudget,
    CortexConfig,
    LatentOptConfig,
    RecurrenceConfig,
    WorkspaceConfig,
)
from core.learning.adaptive_halting import HaltingHead  # noqa: E402
from core.learning.stop_policy import (  # noqa: E402
    STOP_FEATURE_NAMES,
    VerifiedStopExample,
    fit_stop_policy_head,
)

N_LAYERS = 8
HIDDEN = 64
PROMPT_TOKENS = [5, 9, 17, 3, 42, 7, 11, 23, 2, 88]


@pytest.fixture(scope="module")
def tiny_model():
    args = ModelArgs(
        model_type="qwen2", hidden_size=HIDDEN, num_hidden_layers=N_LAYERS,
        intermediate_size=128, num_attention_heads=4, rms_norm_eps=1e-6,
        vocab_size=128, num_key_value_heads=2, max_position_embeddings=512,
        rope_theta=10000.0,
    )
    model = Model(args)
    mx.eval(model.parameters())
    return model


def _config(**overrides) -> CortexConfig:
    return CortexConfig(
        workspace=WorkspaceConfig(n_slots=4, seed=7),
        recurrence=overrides.pop(
            "recurrence", RecurrenceConfig(max_steps=6, min_steps=1)
        ),
        branches=BranchConfig(n_branches=1),
        latent_opt=LatentOptConfig(enabled=False),
        decode_max_tokens=4,
        **overrides,
    )


def _stop_features(should_stop: bool):
    values = {name: 0.0 for name in STOP_FEATURE_NAMES}
    values.update(
        {
            "step_fraction": 0.7 if should_stop else 0.2,
            "residual": 0.04 if should_stop else 0.4,
            "residual_contraction_ratio": 0.3 if should_stop else 0.9,
            "quality_probability": 0.95 if should_stop else 0.55,
            "quality_uncertainty": 0.1 if should_stop else 0.9,
            "evidence_improvement": 0.2 if should_stop else 0.01,
            "policy_uncertainty": 0.1 if should_stop else 0.8,
            "expected_gain_lcb": 0.01 if should_stop else 0.5,
            "expected_cost_ucb": 0.2 if should_stop else 0.1,
            "expected_net_value": -0.19 if should_stop else 0.4,
            "budget_remaining_fraction": 0.5,
            "proposal_accepted": 1.0,
            "quality_measured": 1.0,
            "evoc_measured": 1.0,
        }
    )
    return values


def _fitted_stop_head():
    def rows(prefix, count):
        return [
            VerifiedStopExample.from_values(
                example_id=f"{prefix}-{index}",
                task_id=f"{prefix}-task-{index}",
                features=_stop_features(index % 2 == 0),
                should_stop=index % 2 == 0,
                verifier_receipt_sha256=hashlib.sha256(
                    f"{prefix}:{index}".encode()
                ).hexdigest(),
            )
            for index in range(count)
        ]

    return fit_stop_policy_head(rows("train", 64), rows("cal", 40))


# ── Head persistence ────────────────────────────────────────────────────


def test_head_save_load_roundtrip(tmp_path):
    head = HaltingHead(HIDDEN, threshold=0.7)
    head.weight = mx.ones((HIDDEN, 1)) * 0.03
    head.bias = mx.array([0.25])
    path = tmp_path / "halting_head.npz"
    head.save(path)
    loaded = HaltingHead.load(path)
    assert loaded.hidden_size == HIDDEN
    assert loaded.threshold == pytest.approx(0.7)
    assert bool(mx.allclose(loaded.weight, head.weight))
    assert bool(mx.allclose(loaded.bias, head.bias))
    assert loaded.is_identity() is False


def test_malformed_head_file_is_refused(tmp_path):
    import numpy as np

    path = tmp_path / "broken.npz"
    np.savez(path, weight=np.zeros((4, 1), dtype=np.float32))
    with pytest.raises(ValueError, match="missing fields"):
        HaltingHead.load(path)


# ── Config validation ───────────────────────────────────────────────────


def test_learned_mode_requires_a_head_path():
    problems = _config(halting={"mode": "learned"}).validate()
    assert any("head_path" in problem for problem in problems)
    assert any("head_sha256" in problem for problem in problems)
    unknown = _config(halting={"mode": "residual", "warp": 1}).validate()
    assert any("unknown keys" in problem for problem in unknown)
    fine = _config(halting={"mode": "residual"}).validate()
    assert fine == []


# ── Engine attachment ───────────────────────────────────────────────────


def test_residual_default_attaches_nothing(tiny_model):
    engine = LatentCortexEngine(tiny_model, config=_config())
    result = engine.reason(token_ids=PROMPT_TOKENS, budget=ComputeBudget())
    halting = result.receipt.halting
    assert halting["mode"] == "residual"
    assert halting["head_sha256"] == ""
    assert halting["decision_count"] == 0
    assert halting["head_was_causal"] is False
    assert "halting" in result.receipt.to_dict()


def test_calibrated_pinned_head_attaches_to_live_engine(tiny_model, tmp_path):
    head = _fitted_stop_head()
    path = tmp_path / "stop-head.json"
    digest = head.save(path)
    engine = LatentCortexEngine(
        tiny_model,
        config=_config(
            halting={
                "mode": "learned",
                "head_path": str(path),
                "head_sha256": digest,
            },
        ),
    )
    gate = engine._resolve_halting_head()
    assert gate.mode == "learned"
    assert gate.head_sha256 == digest
    assert gate.manifest == head.manifest()


def test_learned_head_with_unmeasured_runtime_evidence_stays_noncausal(
    tiny_model, tmp_path
):
    head = _fitted_stop_head()
    path = tmp_path / "stop-head.json"
    digest = head.save(path)
    engine = LatentCortexEngine(
        tiny_model,
        config=_config(
            halting={
                "mode": "learned",
                "head_path": str(path),
                "head_sha256": digest,
            }
        ),
    )
    result = engine.reason(token_ids=PROMPT_TOKENS, budget=ComputeBudget())
    halting = result.receipt.halting
    assert halting["mode"] == "learned"
    assert halting["head_was_causal"] is False
    assert "learned_halting_not_causal" in result.receipt.honest_flags


def test_missing_head_refuses_rather_than_silently_reverting(tiny_model):
    engine = LatentCortexEngine(
        tiny_model,
        config=_config(
            halting={
                "mode": "learned",
                "head_path": "/nonexistent/head.json",
                "head_sha256": "a" * 64,
            }
        ),
    )
    with pytest.raises(ValueError, match="unreadable"):
        engine._resolve_halting_head()


def test_runtime_threshold_override_is_rejected():
    problems = _config(
        halting={
            "mode": "learned",
            "head_path": "head.json",
            "head_sha256": "a" * 64,
            "threshold": 0.9,
        }
    ).validate()
    assert any("unknown keys" in problem for problem in problems)
