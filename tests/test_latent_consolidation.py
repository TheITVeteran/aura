"""Contract tests: consolidation pipeline + anti-interference battery.

End-to-end on real machinery: a tiny-model episode with export enabled
lands a candidate in the queue; the consumer validates evidence and refuses
corrupt/unproven candidates; domains only propose after enough independent
wins; and the interference battery passes an identity change while failing
a genuinely disruptive one — proving accumulated learning is gated on NOT
trashing prior behavior.
"""
from __future__ import annotations

import json

import pytest

from core.learning.latent_consolidation import (
    build_proposals,
    run_consolidation_cycle,
    scan_queue,
    validate_candidate,
)


def _make_candidate(root, episode_id, *, domain="math", erase=True, steps=2,
                    trail=(3.0, 2.0), flags=()):
    d = root / episode_id
    d.mkdir(parents=True)
    (d / "delta_weights.npz").write_bytes(b"npz-bytes")
    (d / "evidence.json").write_text(json.dumps({
        "episode_id": episode_id,
        "created_at": 1000.0,
        "lifecycle": {"erase_proven": erase, "optimized_steps": steps},
        "evidence": {
            "domain": domain,
            "loss_trail": list(trail),
            "honest_flags": list(flags),
        },
    }))
    return d


# ── Candidate validation ────────────────────────────────────────────────


def test_clean_candidate_validates(tmp_path):
    d = _make_candidate(tmp_path, "ep-1")
    record = validate_candidate(d)
    assert record.valid and record.domain == "math"
    assert record.loss_improvement == pytest.approx(1.0)


def test_unproven_erase_is_rejected(tmp_path):
    record = validate_candidate(_make_candidate(tmp_path, "ep-2", erase=False))
    assert not record.valid and "erase_unproven" in record.rejection_reasons


def test_flat_loss_and_no_steps_rejected(tmp_path):
    flat = validate_candidate(_make_candidate(tmp_path, "ep-3", trail=(2.0, 2.0)))
    assert "loss_not_descending" in flat.rejection_reasons
    lazy = validate_candidate(_make_candidate(tmp_path, "ep-4", steps=0))
    assert "no_accepted_optimization" in lazy.rejection_reasons


def test_honest_flags_block_consolidation(tmp_path):
    record = validate_candidate(
        _make_candidate(tmp_path, "ep-5", flags=("fallback_vanilla:RuntimeError",))
    )
    assert not record.valid
    assert any("honest_flags_present" in r for r in record.rejection_reasons)


def test_corrupt_evidence_rejected(tmp_path):
    d = tmp_path / "ep-6"
    d.mkdir()
    (d / "evidence.json").write_text("{not json")
    record = validate_candidate(d)
    assert not record.valid
    assert {"evidence_unreadable", "delta_weights_missing"} <= set(record.rejection_reasons)


# ── Aggregation ─────────────────────────────────────────────────────────


def test_domain_needs_enough_independent_wins(tmp_path):
    for i in range(2):
        _make_candidate(tmp_path, f"math-{i}", domain="math")
    for i in range(3):
        _make_candidate(tmp_path, f"code-{i}", domain="code")
    _make_candidate(tmp_path, "bad-1", domain="code", erase=False)

    records = scan_queue(tmp_path)
    proposals = build_proposals(records, min_candidates=3)
    assert [p["domain"] for p in proposals] == ["code"]
    assert proposals[0]["candidate_count"] == 3  # the invalid one never counts
    assert proposals[0]["mean_loss_improvement"] == pytest.approx(1.0)
    assert "interference battery verdict PASS before activation" in (
        proposals[0]["activation_requirements"]
    )


def test_cycle_writes_proposals_and_reports_rejections(tmp_path, monkeypatch):
    monkeypatch.setenv("AURA_LOG_DIR", str(tmp_path / "logs"))
    queue = tmp_path / "queue"
    for i in range(3):
        _make_candidate(queue, f"ep-{i}", domain="planning")
    _make_candidate(queue, "ep-bad", domain="planning", erase=False)

    receipt = run_consolidation_cycle(queue, tmp_path / "proposals")
    assert receipt["scanned"] == 4 and receipt["valid"] == 3
    assert receipt["proposals"] == ["planning"]
    assert receipt["rejections"]["ep-bad"] == ["erase_unproven"]
    assert len(receipt["written"]) == 1
    written = json.loads((tmp_path / "proposals" / receipt["written"][0]).read_text())
    assert written["domain"] == "planning" and written["status"] == "proposed"


# ── Interference battery ────────────────────────────────────────────────


@pytest.fixture(scope="module")
def tiny_model():
    mx = pytest.importorskip("mlx.core")
    pytest.importorskip("mlx_lm")
    from mlx_lm.models.qwen2 import Model, ModelArgs

    args = ModelArgs(
        model_type="qwen2", hidden_size=64, num_hidden_layers=8,
        intermediate_size=128, num_attention_heads=4, rms_norm_eps=1e-6,
        vocab_size=128, num_key_value_heads=2, max_position_embeddings=512,
        rope_theta=10000.0,
    )
    model = Model(args)
    mx.eval(model.parameters())
    return model


def test_identity_change_passes_battery(tiny_model):
    from core.brain.llm.latent_cortex.fast_weights import EpisodicFastWeights
    from core.brain.llm.latent_cortex.types import FastWeightsConfig
    from core.learning.interference_battery import run_interference_battery

    fw = EpisodicFastWeights(FastWeightsConfig(enabled=True, rank=2, target="o_proj"))
    receipt = run_interference_battery(
        tiny_model,
        # V=0 attach is EXACT identity — the battery must agree.
        apply_change=lambda: fw.attach(
            tiny_model.model, (2, 6), seed_stat=0.4, episode_id="battery-identity"
        ),
        revert_change=fw.detach,
    )
    assert receipt["verdict"] == "PASS"
    assert receipt["stable_fraction"] == 1.0


def test_disruptive_change_fails_battery_and_reverts(tiny_model):
    import mlx.core as mx

    from core.learning.interference_battery import (
        run_interference_battery,
        snapshot_probe_behavior,
    )

    layer = tiny_model.model.layers[3]
    original = layer.mlp.down_proj.weight
    baseline = snapshot_probe_behavior(tiny_model)

    def wreck():
        layer.mlp.down_proj.weight = original + mx.random.normal(
            original.shape, key=mx.random.key(1)
        ) * 0.5

    def restore():
        layer.mlp.down_proj.weight = original

    receipt = run_interference_battery(tiny_model, wreck, restore)
    assert receipt["verdict"] == "FAIL"
    assert receipt["stable_fraction"] < 0.9
    # Revert restored protected behavior exactly.
    after = snapshot_probe_behavior(tiny_model)
    assert [r["digest"] for r in after] == [r["digest"] for r in baseline]


def test_engine_export_lands_valid_candidate_in_queue(tiny_model, tmp_path, monkeypatch):
    """Full loop: episode with export enabled → queue → consumer validates."""
    monkeypatch.setenv("AURA_LOG_DIR", str(tmp_path / "logs"))
    import core.config as config_mod

    monkeypatch.setattr(config_mod, "DATA_DIR", tmp_path / "data", raising=False)

    from core.brain.llm.latent_cortex.engine import LatentCortexEngine
    from core.brain.llm.latent_cortex.types import (
        BranchConfig,
        CortexConfig,
        FastWeightsConfig,
        LatentOptConfig,
        RecurrenceConfig,
        WorkspaceConfig,
    )

    engine = LatentCortexEngine(
        tiny_model,
        config=CortexConfig(
            workspace=WorkspaceConfig(n_slots=4, seed=3),
            recurrence=RecurrenceConfig(max_steps=3, min_steps=2),
            branches=BranchConfig(n_branches=1),
            latent_opt=LatentOptConfig(enabled=False),
            fast_weights=FastWeightsConfig(
                enabled=True, rank=2, target="o_proj", opt_steps=3, lr=0.05,
                export_candidates=True,
            ),
            decode_max_tokens=6,
        ),
    )
    result = engine.reason(token_ids=[5, 9, 17, 3, 42], domain="unit-loop")
    assert result.ok
    queue = tmp_path / "data" / "latent_cortex" / "consolidation_queue"
    if "fast_weight_candidate_exported" in result.receipt.honest_flags:
        records = scan_queue(queue)
        assert len(records) == 1
        assert records[0].valid, records[0].rejection_reasons
        assert records[0].domain == "unit-loop"
    else:
        # A tiny random model may reject every optimizer step — then the
        # export must NOT have happened and the queue must be empty.
        assert not queue.exists() or not any(queue.iterdir())
