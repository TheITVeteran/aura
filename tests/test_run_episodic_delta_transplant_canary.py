from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

from tools.run_episodic_delta_transplant_canary import (
    EPISODE_ID,
    _load_candidate,
    _producer_export_diagnostic,
)


def _candidate(tmp_path, *, extra_tensor: bool = False):
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    arrays = {
        "layer7_U": np.ones((5, 2), dtype=np.float32),
        "layer7_V": np.ones((2, 5), dtype=np.float32),
    }
    if extra_tensor:
        arrays["layer8_U"] = np.ones((5, 2), dtype=np.float32)
    np.savez(candidate / "delta_weights.npz", **arrays)
    delta = (candidate / "delta_weights.npz").read_bytes()
    evidence = {
        "schema": "aura.latent_cortex.fast_weight_candidate.v1",
        "episode_id": "episode",
        "target": "o_proj",
        "rank": 2,
        "layers": [7],
        "artifacts": {
            "delta_weights.npz": {
                "sha256": hashlib.sha256(delta).hexdigest(),
                "size_bytes": len(delta),
            }
        },
    }
    (candidate / "evidence.json").write_text(
        json.dumps(evidence),
        encoding="utf-8",
    )
    return candidate


def test_load_candidate_reconstructs_exact_private_tensor_inventory(tmp_path) -> None:
    snapshots, binding = _load_candidate(
        _candidate(tmp_path),
        episode_id="episode",
        expected_layers=(7,),
        expected_rank=2,
        scale=0.25,
    )

    assert len(snapshots) == 1
    assert snapshots[0]["layer"] == 7
    assert snapshots[0]["scale"] == 0.25
    assert snapshots[0]["U"].shape == (5, 2)
    assert snapshots[0]["V"].shape == (2, 5)
    assert len(binding["evidence_sha256"]) == 64
    assert len(binding["delta_sha256"]) == 64


def test_load_candidate_rejects_unbound_extra_tensor(tmp_path) -> None:
    with pytest.raises(RuntimeError, match="tensor inventory differs"):
        _load_candidate(
            _candidate(tmp_path, extra_tensor=True),
            episode_id="episode",
            expected_layers=(7,),
            expected_rank=2,
            scale=0.25,
        )


def test_producer_export_diagnostic_distinguishes_ineligible_delta() -> None:
    diagnostic = _producer_export_diagnostic(
        {
            "checkpoint_fingerprint": "sha256:checkpoint",
            "fast_weights_erased": True,
            "fast_weight_optimized_steps": 0,
            "fast_weight_rejected_steps": 2,
            "fast_weight_loss_trail": [1.0],
            "fast_weight_verifier": {"decision": "erased_non_improvement"},
            "fast_weight_learning": {"disposition": "rejected_non_improvement"},
            "honest_flags": ["fast_weight_no_accepted_step"],
        }
    )

    assert diagnostic["exported"] is False
    assert diagnostic["eligible_by_receipt"] is False
    assert diagnostic["reason"] == "producer_not_export_eligible"
    assert diagnostic["prerequisites"]["adaptation_retained"] is False
    assert diagnostic["prerequisites"]["accepted_step"] is False
    assert diagnostic["prerequisites"]["loss_improved"] is False


def test_producer_export_diagnostic_exposes_export_boundary_failure() -> None:
    diagnostic = _producer_export_diagnostic(
        {
            "checkpoint_fingerprint": "sha256:checkpoint",
            "fast_weights_erased": True,
            "fast_weight_optimized_steps": 1,
            "fast_weight_rejected_steps": 0,
            "fast_weight_loss_trail": [1.0, 0.5],
            "fast_weight_verifier": {"decision": "accepted_causal_improvement"},
            "fast_weight_learning": {
                "disposition": "accepted_probe_not_output_under_incumbent_policy"
            },
            "honest_flags": [],
        }
    )

    assert diagnostic["exported"] is False
    assert diagnostic["eligible_by_receipt"] is True
    assert diagnostic["reason"] == "export_boundary_failed_or_refused"


def test_matched_control_rejection_is_not_misreported_as_export_failure() -> None:
    diagnostic = _producer_export_diagnostic(
        {
            "checkpoint_fingerprint": "sha256:checkpoint",
            "fast_weights_erased": True,
            "fast_weight_optimized_steps": 4,
            "fast_weight_rejected_steps": 0,
            "fast_weight_loss_trail": [2.0, 1.0],
            "fast_weight_verifier": {"decision": "erased_matched_control"},
            "fast_weight_learning": {"disposition": "rejected_matched_control"},
            "honest_flags": ["fast_weight_matched_control_rejected"],
        }
    )

    assert diagnostic["eligible_by_receipt"] is False
    assert diagnostic["prerequisites"]["adaptation_retained"] is False
    assert diagnostic["reason"] == "producer_not_export_eligible"


def test_scientific_episode_identity_is_source_revision_independent() -> None:
    assert EPISODE_ID == "episodic-transplant-modular-d3-v1"
    assert "commit" not in EPISODE_ID
