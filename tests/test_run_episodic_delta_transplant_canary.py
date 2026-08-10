from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

from tools.run_episodic_delta_transplant_canary import _load_candidate


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
