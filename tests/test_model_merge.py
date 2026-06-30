"""Tests for the task-arithmetic model-merge harness.

Contract: the delta math is exact (transplanting a fine-tune's delta onto a base recovers
the expected weights), TIES elects the right sign under conflict, DARE preserves the
delta in expectation, and the safetensors round-trip is lossless.
"""
from __future__ import annotations

import numpy as np

from core.learning.model_merge import (
    dare_merge,
    linear_merge,
    load_state_dict,
    merge_state_dicts,
    save_state_dict,
    task_vector,
    ties_merge,
    transplant_delta,
)


def _sd(**kw):
    return {k: np.asarray(v, dtype=np.float32) for k, v in kw.items()}


def test_task_vector_and_linear_merge_are_exact():
    base = _sd(w=[1.0, 2.0, 3.0])
    ft = _sd(w=[1.5, 2.0, 1.0])
    tv = task_vector(base, ft)
    np.testing.assert_allclose(tv["w"], [0.5, 0.0, -2.0])
    # base + (ft - base) == ft, exactly.
    recovered = linear_merge(base, [tv])
    np.testing.assert_allclose(recovered["w"], ft["w"])


def test_transplant_delta_moves_personality_onto_a_new_base():
    """Aura's case: personality delta (aura - qwen) applied to a reasoning base."""
    qwen = _sd(w=[0.0, 0.0, 0.0])
    aura = _sd(w=[1.0, -1.0, 2.0])          # personality fine-tune
    reasoning = _sd(w=[10.0, 10.0, 10.0])   # QwQ / R1-distill base
    merged = transplant_delta(qwen, aura, reasoning, scale=1.0)
    # reasoning + (aura - qwen)
    np.testing.assert_allclose(merged["w"], [11.0, 9.0, 12.0])


def test_ties_elects_the_dominant_sign_under_conflict():
    base = _sd(w=[0.0, 0.0])
    # On param 0 the deltas DISAGREE in sign; the larger-magnitude one should win.
    d1 = _sd(w=[1.0, 0.5])
    d2 = _sd(w=[-0.2, 0.5])
    merged = ties_merge(base, [d1, d2], density=1.0)
    assert merged["w"][0] > 0          # +1.0 outweighs -0.2 → positive sign elected
    np.testing.assert_allclose(merged["w"][1], 0.5)  # agreeing param averaged


def test_dare_preserves_delta_in_expectation():
    base = {"w": np.zeros(20000, dtype=np.float32)}
    delta = {"w": np.full(20000, 0.3, dtype=np.float32)}
    merged = dare_merge(base, [delta], drop=0.9, seed=7)
    # E[dropped+rescaled] ≈ original delta; mean should be close to 0.3.
    assert abs(float(merged["w"].mean()) - 0.3) < 0.02


def test_merge_methods_dispatch():
    base = _sd(w=[0.0, 0.0, 0.0])
    fts = {"a": _sd(w=[1.0, 0.0, 0.0]), "b": _sd(w=[0.0, 1.0, 0.0])}
    for method in ("linear", "ties", "dare"):
        out = merge_state_dicts(base, fts, method=method, density=1.0, drop=0.0)
        assert "w" in out and out["w"].shape == (3,)


def test_safetensors_round_trip(tmp_path):
    state = _sd(a=[[1.0, 2.0], [3.0, 4.0]], b=[0.5, -0.5])
    path = tmp_path / "m.safetensors"
    save_state_dict(state, path)
    loaded = load_state_dict(path)
    np.testing.assert_allclose(loaded["a"], state["a"])
    np.testing.assert_allclose(loaded["b"], state["b"])


def test_end_to_end_dir_merge(tmp_path):
    from core.learning.model_merge import merge_model_dirs

    base_dir = tmp_path / "base"; base_dir.mkdir()
    ft_dir = tmp_path / "ft"; ft_dir.mkdir()
    save_state_dict(_sd(w=[0.0, 0.0]), base_dir / "model.safetensors")
    save_state_dict(_sd(w=[2.0, -2.0]), ft_dir / "model.safetensors")
    (base_dir / "config.json").write_text("{}")
    manifest = merge_model_dirs(base_dir, {"ft": ft_dir}, tmp_path / "out", method="linear")
    assert manifest["tensors"] == 1
    assert "config.json" in manifest["copied_artifacts"]
    merged = load_state_dict(tmp_path / "out" / "model.safetensors")
    np.testing.assert_allclose(merged["w"], [2.0, -2.0])  # base + delta
