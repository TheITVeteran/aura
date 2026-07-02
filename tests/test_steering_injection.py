"""Steering injection must actually modify hidden states when active.

The original live A/B runner assigned ``layer.__call__`` on the instance,
which Python's special-method lookup bypasses entirely — its "steered"
condition never injected anything. These tests pin the working mechanism.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from core.evaluation.steering_injection import (  # noqa: E402 — after importorskip
    ResidualSteeringInjector,
    load_production_vectors,
)


class _Layer:
    def __call__(self, h, *args, **kwargs):
        return h * 1.0  # identity-ish transform


class _Inner:
    def __init__(self, n_layers: int):
        self.layers = [_Layer() for _ in range(n_layers)]


class _Model:
    def __init__(self, n_layers: int = 4):
        self.model = _Inner(n_layers)

    def forward_through(self, h):
        for layer in self.model.layers:
            h = layer(h)
        return h


def test_injection_changes_hidden_state_only_when_active():
    model = _Model(n_layers=4)
    vec = np.zeros(8, dtype=np.float32)
    vec[0] = 1.0
    injector = ResidualSteeringInjector(model, {2: vec}, alpha=5.0)
    h = mx.ones((1, 3, 8))

    baseline = model.forward_through(h)
    with injector:
        injector.active = False
        unsteered = model.forward_through(h)
        injector.active = True
        steered = model.forward_through(h)
    restored = model.forward_through(h)

    assert bool(mx.allclose(baseline, unsteered).item())
    assert not bool(mx.allclose(baseline, steered).item()), (
        "active injection must change the hidden state — the instance "
        "__call__ assignment bug produced exactly this failure"
    )
    # Injection adds alpha on the steered axis at the hooked layer.
    delta = np.array(steered - baseline)
    assert delta[..., 0].max() == pytest.approx(5.0, rel=1e-3)
    assert np.abs(delta[..., 1:]).max() == pytest.approx(0.0, abs=1e-6)
    assert injector.injection_count > 0
    # Hooks removed: model behaves like baseline again.
    assert bool(mx.allclose(baseline, restored).item())


def test_calling_convention_actually_intercepts():
    """Guard against regressions to instance-attribute patching."""
    model = _Model(n_layers=2)
    vec = np.ones(4, dtype=np.float32)
    injector = ResidualSteeringInjector(model, {0: vec}, alpha=1.0)
    installed = injector.install()
    try:
        injector.active = True
        layer = model.model.layers[0]
        out = layer(mx.zeros((1, 1, 4)))
        assert float(np.array(out).sum()) != 0.0, (
            "layer(...) did not route through the injection subclass"
        )
    finally:
        injector.remove()
    assert installed == 1


def test_load_production_vectors_filters_and_normalizes(tmp_path):
    def _write(name, dimension, layer, vec, extracted=True):
        np.savez(
            tmp_path / name,
            v=vec.astype(np.float32),
            dimension=np.array(dimension),
            layer=np.array(layer),
            extracted=np.array(extracted),
        )

    _write("a.npz", "valence_positive", 5, np.array([3.0, 0.0, 0.0, 0.0]))
    _write("b.npz", "curiosity", 5, np.array([0.0, 4.0, 0.0, 0.0]))
    _write("c.npz", "valence_positive", 7, np.array([0.0, 0.0, 9.0, 0.0]))
    _write("d.npz", "frustration", 5, np.array([1.0, 1.0, 1.0, 1.0]))  # not requested
    _write("e.npz", "valence_positive", 9, np.array([2.0, 0.0, 0.0, 0.0]), extracted=False)

    vectors = load_production_vectors(tmp_path)

    assert set(vectors) == {5, 7}, "bootstrap and unrequested dimensions must be excluded"
    for vec in vectors.values():
        assert np.linalg.norm(vec) == pytest.approx(1.0, rel=1e-5)
    # Layer 5 averages the two unit axes then renormalizes.
    assert vectors[5][0] == pytest.approx(vectors[5][1], rel=1e-5)


def test_live_ab_artifact_is_not_greedy_theater():
    """The committed A/B artifact must come from sampled, injected runs:
    per-condition samples must not be identical (greedy collapse) once the
    rebuilt runner has produced a fresh artifact with variation metadata."""
    from pathlib import Path

    artifact = Path(__file__).resolve().parent / "CAA_32B_AB_LIVE_RESULTS.json"
    if not artifact.exists():
        pytest.skip("no live A/B artifact present")
    data = json.loads(artifact.read_text(encoding="utf-8"))
    if "sampling" not in data:
        pytest.xfail(
            "legacy greedy artifact (pre-rebuild): known theater, superseded "
            "by the sampled runner; regenerate via tests/run_32b_steering_ab_live.py"
        )
    assert data["sampling"].get("temperature", 0.0) > 0.0
    samples = (data.get("analysis") or {}).get("samples", {})
    steered = samples.get("steered_black_box", [])
    assert len(set(steered)) > 1, "steered samples are identical — greedy collapse"
