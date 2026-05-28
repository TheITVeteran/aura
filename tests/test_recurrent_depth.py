"""Tests for the Mythos-inspired recurrent-depth patch.

Guards the load-bearing assumption: mlx_lm's KVCache state/meta_state
snapshot/restore correctly rewinds offset after a mutation. A silent
failure here would have the recurrent loop accumulate N copies of K/V
into the cache — far worse than leaving recurrent depth off.
"""
from __future__ import annotations

import os
import sys
import types
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.brain.llm.recurrent_depth import (  # noqa: E402
    CacheSnapshotError,
    _self_test_cache_snapshot,
    _snapshot_recurrent_caches,
    _restore_recurrent_caches,
    _get_lane_defaults,
    resolve_loops_for_model,
)

_RUN_NATIVE_MLX_HARDWARE_TESTS = os.getenv("AURA_RUN_MLX_HARDWARE_TESTS") == "1"


def _require_native_mlx_hardware() -> None:
    if _RUN_NATIVE_MLX_HARDWARE_TESTS:
        return
    pytest.importorskip(
        "_aura_native_mlx_hardware_tests_enabled",
        reason=(
            "native MLX/Metal cache tests require AURA_RUN_MLX_HARDWARE_TESTS=1; "
            "the final proof validates recurrent depth on the live 32B lane"
        ),
    )


@pytest.mark.hardware
def test_self_test_cache_snapshot_passes_on_installed_mlx_lm():
    """If this fails, mlx_lm's cache contract changed and we must not patch."""
    _require_native_mlx_hardware()
    _self_test_cache_snapshot()


def test_snapshot_fails_loud_on_unsupported_cache():
    """Incompatible caches must raise, never silently no-op."""

    class _BadCache:
        """Neither state/meta_state nor keys/values/offset."""
        pass

    with pytest.raises(CacheSnapshotError):
        _snapshot_recurrent_caches([_BadCache()], 0, 1)


def test_lane_defaults_cover_real_model_sizes():
    """Qwen2.5-32B has 64 layers; Qwen2.5-72B has 80. Both must land in
    the intended runtime envelopes for interactive use."""
    assert _get_lane_defaults(64)[0] >= 2, "32B (64 layers) must map to a looped lane"
    assert _get_lane_defaults(80)[0] == 1, "72B (80 layers) should default to a single pass for live solver turns"
    # And the small-model lanes must be standard-pass (no unnecessary cost).
    assert _get_lane_defaults(28)[0] == 1, "14B (28-40 layers) should be standard"
    assert _get_lane_defaults(12)[0] == 1, "7B class should be standard"


def test_resolve_loops_honors_72b_lane_override(monkeypatch):
    class _Inner:
        layers = [object()] * 80

    class _Model:
        model = _Inner()

    monkeypatch.setenv("AURA_RECURRENT_LOOPS_72B", "2")
    monkeypatch.delenv("AURA_RECURRENT_LOOPS", raising=False)

    assert resolve_loops_for_model(_Model()) == 2


def test_recurrent_depth_invalid_env_fails_as_runtime_error(monkeypatch):
    class _Inner:
        layers = [object()] * 64

    class _Model:
        model = _Inner()

    monkeypatch.setenv("AURA_RECURRENT_LOOPS_32B", "twice")
    monkeypatch.delenv("AURA_RECURRENT_LOOPS", raising=False)

    with pytest.raises(RuntimeError, match="AURA_RECURRENT_LOOPS_32B"):
        resolve_loops_for_model(_Model())


def test_recurrent_depth_rejects_unsafe_fraction_override(monkeypatch):
    import core.brain.llm.recurrent_depth as rd

    class _Inner:
        layers = [object()] * 64

    class _Model:
        model = _Inner()

    monkeypatch.setenv("AURA_RECURRENT_PRELUDE", "0.95")
    monkeypatch.delenv("AURA_RECURRENT_LOOPS", raising=False)
    monkeypatch.delenv("AURA_RECURRENT_LOOPS_32B", raising=False)

    with pytest.raises(RuntimeError, match="AURA_RECURRENT_PRELUDE"):
        rd.apply_for_model(_Model())


@pytest.mark.hardware
def test_restore_rewinds_mlx_cache():
    """Direct end-to-end proof the snapshot/restore actually works."""
    _require_native_mlx_hardware()
    import mlx.core as mx
    from mlx_lm.models.cache import KVCache

    c = KVCache()
    c.update_and_fetch(mx.ones((1, 2, 8, 16)), mx.ones((1, 2, 8, 16)))
    pre_offset = c.offset
    snap = _snapshot_recurrent_caches([c], 0, 1)

    c.update_and_fetch(mx.ones((1, 2, 1, 16)) * 3, mx.ones((1, 2, 1, 16)) * 3)
    assert c.offset > pre_offset, "Mutation did not advance cache offset"

    _restore_recurrent_caches([c], 0, 1, snap)
    assert c.offset == pre_offset, f"Restore failed: {pre_offset} → {c.offset}"


def _install_fake_mlx_modules(monkeypatch):
    mlx_pkg = types.ModuleType("mlx")
    mlx_core = types.ModuleType("mlx.core")
    mlx_pkg.core = mlx_core
    monkeypatch.setitem(sys.modules, "mlx", mlx_pkg)
    monkeypatch.setitem(sys.modules, "mlx.core", mlx_core)

    mlx_lm = types.ModuleType("mlx_lm")
    mlx_lm_models = types.ModuleType("mlx_lm.models")
    mlx_lm_base = types.ModuleType("mlx_lm.models.base")
    mlx_lm_base.create_attention_mask = lambda _h, _cache: None
    monkeypatch.setitem(sys.modules, "mlx_lm", mlx_lm)
    monkeypatch.setitem(sys.modules, "mlx_lm.models", mlx_lm_models)
    monkeypatch.setitem(sys.modules, "mlx_lm.models.base", mlx_lm_base)


def test_apply_recurrent_depth_is_instance_scoped(monkeypatch):
    import core.brain.llm.recurrent_depth as rd

    _install_fake_mlx_modules(monkeypatch)
    monkeypatch.setattr(rd, "_self_test_cache_snapshot", lambda: None)

    class _Inner:
        def __init__(self):
            self.layers = [object()] * 64

        def __call__(self, *_args, **_kwargs):
            return "original"

    class _Model:
        def __init__(self):
            self.model = _Inner()

    first = _Model()
    second = _Model()
    original_class = second.model.__class__

    assert rd.apply_recurrent_depth(first, n_loops=2) is True

    assert first.model.__class__ is not original_class
    assert second.model.__class__ is original_class
    assert second.model("prompt") == "original"

    assert rd.remove_recurrent_depth(first) is True
    assert first.model.__class__ is original_class


def test_recurrent_forward_executes_middle_block_multiple_times(monkeypatch):
    import core.brain.llm.recurrent_depth as rd

    _install_fake_mlx_modules(monkeypatch)
    monkeypatch.setattr(rd, "_self_test_cache_snapshot", lambda: None)

    class _Layer:
        def __init__(self):
            self.calls = 0

        def __call__(self, h, _mask, _cache):
            self.calls += 1
            return h + 1

    class _Inner:
        def __init__(self):
            self.layers = [_Layer() for _ in range(64)]

        def embed_tokens(self, inputs):
            return inputs

        def norm(self, h):
            return h

        def __call__(self, inputs, cache=None, input_embeddings=None):
            return inputs

    class _Model:
        def __init__(self):
            self.model = _Inner()

    model = _Model()

    assert rd.apply_recurrent_depth(
        model,
        n_loops=2,
        prelude_frac=0.20,
        coda_frac=0.20,
        residual_alpha=0.0,
    ) is True

    result = model.model(1, cache=[None] * 64)
    config = rd.get_recurrent_config(model)

    assert config["prelude_end"] == 12
    assert config["coda_start"] == 52
    assert result == 105
    assert all(layer.calls == 1 for layer in model.model.layers[:12])
    assert all(layer.calls == 2 for layer in model.model.layers[12:52])
    assert all(layer.calls == 1 for layer in model.model.layers[52:])
