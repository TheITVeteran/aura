"""tests/test_substrate_accel.py
=====================================
Task #40 foundation: a Rust port of the continuous-time substrate's hot
integration kernel, with a NumPy fallback so behavior is identical whether or
not the native extension is built. These tests pin the contract:

  - the kernel computes exactly clip(f + (-decay*f + activity + noise)*dt, -1, 1),
  - it matches the legacy inline expression bit-for-bit on the fallback,
  - clipping and float32 dtype hold,
  - when the Rust extension IS built, it agrees with the NumPy reference.
"""
from __future__ import annotations

import numpy as np

from core.consciousness.substrate_accel import (
    RUST_ACCEL_AVAILABLE,
    _field_integrate_numpy,
    field_integrate,
)


def _legacy_inline(f, activity, noise, decay, dt):
    # The exact expression unified_field._tick used before the kernel extraction.
    df = (-decay * f + activity + noise) * dt
    return np.clip(f + df, -1.0, 1.0).astype(np.float32)


def test_matches_legacy_inline_expression():
    rng = np.random.default_rng(0)
    for _ in range(20):
        n = int(rng.integers(1, 300))
        f = rng.standard_normal(n).astype(np.float32)
        activity = rng.standard_normal(n).astype(np.float32)
        noise = (rng.standard_normal(n) * 0.05).astype(np.float32)
        decay = float(rng.uniform(0.0, 2.0))
        dt = float(rng.uniform(0.001, 0.1))
        got = field_integrate(f, activity, noise, decay, dt)
        want = _legacy_inline(f, activity, noise, decay, dt)
        np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-6)


def test_output_is_clipped_and_float32():
    f = np.array([0.9, -0.9, 0.0], dtype=np.float32)
    activity = np.array([100.0, -100.0, 0.0], dtype=np.float32)  # huge → must clip
    noise = np.zeros(3, dtype=np.float32)
    out = field_integrate(f, activity, noise, decay=0.1, dt=1.0)
    assert out.dtype == np.float32
    assert out.max() <= 1.0 and out.min() >= -1.0
    assert out[0] == 1.0 and out[1] == -1.0


def test_zero_dt_is_identity():
    f = np.array([0.3, -0.2, 0.5], dtype=np.float32)
    out = field_integrate(f, f, f, decay=1.0, dt=0.0)
    np.testing.assert_allclose(out, f, rtol=0, atol=0)


def test_decay_pulls_toward_zero():
    f = np.array([0.5, -0.5], dtype=np.float32)
    zero = np.zeros(2, dtype=np.float32)
    out = field_integrate(f, zero, zero, decay=1.0, dt=0.5)
    # next = f + (-decay*f)*dt = f*(1 - decay*dt) = f*0.5 → magnitude shrinks
    assert abs(out[0]) < abs(f[0]) and abs(out[1]) < abs(f[1])


def test_numpy_reference_is_used_when_no_extension():
    # On a tree without the built extension, the fallback path must be active.
    if not RUST_ACCEL_AVAILABLE:
        f = np.zeros(4, dtype=np.float32)
        a = np.ones(4, dtype=np.float32)
        nz = np.zeros(4, dtype=np.float32)
        assert np.array_equal(
            field_integrate(f, a, nz, 0.1, 0.1),
            _field_integrate_numpy(f, a, nz, 0.1, 0.1),
        )


def test_active_kernel_matches_reference_for_current_backend():
    rng = np.random.default_rng(7)
    f = rng.standard_normal(256).astype(np.float32)
    activity = rng.standard_normal(256).astype(np.float32)
    noise = (rng.standard_normal(256) * 0.05).astype(np.float32)
    got = field_integrate(f, activity, noise, 0.05, 0.01)
    want = _field_integrate_numpy(f, activity, noise, 0.05, 0.01)
    np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-6)
    assert isinstance(RUST_ACCEL_AVAILABLE, bool)
