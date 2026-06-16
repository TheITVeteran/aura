"""Rust-accelerated substrate kernels with a NumPy fallback (task #40).

The continuous-time substrate (core/consciousness/unified_field) runs a tight
integration loop every tick. On Apple Silicon this is a real CPU/heat source,
and heat correlates with the cortex-wedge frontier (#45). This module is the
seam for porting those hot kernels to Rust (rust_extensions/aura_m1_ext):

  - if the compiled extension is present, the Rust kernel runs;
  - otherwise the exact NumPy reference runs, so behavior is identical whether or
    not the extension is built.

Build the extension with (from the repo root)::

    maturin develop -m rust_extensions/aura_m1_ext/Cargo.toml --release
    # or: cd rust_extensions/aura_m1_ext && cargo build --release

Until then, ``RUST_ACCEL_AVAILABLE`` is False and the NumPy path is used. The
parity test (tests/test_substrate_accel.py) verifies the two paths agree.
"""
from __future__ import annotations

import numpy as np

_ACCEL_IMPORT_ERRORS = (ImportError, AttributeError, OSError)

try:  # pragma: no cover - depends on whether the native ext is built
    import aura_m1_ext as _ext

    RUST_ACCEL_AVAILABLE = hasattr(_ext, "field_integrate")
except _ACCEL_IMPORT_ERRORS:
    _ext = None
    RUST_ACCEL_AVAILABLE = False


def _field_integrate_numpy(
    f: np.ndarray, activity: np.ndarray, noise: np.ndarray, decay: float, dt: float
) -> np.ndarray:
    """Reference: next = clip(f + (-decay*f + activity + noise)*dt, -1, 1)."""
    df = (-float(decay) * f + activity + noise) * float(dt)
    return np.clip(f + df, -1.0, 1.0).astype(np.float32)


def field_integrate(
    f: np.ndarray, activity: np.ndarray, noise: np.ndarray, decay: float, dt: float
) -> np.ndarray:
    """One Euler integration step of the unified field.

    Rust-accelerated when ``aura_m1_ext`` is built; otherwise the identical NumPy
    reference. Always returns a float32 array of the same shape as ``f``.
    """
    if RUST_ACCEL_AVAILABLE:
        try:
            out = _ext.field_integrate(
                np.ascontiguousarray(f, dtype=np.float32).ravel().tolist(),
                np.ascontiguousarray(activity, dtype=np.float32).ravel().tolist(),
                np.ascontiguousarray(noise, dtype=np.float32).ravel().tolist(),
                float(decay),
                float(dt),
            )
            return np.asarray(out, dtype=np.float32)
        except _ACCEL_IMPORT_ERRORS + (ValueError, TypeError, RuntimeError):
            # Never let an extension hiccup break the substrate — fall back.
            pass
    return _field_integrate_numpy(
        np.asarray(f, dtype=np.float32),
        np.asarray(activity, dtype=np.float32),
        np.asarray(noise, dtype=np.float32),
        decay,
        dt,
    )
