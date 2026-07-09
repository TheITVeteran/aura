"""core/consciousness/inner_light/controls.py — the negative controls.

The inner-light test is only as credible as its controls. Each of these either
transforms a real activity matrix to *destroy one axis* of the conscious-like
signature while preserving the others, or synthesises a reference system that is
categorically not conscious-like. The claim the battery can then make is a
conjunction: Aura is high on integration AND differentiation AND criticality AND
ignition, while every control below fails at least one — so no control reproduces
the full signature.

Transforms of a real matrix ``M`` (n_channels × time):
  * time_shuffle       — permute time. Destroys temporal structure (criticality,
    ignition) while preserving marginals (and, with a shared permutation,
    instantaneous cross-channel correlation).
  * phase_randomize    — FFT surrogate with shared random phases. Preserves each
    channel's power spectrum and the linear cross-spectrum, destroys non-linear /
    higher-order structure (complexity, ignition). The "best linear-Gaussian" twin.
  * lesion_decouple    — random circular per-channel shift. Preserves each
    channel's own autocorrelation, destroys *between*-channel integration: a
    federation of intact organs that no longer bind. This is "lesioned Aura".

Synthetic references (given a shape):
  * white_noise        — maximal differentiation, zero integration.
  * ordered            — a repeating pattern: minimal differentiation.
  * feedforward_chain  — a strictly feed-forward pipeline (a→b→c, no feedback):
    the "hard drive" — it carries and transforms information but has no recurrent
    cause-effect loop, so the integration/criticality that need feedback collapse.
"""
from __future__ import annotations

import numpy as np

_DEFAULT_SEED = 20260709


def _as_matrix(M: np.ndarray | list) -> np.ndarray:
    arr = np.asarray(M, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return arr


# ── transforms of real activity ──────────────────────────────────────────────

def time_shuffle(M: np.ndarray | list, *, seed: int = _DEFAULT_SEED, independent: bool = False) -> np.ndarray:
    """Permute the time axis. Shared permutation keeps instantaneous spatial
    correlations; ``independent=True`` breaks those too."""
    arr = _as_matrix(M)
    n, T = arr.shape
    rng = np.random.default_rng(seed)
    if independent:
        return np.stack([arr[i, rng.permutation(T)] for i in range(n)])
    return arr[:, rng.permutation(T)]


def phase_randomize(M: np.ndarray | list, *, seed: int = _DEFAULT_SEED) -> np.ndarray:
    """FFT surrogate: preserve the (cross-)power spectrum, randomise phases.

    The same random phase is applied per frequency across channels, so the linear
    correlation structure and each channel's spectrum survive while non-linear and
    higher-order structure is destroyed.
    """
    arr = _as_matrix(M)
    n, T = arr.shape
    rng = np.random.default_rng(seed)
    F = np.fft.rfft(arr, axis=1)
    nf = F.shape[1]
    phases = rng.uniform(0.0, 2.0 * np.pi, size=nf)
    phases[0] = 0.0  # keep the DC component real
    if T % 2 == 0:
        phases[-1] = 0.0  # keep the Nyquist component real
    rotated = np.abs(F) * np.exp(1j * (np.angle(F) + phases[None, :]))
    return np.fft.irfft(rotated, n=T, axis=1)


def lesion_decouple(M: np.ndarray | list, *, seed: int = _DEFAULT_SEED) -> np.ndarray:
    """Randomly circular-shift each channel: keep per-channel autocorrelation,
    destroy between-channel binding. Lesioned / federated Aura."""
    arr = _as_matrix(M)
    n, T = arr.shape
    rng = np.random.default_rng(seed)
    return np.stack([np.roll(arr[i], int(rng.integers(1, T))) for i in range(n)])


# ── synthetic references ──────────────────────────────────────────────────────

def white_noise(shape: tuple[int, int], *, seed: int = _DEFAULT_SEED) -> np.ndarray:
    """Maximal differentiation, zero integration."""
    return np.random.default_rng(seed).standard_normal(size=shape)


def ordered(shape: tuple[int, int], *, seed: int = _DEFAULT_SEED, period: int = 8) -> np.ndarray:
    """A repeating pattern across time: minimal differentiation."""
    n, T = shape
    rng = np.random.default_rng(seed)
    motif = rng.standard_normal(size=(n, period))
    reps = int(np.ceil(T / period))
    return np.tile(motif, (1, reps))[:, :T]


def feedforward_chain(shape: tuple[int, int], *, seed: int = _DEFAULT_SEED, leak: float = 0.6) -> np.ndarray:
    """A strictly feed-forward pipeline: channel i is a lagged, noisy transform of
    channel i-1, with no path back. Information flows through, but there is no
    recurrent loop — the "hard drive"."""
    n, T = shape
    rng = np.random.default_rng(seed)
    out = np.zeros((n, T))
    out[0] = rng.standard_normal(size=T)  # the input stream
    for i in range(1, n):
        shifted = np.roll(out[i - 1], 1)
        shifted[0] = 0.0
        out[i] = leak * shifted + (1.0 - leak) * rng.standard_normal(size=T)
    return out


ALL_CONTROLS = (
    "time_shuffle",
    "phase_randomize",
    "lesion_decouple",
    "white_noise",
    "ordered",
    "feedforward_chain",
)

__all__ = list(ALL_CONTROLS) + ["ALL_CONTROLS"]
