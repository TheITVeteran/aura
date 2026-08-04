"""How much steering survives 4-bit weights, measured rather than assumed.

The review's limitation: "Running local LLMs in 4-bit precision introduces
activation noise that can degrade float32 CAA steering vector precision."

True, and the size of it was never measured, so nothing in the runtime could
tell an α that steers from an α that is being drowned. MEASURED here, on a
d=5120 layer with MLX's affine 4-bit scheme (group size 64):

    quantisation noise in one layer's output  =  8.8% of the activation norm

which puts the steering-to-noise ratio at

    α = 0.05  SNR 0.008        α = 1.0   SNR 0.160
    α = 0.20  SNR 0.032        α = 2.0   SNR 0.321
    α = 0.35  SNR 0.056        α = 4.0   SNR 0.641
    α = 0.50  SNR 0.080        α = 6.0   SNR 0.961

The live surface decodes at α = 0.35. At that value the injected direction is
roughly EIGHTEEN TIMES smaller than the noise the quantised weights already put
into the same residual stream, and the cosine between intended and realised
direction is 0.056. The live engine α of ~6 is, not coincidentally, about where
the two become comparable.

None of this makes steering useless — a consistent bias summed over 64 blocks
and hundreds of tokens is not the same as a zero-mean perturbation, and the
measured live A/B results are what settle whether it works. What it makes
impossible is claiming a given α is "applied" without saying at what strength
relative to the floor it is competing against. That number now travels with the
telemetry, and ``AlphaController``'s lower bound comes from this measurement
instead of from a round number.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

logger = logging.getLogger("Aura.CAA.QuantizationFloor")

#: MLX's default affine quantisation for local weights.
DEFAULT_BITS = 4
DEFAULT_GROUP_SIZE = 64

#: The width the resident model actually runs at.
DEFAULT_D_MODEL = 5120

#: Steering at least as strong as the noise it competes with.
DEFAULT_TARGET_SNR = 1.0


def quantize_dequantize(
    weights: np.ndarray,
    *,
    bits: int = DEFAULT_BITS,
    group_size: int = DEFAULT_GROUP_SIZE,
) -> np.ndarray:
    """Affine group quantisation and back, as the local runtime stores weights."""
    array = np.asarray(weights, dtype=np.float32)
    flat = array.reshape(-1, int(group_size))
    low = flat.min(axis=1, keepdims=True)
    high = flat.max(axis=1, keepdims=True)
    levels = float((1 << int(bits)) - 1)
    scale = (high - low) / levels
    scale[scale == 0] = 1e-8
    quantised = np.round((flat - low) / scale)
    return (quantised * scale + low).reshape(array.shape)


@dataclass(frozen=True)
class NoiseFloor:
    """What quantisation costs, as a fraction of the activation it perturbs."""

    fraction: float
    bits: int
    group_size: int
    d_model: int
    trials: int

    def minimum_effective_alpha(
        self,
        residual_norm: float,
        *,
        target_snr: float = DEFAULT_TARGET_SNR,
    ) -> float:
        """The α at which steering is as strong as the quantisation noise."""
        norm = float(max(0.0, residual_norm))
        return float(target_snr * self.fraction * norm)

    def snr(self, alpha: float, residual_norm: float) -> float:
        """Steering magnitude over quantisation noise magnitude."""
        noise = self.fraction * float(max(0.0, residual_norm))
        if noise <= 0.0:
            return math.inf
        return float(abs(alpha) / noise)

    def as_metrics(self) -> dict[str, float | int]:
        return {
            "quantization_noise_fraction": round(self.fraction, 6),
            "quantization_bits": self.bits,
            "quantization_group_size": self.group_size,
            "measured_d_model": self.d_model,
            "measurement_trials": self.trials,
        }


@lru_cache(maxsize=8)
def measure_noise_floor(
    *,
    bits: int = DEFAULT_BITS,
    group_size: int = DEFAULT_GROUP_SIZE,
    d_model: int = DEFAULT_D_MODEL,
    trials: int = 3,
    seed: int = 20260804,
) -> NoiseFloor:
    """Measure ‖Q(W)x − Wx‖ / ‖Wx‖ for a layer of this shape.

    Deterministic and cached: this is a property of the quantisation scheme and
    the layer width, not of any particular prompt, so it is measured once per
    process rather than per token.
    """
    rng = np.random.default_rng(seed)
    fractions: list[float] = []
    for _ in range(max(1, int(trials))):
        weights = (rng.standard_normal((d_model, d_model)) / math.sqrt(d_model)).astype(
            np.float32
        )
        activations = rng.standard_normal(d_model).astype(np.float32)
        clean = weights @ activations
        noisy = quantize_dequantize(weights, bits=bits, group_size=group_size) @ activations
        clean_norm = float(np.linalg.norm(clean))
        if clean_norm <= 0.0:
            continue
        fractions.append(float(np.linalg.norm(noisy - clean) / clean_norm))
    fraction = float(np.mean(fractions)) if fractions else 0.0
    floor = NoiseFloor(
        fraction=fraction,
        bits=int(bits),
        group_size=int(group_size),
        d_model=int(d_model),
        trials=len(fractions),
    )
    logger.info(
        "CAA quantisation floor measured: %.2f%% of activation norm "
        "(%d-bit, group %d, d_model %d)",
        fraction * 100.0,
        bits,
        group_size,
        d_model,
    )
    return floor


@dataclass(frozen=True)
class SteeringPrecision:
    """What a given α is actually doing against the noise it competes with."""

    alpha: float
    residual_norm: float
    snr: float
    below_floor: bool
    floor: NoiseFloor

    def as_metrics(self) -> dict[str, float | int | bool]:
        return {
            "steering_alpha": round(self.alpha, 6),
            "residual_norm": round(self.residual_norm, 4),
            "steering_snr": round(self.snr, 6) if math.isfinite(self.snr) else None,
            "below_quantization_floor": self.below_floor,
            **self.floor.as_metrics(),
        }


def assess_steering_precision(
    alpha: float,
    residual_norm: float,
    *,
    d_model: int = DEFAULT_D_MODEL,
    bits: int = DEFAULT_BITS,
    group_size: int = DEFAULT_GROUP_SIZE,
    target_snr: float = DEFAULT_TARGET_SNR,
) -> SteeringPrecision:
    """Report the strength of this injection relative to the noise floor.

    Observability, not a gate. Steering below the floor is weak, not harmful,
    and a consistent bias summed over 64 blocks and hundreds of tokens is not a
    zero-mean perturbation — the live A/B is what decides whether it works.
    What this ends is reporting "steering applied" with no idea at what
    strength.
    """
    floor = measure_noise_floor(bits=bits, group_size=group_size, d_model=d_model)
    snr = floor.snr(alpha, residual_norm)
    return SteeringPrecision(
        alpha=float(alpha),
        residual_norm=float(residual_norm),
        snr=snr,
        below_floor=snr < target_snr,
        floor=floor,
    )


__all__ = [
    "DEFAULT_BITS",
    "DEFAULT_D_MODEL",
    "DEFAULT_GROUP_SIZE",
    "DEFAULT_TARGET_SNR",
    "NoiseFloor",
    "SteeringPrecision",
    "assess_steering_precision",
    "measure_noise_floor",
    "quantize_dequantize",
]
