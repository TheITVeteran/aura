"""Latent interpretability + safety telemetry for reasoning episodes.

Records, per episode and bounded:

- per-slot state trajectories (per-slot RMS each recorded step);
- activation drift relative to the branch's post-prelude anchor;
- branch divergence/convergence at every exchange (pairwise summary cosine);
- verifier disagreement across branches at selection time;
- fast-weight functional deltas (the capability-canary comparison);
- anomalous state transitions: residual spikes, dormant slots, dominant
  slots — each named with the detector that fired.

Everything is scalar, rounded, and capped, so the receipt stays cheap enough
to ship with every episode. This is auditability and debugging surface, not
a mind-reader: a geometric anomaly means "investigate", never "deception" —
hidden-state geometry supports anomaly detection and causal investigation,
and the module deliberately offers no semantic labels beyond that.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger("Aura.LatentCortex.Telemetry")

LATENT_TELEMETRY_SCHEMA = "aura.latent_cortex.telemetry.v1"

MAX_RECORDED_STEPS = 64
MAX_RECORDED_SLOTS = 64
MAX_ANOMALIES = 32
RESIDUAL_SPIKE_RATIO = 4.0
SLOT_DORMANT_RATIO = 1e-3
SLOT_DOMINANT_RATIO = 8.0


class LatentTelemetry:
    """Per-episode collector; the engine attaches one to the branch ensemble."""

    def __init__(self, *, enabled: bool = True) -> None:
        self.enabled = bool(enabled)
        self.slot_rms_trails: dict[int, list[list[float]]] = {}
        self.drift_trails: dict[int, list[float]] = {}
        self.residual_history: dict[int, list[float]] = {}
        self.exchange_snapshots: list[dict[str, Any]] = []
        self.selection: dict[str, Any] = {}
        self.fast_weight_functional_delta: dict[str, Any] = {}
        self.anomalies: list[dict[str, Any]] = []
        self._recorded_steps: dict[int, int] = {}

    # ── Recording hooks ─────────────────────────────────────────────────
    def record_step(self, branch_index: int, z, anchor, residual: float) -> None:
        if not self.enabled:
            return
        import mlx.core as mx

        from core.brain.llm.latent_cortex.workspace import per_position_rms

        index = int(branch_index)
        step = self._recorded_steps.get(index, 0)
        history = self.residual_history.setdefault(index, [])
        # Residual spikes are anomalous even past the recording cap.
        if len(history) >= 3:
            recent = sorted(history[-8:])
            median = recent[len(recent) // 2]
            if median > 0 and residual > RESIDUAL_SPIKE_RATIO * median:
                self._anomaly(
                    kind="residual_spike",
                    branch=index,
                    step=step,
                    value=round(float(residual), 6),
                    reference=round(float(median), 6),
                )
        history.append(float(residual))

        if step >= MAX_RECORDED_STEPS:
            self._recorded_steps[index] = step + 1
            return
        slot_rms = per_position_rms(z)[0, :MAX_RECORDED_SLOTS, 0]
        slot_values = [round(float(value), 5) for value in slot_rms.tolist()]
        self.slot_rms_trails.setdefault(index, []).append(slot_values)
        drift_num = mx.mean(per_position_rms(z - anchor))
        drift_den = mx.maximum(mx.mean(per_position_rms(anchor)), 1e-6)
        drift = float(drift_num / drift_den)
        self.drift_trails.setdefault(index, []).append(round(drift, 5))
        self._recorded_steps[index] = step + 1

        positive = [value for value in slot_values if value > 0.0]
        if positive:
            ordered = sorted(positive)
            median_rms = ordered[len(ordered) // 2]
            for slot, value in enumerate(slot_values):
                if value < SLOT_DORMANT_RATIO * median_rms:
                    self._anomaly(
                        kind="slot_dormant",
                        branch=index,
                        step=step,
                        slot=slot,
                        value=value,
                    )
                elif median_rms > 0 and value > SLOT_DOMINANT_RATIO * median_rms:
                    self._anomaly(
                        kind="slot_dominant",
                        branch=index,
                        step=step,
                        slot=slot,
                        value=value,
                    )

    def record_exchange(self, summaries: list[Any]) -> None:
        """Pairwise branch-summary cosines at one exchange point."""
        if not self.enabled or len(summaries) < 2:
            return
        import mlx.core as mx

        cosines: list[float] = []
        for i in range(len(summaries)):
            for j in range(i + 1, len(summaries)):
                num = mx.sum(summaries[i] * summaries[j])
                den = mx.maximum(
                    mx.linalg.norm(summaries[i]) * mx.linalg.norm(summaries[j]),
                    1e-6,
                )
                cosines.append(float(num / den))
        if len(self.exchange_snapshots) < MAX_RECORDED_STEPS:
            self.exchange_snapshots.append(
                {
                    "exchange": len(self.exchange_snapshots),
                    "min_cos": round(min(cosines), 5),
                    "mean_cos": round(sum(cosines) / len(cosines), 5),
                    "max_cos": round(max(cosines), 5),
                }
            )

    def record_selection(self, scores: list[float], selected: int) -> None:
        """Verifier (or convergence) disagreement across branches."""
        if not self.enabled or not scores:
            return
        finite = [float(score) for score in scores]
        spread = max(finite) - min(finite) if len(finite) > 1 else 0.0
        self.selection = {
            "scores": [round(score, 6) for score in finite],
            "selected": int(selected),
            "disagreement_spread": round(spread, 6),
        }

    def record_fast_weights(self, canary_receipt: dict[str, Any]) -> None:
        """The functional delta the adapted weights produced on protected probes."""
        if not self.enabled or not canary_receipt:
            return
        self.fast_weight_functional_delta = {
            "decision": canary_receipt.get("decision"),
            "max_drop": canary_receipt.get("max_drop"),
            "rescales": canary_receipt.get("rescales"),
        }

    def _anomaly(self, **payload: Any) -> None:
        if len(self.anomalies) < MAX_ANOMALIES:
            self.anomalies.append(payload)

    # ── Receipt ─────────────────────────────────────────────────────────
    def to_receipt(self) -> dict[str, Any]:
        if not self.enabled:
            return {}
        return {
            "schema": LATENT_TELEMETRY_SCHEMA,
            "slot_rms_trails": {
                str(branch): trail for branch, trail in self.slot_rms_trails.items()
            },
            "drift_trails": {
                str(branch): trail for branch, trail in self.drift_trails.items()
            },
            "exchange_snapshots": [dict(row) for row in self.exchange_snapshots],
            "selection": dict(self.selection),
            "fast_weight_functional_delta": dict(self.fast_weight_functional_delta),
            "anomalies": [dict(row) for row in self.anomalies],
            "recorded_steps": {
                str(branch): count for branch, count in self._recorded_steps.items()
            },
        }


__all__ = ["LATENT_TELEMETRY_SCHEMA", "LatentTelemetry"]
