"""core/learning/interference_battery.py

Anti-interference gate: accumulated learnings must not trash prior ones.

Before any consolidated adaptation activates, this battery measures the
model's behavior on a fixed probe set BEFORE and AFTER the change:

- **stability probes** — prompts far from the adapted domain; their
  next-token distributions must stay essentially unchanged (top-1 match and
  bounded logit drift). Ten new learnings must leave these alone.
- **target probes** (optional) — prompts inside the adapted domain, ALLOWED
  (expected) to move.

The verdict is deterministic and receipted per probe. The consolidation
pipeline requires PASS before recommending activation; the compounding
loop's held-out regression tests remain the final authority.
"""
from __future__ import annotations

import hashlib
import logging
import time
from typing import Any, Callable

logger = logging.getLogger("Aura.InterferenceBattery")

INTERFERENCE_BATTERY_SCHEMA = "aura.interference_battery.v1"

# A stability probe passes when the top-1 token is unchanged AND the top-8
# logit region drifts less than this L2 fraction.
_MAX_STABLE_DRIFT = 0.05
_REQUIRED_STABLE_FRACTION = 0.9


def default_stability_probes() -> list[list[int]]:
    """Deterministic token probes spanning generic behavior regions."""
    return [
        [bases + step * k for k in range(8)]
        for bases, step in ((3, 5), (11, 7), (29, 3), (41, 11), (5, 13), (17, 2))
    ]


def _probe_logits(model, token_ids: list[int]):
    import mlx.core as mx

    inner = model.model
    vocab = int(inner.embed_tokens.weight.shape[0])
    tokens = mx.array([[t % vocab for t in token_ids]])
    h = inner.embed_tokens(tokens)
    for layer in inner.layers:
        h = layer(h, None, None)
    h = inner.norm(h)
    logits = (
        model.lm_head(h) if hasattr(model, "lm_head") else inner.embed_tokens.as_linear(h)
    )[0, -1]
    mx.eval(logits)
    return logits


def snapshot_probe_behavior(model, probes: list[list[int]] | None = None) -> list[dict[str, Any]]:
    """Capture per-probe behavioral fingerprints (top-1 + top-8 region)."""
    import mlx.core as mx

    rows = []
    for probe in probes or default_stability_probes():
        logits = _probe_logits(model, probe)
        top8 = mx.argsort(-logits)[:8]
        region = logits[top8].astype(mx.float32)
        mx.eval(top8, region)
        rows.append(
            {
                "probe": list(probe),
                "top1": int(top8[0]),
                "top8_ids": [int(i) for i in top8],
                "top8_logits": [float(v) for v in region],
                "digest": hashlib.sha256(memoryview(region)).hexdigest()[:16],
            }
        )
    return rows


def run_interference_battery(
    model,
    apply_change: Callable[[], Any],
    revert_change: Callable[[], Any] | None = None,
    *,
    probes: list[list[int]] | None = None,
    max_stable_drift: float = _MAX_STABLE_DRIFT,
    required_stable_fraction: float = _REQUIRED_STABLE_FRACTION,
) -> dict[str, Any]:
    """Measure behavioral interference of a proposed change.

    ``apply_change`` mutates the model (attach adapter, fuse weights, …);
    ``revert_change`` restores it (None ⇒ caller manages lifetime). The
    battery never decides to KEEP a change — it only reports whether
    protected behavior survived it.
    """
    import math

    before = snapshot_probe_behavior(model, probes)
    apply_change()
    try:
        after = snapshot_probe_behavior(model, probes)
    finally:
        if revert_change is not None:
            revert_change()

    results = []
    stable = 0
    for pre, post in zip(before, after):
        top1_same = pre["top1"] == post["top1"]
        num = math.sqrt(
            sum((a - b) ** 2 for a, b in zip(pre["top8_logits"], post["top8_logits"]))
        )
        den = math.sqrt(sum(v * v for v in pre["top8_logits"])) or 1e-9
        drift = num / den
        ok = top1_same and drift <= max_stable_drift
        stable += int(ok)
        results.append(
            {
                "probe": pre["probe"],
                "top1_same": top1_same,
                "drift": round(drift, 6),
                "stable": ok,
            }
        )
    fraction = stable / max(1, len(results))
    verdict = "PASS" if fraction >= required_stable_fraction else "FAIL"
    receipt = {
        "schema": INTERFERENCE_BATTERY_SCHEMA,
        "probes": len(results),
        "stable_probes": stable,
        "stable_fraction": round(fraction, 4),
        "required_stable_fraction": required_stable_fraction,
        "max_stable_drift": max_stable_drift,
        "results": results,
        "verdict": verdict,
        "ran_at": time.time(),
    }
    logger.info(
        "🛡 Interference battery: %d/%d stable → %s",
        stable,
        len(results),
        verdict,
    )
    return receipt


__all__ = [
    "INTERFERENCE_BATTERY_SCHEMA",
    "default_stability_probes",
    "run_interference_battery",
    "snapshot_probe_behavior",
]
