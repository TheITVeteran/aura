"""core/learning/model_merge.py — transplant Aura's personality onto a new base, in minutes.

The problem this solves: Aura's serving weights took ~a week to train (personality +
architecture), so "just swap to a reasoning-distilled base (QwQ / R1-Distill-Qwen)" would
throw that week away and cost another week of downtime. It doesn't have to. Because the
reasoning-distilled Qwen models and Aura's model are BOTH fine-tunes of the same ancestor
(Qwen2.5), the personality is recoverable as a *weight delta* and can be transplanted onto
the reasoning base by arithmetic — no retraining, minutes not a week:

    personality_delta = Aura_weights − Qwen2.5_base
    merged            = reasoning_base + λ · personality_delta            (task arithmetic)

This module is that arithmetic, with the three established methods:

* **task arithmetic** (Ilharco et al. 2023) — add scaled task vectors.
* **TIES** (Yadav et al. 2023) — trim each delta to its largest entries, elect a sign per
  parameter, and merge only the entries that agree — reduces interference between deltas.
* **DARE** (Yu et al. 2024) — randomly drop most of each delta and rescale the survivors;
  deltas are highly redundant, so this preserves the effect while cutting interference.

Honest boundaries:
* The math is exact and unit-tested (below) on real tensors. The GB-scale run is local
  compute on your hardware (load → merge per-tensor → save), and a merged model MUST be
  eval-gated through the RSI gauntlet before it becomes the serving model — merging can
  degrade either parent, so you verify before you promote.
* Merge the FP16/FP32 weights (or the LoRA adapter delta), never the 4-bit quantized
  artifact — you re-quantize after. bf16 needs a one-line cast first (numpy has no bf16).
* This is also the cheap "merge specialists" lever: a math LoRA + a code LoRA + the base.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

logger = logging.getLogger("Aura.ModelMerge")

StateDict = dict[str, np.ndarray]


# ---------------------------------------------------------------------------
# Core arithmetic (exact, unit-tested). Operates on aligned state dicts.
# ---------------------------------------------------------------------------
def task_vector(base: StateDict, finetuned: StateDict) -> StateDict:
    """The delta a fine-tune applied: ``finetuned − base`` per shared parameter."""
    return {
        k: (finetuned[k].astype(np.float32) - base[k].astype(np.float32))
        for k in base
        if k in finetuned and finetuned[k].shape == base[k].shape
    }


def _weights_for(deltas: Sequence[StateDict], weights: Optional[Sequence[float]]) -> list[float]:
    if weights is None:
        return [1.0] * len(deltas)
    if len(weights) != len(deltas):
        raise ValueError("weights length must match number of deltas")
    return [float(w) for w in weights]


def linear_merge(
    base: StateDict, deltas: Sequence[StateDict], *, weights: Optional[Sequence[float]] = None
) -> StateDict:
    """Plain task arithmetic: ``base + Σ wᵢ·δᵢ``."""
    ws = _weights_for(deltas, weights)
    out: StateDict = {}
    for k, b in base.items():
        acc = b.astype(np.float32).copy()
        for w, d in zip(ws, deltas):
            if k in d:
                acc += w * d[k]
        out[k] = acc.astype(b.dtype)
    return out


def ties_merge(
    base: StateDict,
    deltas: Sequence[StateDict],
    *,
    density: float = 0.2,
    weights: Optional[Sequence[float]] = None,
) -> StateDict:
    """TIES-merge: trim → elect sign → disjoint-average. ``density`` ∈ (0,1] kept per delta."""
    if not 0.0 < density <= 1.0:
        raise ValueError("density must be in (0, 1]")
    ws = _weights_for(deltas, weights)
    out: StateDict = {}
    for k, b in base.items():
        b32 = b.astype(np.float32)
        present = [(w, d[k].astype(np.float32)) for w, d in zip(ws, deltas) if k in d]
        if not present:
            out[k] = b
            continue
        trimmed: list[tuple[float, np.ndarray]] = []
        for w, d in present:
            flat = d.ravel()
            if flat.size and density < 1.0:
                keep = max(1, int(round(density * flat.size)))
                thresh = np.partition(np.abs(flat), -keep)[-keep]
                d = np.where(np.abs(d) >= thresh, d, 0.0)
            trimmed.append((w, d))
        # Elect a sign per parameter by the sign of the weighted sum of magnitudes.
        agg_sign = np.sign(sum(w * t for w, t in trimmed))
        num = np.zeros_like(b32)
        den = np.zeros_like(b32)
        for w, t in trimmed:
            agree = (np.sign(t) == agg_sign) & (t != 0.0)
            num += np.where(agree, w * t, 0.0)
            den += np.where(agree, 1.0, 0.0)
        merged_delta = np.where(den > 0, num / np.maximum(den, 1.0), 0.0)
        out[k] = (b32 + merged_delta).astype(b.dtype)
    return out


def dare_merge(
    base: StateDict,
    deltas: Sequence[StateDict],
    *,
    drop: float = 0.9,
    weights: Optional[Sequence[float]] = None,
    seed: int = 0,
) -> StateDict:
    """DARE-merge: Bernoulli-drop ``drop`` of each delta, rescale survivors by 1/(1−drop)."""
    if not 0.0 <= drop < 1.0:
        raise ValueError("drop must be in [0, 1)")
    ws = _weights_for(deltas, weights)
    rng = np.random.default_rng(seed)
    keep = 1.0 - drop
    out: StateDict = {}
    for k, b in base.items():
        acc = b.astype(np.float32).copy()
        for w, d in zip(ws, deltas):
            if k not in d:
                continue
            delta = d[k].astype(np.float32)
            mask = rng.random(delta.shape) >= drop
            acc += w * np.where(mask, delta / keep, 0.0)
        out[k] = acc.astype(b.dtype)
    return out


def merge_state_dicts(
    base: StateDict,
    finetunes: dict[str, StateDict],
    *,
    method: str = "ties",
    weights: Optional[Sequence[float]] = None,
    density: float = 0.2,
    drop: float = 0.9,
    seed: int = 0,
) -> StateDict:
    """Merge ``finetunes`` (name→state dict) onto ``base`` by the chosen method."""
    deltas = [task_vector(base, ft) for ft in finetunes.values()]
    if method == "linear":
        return linear_merge(base, deltas, weights=weights)
    if method == "ties":
        return ties_merge(base, deltas, density=density, weights=weights)
    if method == "dare":
        return dare_merge(base, deltas, drop=drop, weights=weights, seed=seed)
    raise ValueError(f"unknown merge method: {method!r} (use linear|ties|dare)")


def transplant_delta(
    common_base: StateDict, finetuned: StateDict, onto: StateDict, *, scale: float = 1.0
) -> StateDict:
    """Aura's exact case: move a fine-tune's delta onto a DIFFERENT base.

    ``onto + scale·(finetuned − common_base)`` — e.g. apply the personality delta
    (Aura − Qwen2.5) onto a reasoning base (QwQ / R1-Distill, also a Qwen2.5 descendant).
    """
    delta = task_vector(common_base, finetuned)
    return linear_merge(onto, [delta], weights=[scale])


# ---------------------------------------------------------------------------
# Safetensors I/O (tested round-trip). Loads FP16/FP32; bf16/quantized need a cast.
# ---------------------------------------------------------------------------
def load_state_dict(path: str | Path) -> StateDict:
    """Load a .safetensors file, or all shards in a directory, into one state dict."""
    from safetensors.numpy import load_file

    p = Path(path)
    files = sorted(p.glob("*.safetensors")) if p.is_dir() else [p]
    if not files:
        raise FileNotFoundError(f"no .safetensors found at {p}")
    state: StateDict = {}
    for f in files:
        state.update(load_file(str(f)))
    return state


def save_state_dict(state: StateDict, path: str | Path) -> None:
    from safetensors.numpy import save_file

    save_file(state, str(path))


def merge_model_dirs(
    base_dir: str | Path,
    finetune_dirs: dict[str, str | Path],
    out_dir: str | Path,
    *,
    method: str = "ties",
    weights: Optional[Sequence[float]] = None,
    density: float = 0.2,
    drop: float = 0.9,
) -> dict[str, Any]:
    """Load → merge → save a merged model, copying config/tokenizer alongside.

    For 72B-scale this holds the tensors in RAM; on a large-unified-memory Mac that is
    fine, otherwise merge the LoRA *adapters* (tiny) instead of the full bases. The output
    is NOT yet the serving model — run it through the RSI gauntlet / live_learner benchmark
    and only promote it if it wins.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    base = load_state_dict(base_dir)
    finetunes = {name: load_state_dict(d) for name, d in finetune_dirs.items()}
    merged = merge_state_dicts(
        base, finetunes, method=method, weights=weights, density=density, drop=drop
    )
    save_state_dict(merged, out / "model.safetensors")
    # Copy non-weight artifacts (tokenizer, config) from the base so the model loads.
    copied = []
    base_path = Path(base_dir)
    if base_path.is_dir():
        for art in base_path.iterdir():
            if art.suffix in {".json", ".model"} or art.name.startswith("tokenizer"):
                (out / art.name).write_bytes(art.read_bytes())
                copied.append(art.name)
    manifest = {
        "method": method,
        "base": str(base_dir),
        "finetunes": {k: str(v) for k, v in finetune_dirs.items()},
        "weights": list(weights) if weights else None,
        "tensors": len(merged),
        "copied_artifacts": copied,
        "note": "EVAL-GATE through the RSI gauntlet before promoting to the serving model.",
    }
    (out / "merge_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    logger.info("🧬 Merged %d tensors via %s → %s", len(merged), method, out)
    return manifest
