"""Wrap-vs-load effective-weight parity for expert adapters (remainder 4).

The 2026-07-08 specialist proof left one open observation: the sealed gate
(fresh ``mlx_lm.load(base, adapter_path=…)``) scored the modular specialist
0.50 on-domain while the worker's hot-attach onto the resident model scored
0.312 — routing (AURA_EXPERT_LORA_ROUTING) was blocked on explaining that
gap. This proof settles the mechanism half decisively on the real 4-bit
reflex model: a synthetic non-identity adapter (random A AND B, so the probe
has power — it moves final-token logits by tens of units) produces
BIT-IDENTICAL forward logits through both paths, and detach restores base
logits exactly. The effective weights are the same; the historical gap was
harness-side (the gate's bare-prompt eval vs the worker generation path's
chat template/sampling), not a weight-application defect.

Marked ``model`` + ``live``: needs the cached 4-bit 1.5B and ~7GB of
transient RAM — excluded from the offline chunks, run on demand:

    pytest tests/test_expert_adapter_parity.py -m model --no-header -q
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

pytestmark = [pytest.mark.model, pytest.mark.live]

BASE = "mlx-community/Qwen2.5-1.5B-Instruct-4bit"
LORA_PARAMS = {"rank": 8, "scale": 20.0, "dropout": 0.0}
NUM_LAYERS = 8
PROMPT = "Compute 17 mod 5 and explain the congruence briefly."


def _load_stack():
    mx = pytest.importorskip("mlx.core")
    pytest.importorskip("mlx_lm")
    import numpy as np
    from mlx.utils import tree_flatten
    from mlx_lm import load
    from mlx_lm.tuner.utils import linear_to_lora_layers

    return mx, np, tree_flatten, load, linear_to_lora_layers


def _forward_logits(mx, np, model, tokenizer):
    tokens = tokenizer.encode(PROMPT)
    logits = model(mx.array([tokens]))
    mx.eval(logits)
    return np.array(logits[0, -1, :].astype(mx.float32))


def _build_synthetic_adapter(mx, np, tree_flatten, load, linear_to_lora_layers,
                             adapter_dir: Path) -> None:
    model, _tokenizer = load(BASE)
    linear_to_lora_layers(model, NUM_LAYERS, dict(LORA_PARAMS))
    rng = np.random.default_rng(42)
    updates = {}
    for name, value in tree_flatten(model.trainable_parameters()):
        if "lora_a" in name or "lora_b" in name:
            updates[name] = mx.array(
                rng.standard_normal(value.shape).astype(np.float32) * 0.05
            )
    adapter_dir.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(str(adapter_dir / "adapters.safetensors"), updates)
    (adapter_dir / "adapter_config.json").write_text(
        json.dumps(
            {
                "fine_tune_type": "lora",
                "num_layers": NUM_LAYERS,
                "lora_parameters": LORA_PARAMS,
            }
        ),
        encoding="utf-8",
    )
    del model


def test_wrap_and_load_paths_produce_identical_logits(tmp_path):
    mx, np, tree_flatten, load, linear_to_lora_layers = _load_stack()
    from core.brain.llm.mlx_worker import (
        _attach_expert_adapter,
        _detach_expert_adapter,
    )

    adapter_dir = tmp_path / "adapter"
    _build_synthetic_adapter(
        mx, np, tree_flatten, load, linear_to_lora_layers, adapter_dir
    )

    # Path A: the gate's fresh load with adapter_path.
    model_a, tokenizer = load(BASE, adapter_path=str(adapter_dir))
    logits_load = _forward_logits(mx, np, model_a, tokenizer)
    del model_a

    # Path B: the worker's hot-attach onto an already-loaded base.
    model_b, tokenizer_b = load(BASE)
    logits_base = _forward_logits(mx, np, model_b, tokenizer_b)
    wrapped = _attach_expert_adapter(model_b, str(adapter_dir))
    logits_wrap = _forward_logits(mx, np, model_b, tokenizer_b)
    _detach_expert_adapter(model_b, wrapped)
    logits_detached = _forward_logits(mx, np, model_b, tokenizer_b)

    adapter_effect = float(np.abs(logits_load - logits_base).max())
    path_gap = float(np.abs(logits_load - logits_wrap).max())
    detach_gap = float(np.abs(logits_detached - logits_base).max())

    assert wrapped, "hot-attach wrapped no layers — probe has no subject"
    assert adapter_effect > 1.0, (
        "synthetic adapter left logits unchanged — the probe has no power"
    )
    assert path_gap < 1e-4, (
        f"wrap-vs-load effective weights DIVERGED: max|Δ|={path_gap:.6f}"
    )
    assert detach_gap < 1e-6, (
        f"detach did not restore base logits: max|Δ|={detach_gap:.6f}"
    )
