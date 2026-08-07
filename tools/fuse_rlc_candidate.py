#!/usr/bin/env python
"""Merge one recurrence adapter into a copy of the base weights.

This is deliberately a separate step from activation. The recurrence adapter
is a ScopedLoRALinear: at run time its delta applies at latent slot positions
and nowhere else. Folding it into the linear weights removes that scoping, so
the result is a different function on every ordinary token as well -- not a
repackaging of the same behavior. The fused model is therefore a CANDIDATE
that has to re-earn ordinary decode before it can replace anything, and this
tool never touches the resident or any pointer to it.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="resident base model dir")
    parser.add_argument("--adapter", required=True, help="checkpoint generation dir")
    parser.add_argument("--out", required=True, help="candidate model dir to create")
    args = parser.parse_args()

    base = Path(args.model)
    adapter_dir = Path(args.adapter)
    out = Path(args.out)

    if not (base / "config.json").exists():
        print(f"base model missing config.json: {base}", file=sys.stderr)
        return 2
    if not (adapter_dir / "adapter.safetensors").exists():
        print(f"adapter missing safetensors: {adapter_dir}", file=sys.stderr)
        return 2
    if out.exists():
        print(f"refusing to overwrite existing candidate: {out}", file=sys.stderr)
        return 2

    import mlx.core as mx
    from mlx.utils import tree_flatten, tree_unflatten
    from mlx_lm import load

    from core.runtime.mlx_memory_guard import mlx_memory_envelope

    with mlx_memory_envelope(fraction=0.40):
        model, tokenizer = load(str(base))
        adapter = mx.load(str(adapter_dir / "adapter.safetensors"))

        # lora_a/lora_b pairs fold as W + (b @ a).T scaled by the package's
        # own alpha/rank, matching how ScopedLoRALinear applies them live.
        manifest_path = adapter_dir.parent.parent / "recurrence_adapter_manifest.json"
        found = list(adapter_dir.parent.parent.rglob("recurrence_adapter_manifest.json"))
        if not manifest_path.exists() and found:
            manifest_path = found[0]
        if not manifest_path.exists():
            print("no recurrence adapter manifest; cannot fuse safely", file=sys.stderr)
            return 2
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        lora = manifest["lora"]
        rank = int(lora["rank"])
        scale = float(lora.get("alpha", rank)) / float(rank)

        params = dict(tree_flatten(model.parameters()))
        fused = 0
        for key in sorted(adapter):
            if not key.endswith(".lora_a"):
                continue
            stem = key[: -len(".lora_a")]
            b_key = f"{stem}.lora_b"
            weight_key = f"{stem}.weight"
            if b_key not in adapter or weight_key not in params:
                print(f"skipping unpaired adapter tensor: {stem}", file=sys.stderr)
                continue
            a = adapter[key].astype(mx.float32)
            b = adapter[b_key].astype(mx.float32)
            delta = (a @ b).T * scale
            weight = params[weight_key]
            if delta.shape != weight.shape:
                print(
                    f"delta shape {delta.shape} != weight {weight.shape} for {stem}",
                    file=sys.stderr,
                )
                return 2
            params[weight_key] = (weight.astype(mx.float32) + delta).astype(weight.dtype)
            fused += 1
        if fused == 0:
            print("no adapter tensors fused", file=sys.stderr)
            return 2

        model.update(tree_unflatten(list(params.items())))
        mx.eval(model.parameters())

        out.mkdir(parents=True, exist_ok=False)
        for name in ("config.json", "tokenizer.json", "tokenizer_config.json"):
            candidate = base / name
            if candidate.exists():
                shutil.copy2(candidate, out / name)
        for extra in base.glob("*.txt"):
            shutil.copy2(extra, out / extra.name)
        for extra in base.glob("*.model"):
            shutil.copy2(extra, out / extra.name)
        weights = dict(tree_flatten(model.parameters()))
        mx.save_safetensors(str(out / "model.safetensors"), weights)
        (out / "FUSION_PROVENANCE.json").write_text(
            json.dumps(
                {
                    "schema": "aura.rlc_fused_candidate.v1",
                    "base_model": str(base),
                    "adapter": str(adapter_dir),
                    "fused_tensors": fused,
                    "scale": scale,
                    "rank": rank,
                    "scoping_note": (
                        "the adapter applied at latent slot positions only; "
                        "fusion applies it everywhere, so this is a different "
                        "function on ordinary tokens and must re-earn them"
                    ),
                },
                indent=1,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        del tokenizer
    print(f"fused {fused} tensors into {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
