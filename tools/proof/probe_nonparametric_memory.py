"""Opt-in end-to-end probe for non-parametric memory on a real local model.

Uses FICTIONAL facts the base model cannot know, so any correct next-token recall is
*purely* from the datastore — a clean proof that capacity was added, not coincidence.

Procedure (real model, real hidden states):
  1. Build a datastore: for each fact context, key = the model's last-token hidden state,
     value = the gold next token. (This is the ingestion seam, made real.)
  2. Test next-token prediction BARE vs INTERPOLATED on:
       - exact contexts (lookup floor),
       - PARAPHRASED contexts (the real test: representation-space recall, not string match),
       - a control prompt the model DOES know (must not be corrupted by fictional memory).

This is a proof tool, not a runtime entrypoint. It loads a local model and must
be run explicitly because doing so alongside Aura can exceed the desktop memory
budget. Reports a real number, or it does not count.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.runtime.model_lane_control import standalone_model_lane  # noqa: E402

# Fictional facts — guaranteed absent from the base model's weights.
FACTS = [
    ("In the Aetherium archives, the keeper of the seventh gate is named", "Tessaly"),
    ("The capital city of the floating realm of Vorth is", "Myrrhal"),
    ("The ancient dragon who guards Mount Skellig is called", "Brannoch"),
    ("In Zorblax mythology, the first star was born from the tears of", "Velreth"),
    ("The river that flows backward through Hollowmere is the", "Sethryn"),
    ("The currency used in the underground city of Drennim is the", "Groat"),
    ("The forbidden ninth note in the Tessmark scale is called", "Quorl"),
    ("The twin moons of the planet Ashkar are named Pellan and", "Vexis"),
]
# Paraphrases — same facts, reworded. Tests semantic (hidden-state) recall vs string lookup.
PARAPHRASES = [
    ("Who keeps the seventh gate of the Aetherium archives? Their name is", "Tessaly"),
    ("Vorth is a realm that floats in the sky; the name of its capital city is", "Myrrhal"),
    ("There is a dragon guarding Mount Skellig. People call it", "Brannoch"),
    ("In the myths of Zorblax, the very first star came from the tears of", "Velreth"),
    ("Through Hollowmere runs a river that flows in reverse, known as the", "Sethryn"),
    ("Down in the underground city of Drennim, people pay using a coin called the", "Groat"),
    ("The Tessmark musical scale has a forbidden ninth note named", "Quorl"),
    ("Ashkar is a planet with two moons; one is Pellan and the other is", "Vexis"),
]
# Control — a fact the base model DOES know. Memory must not corrupt it.
CONTROL = ("The capital of France is", "Paris")


def _gold_token(tok, answer: str) -> int:
    ids = tok.encode(" " + answer)
    # strip a leading BOS/special if present
    specials = set(getattr(tok, "all_special_ids", []) or [])
    ids = [i for i in ids if i not in specials]
    return ids[0]


def _run_probe(model_path: str) -> int:
    import mlx.core as mx
    from mlx_lm import load

    from core.brain.nonparametric_memory import NonParametricMemory

    print(f"Loading {model_path} ...")
    model, tok = load(model_path)
    dim = int(model.args.hidden_size)

    def hidden_and_logits(prompt: str):
        ids = mx.array([tok.encode(prompt)])
        h = model.model(ids)
        logits = model(ids)
        key = np.array(h[0, -1], dtype=np.float32)
        lg = np.array(logits[0, -1], dtype=np.float32)
        return key, lg

    def topk_probs(logits: np.ndarray, k: int = 50, include: int | None = None) -> dict[int, float]:
        idx = np.argpartition(logits, -k)[-k:]
        idx = set(int(i) for i in idx)
        if include is not None:
            idx.add(int(include))
        idx = np.array(sorted(idx))
        sub = logits[idx]
        sub = sub - sub.max()
        ex = np.exp(sub)
        ex /= ex.sum()
        return {int(t): float(p) for t, p in zip(idx, ex, strict=True)}

    # 1. Build the datastore from real hidden states (ingestion, made real).
    probe_path = Path(tempfile.gettempdir()) / "aura_nonparametric_memory_probe"
    mem = NonParametricMemory(dim=dim, path=probe_path, base_lambda=0.4, max_lambda=0.8)
    for ctx, ans in FACTS:
        key, _ = hidden_and_logits(ctx)
        mem.add(key, _gold_token(tok, ans), token=ans, weight=1.0)
    print(f"Datastore built: {len(mem)} fictional facts.\n")

    def evaluate(name, pairs):
        bare_hits = interp_hits = 0
        bare_gold = interp_gold = 0.0
        for ctx, ans in pairs:
            gold = _gold_token(tok, ans)
            key, logits = hidden_and_logits(ctx)
            bare = topk_probs(logits, include=gold)
            blended = mem.interpolate(bare, key, k=4, temperature=2.0, phi=0.5, free_energy=0.9)
            bare_top = max(bare, key=bare.get)
            interp_top = max(blended, key=blended.get)
            bare_hits += int(bare_top == gold)
            interp_hits += int(interp_top == gold)
            bare_gold += bare.get(gold, 0.0)
            interp_gold += blended.get(gold, 0.0)
            flag = "✓" if interp_top == gold else "✗"
            print(f"  {flag} [{name}] '{ans}': bare_top={tok.decode([bare_top])!r} "
                  f"interp_top={tok.decode([interp_top])!r} gold_p {bare.get(gold,0):.3f}→{blended.get(gold,0):.3f}")
        n = len(pairs)
        print(f"  → {name}: bare top1 {bare_hits}/{n}, interpolated {interp_hits}/{n}; "
              f"gold prob {bare_gold/n:.3f}→{interp_gold/n:.3f}\n")
        return bare_hits, interp_hits, n

    print("=== EXACT contexts (lookup floor) ===")
    e_bare, e_int, e_n = evaluate("exact", FACTS)
    print("=== PARAPHRASED contexts (real test: semantic recall) ===")
    p_bare, p_int, p_n = evaluate("para", PARAPHRASES)

    print("=== CONTROL (model already knows; memory must NOT corrupt) ===")
    gold = _gold_token(tok, CONTROL[1])
    key, logits = hidden_and_logits(CONTROL[0])
    bare = topk_probs(logits, include=gold)
    blended = mem.interpolate(bare, key, k=4, temperature=2.0, phi=0.5, free_energy=0.9)
    ctrl_ok = max(blended, key=blended.get) == gold
    print(f"  control '{CONTROL[1]}': bare_top={tok.decode([max(bare,key=bare.get)])!r} "
          f"interp_top={tok.decode([max(blended,key=blended.get)])!r} preserved={ctrl_ok}\n")

    print("================= RESULT =================")
    print(f"exact:      bare {e_bare}/{e_n}  →  +memory {e_int}/{e_n}")
    print(f"paraphrase: bare {p_bare}/{p_n}  →  +memory {p_int}/{p_n}")
    print(f"control preserved: {ctrl_ok}")
    print("==========================================")
    return 0


def main() -> int:
    model_path = sys.argv[1] if len(sys.argv) > 1 else "models/Qwen2.5-7B-Instruct-4bit"
    with standalone_model_lane(
        owner_id="nonparametric-memory-probe",
        model_path=model_path,
        purpose="benchmark",
        metadata={"tool": "probe_nonparametric_memory"},
    ):
        return _run_probe(model_path)


if __name__ == "__main__":
    raise SystemExit(main())
