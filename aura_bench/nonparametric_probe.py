"""Real end-to-end probe: does non-parametric memory inject knowledge the weights lack?

Uses FICTIONAL facts the base model cannot know, so any correct recall is *purely* from the
datastore — a clean proof that capacity was added, not coincidence. Tests next-token recall
AND full end-to-end generation, bare vs interpolated, plus a control the model already knows.

Run: python -m aura_bench.nonparametric_probe [model_path]
"""
from __future__ import annotations

import sys

import numpy as np

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
CONTROL = ("The capital of France is", "Paris")


def _gold_token(tok, answer: str) -> int:
    specials = set(getattr(tok, "all_special_ids", []) or [])
    ids = [i for i in tok.encode(" " + answer) if i not in specials]
    return ids[0]


def _run_probe(model_path: str) -> int:
    import mlx.core as mx
    from mlx_lm import load

    from core.brain.nonparametric_generation import MLXEncoder, generate_with_memory
    from core.brain.nonparametric_ingest import NonParametricIngestor
    from core.brain.nonparametric_memory import NonParametricMemory

    print(f"Loading {model_path} ...")
    model, tok = load(model_path)
    dim = int(model.args.hidden_size)

    def hidden_and_logits(prompt: str):
        ids = mx.array([tok.encode(prompt)])
        h = model.model(ids)
        logits = model(ids)
        return np.array(h[0, -1], dtype=np.float32), np.array(logits[0, -1], dtype=np.float32)

    def topk_probs(logits, k=50, include=None):
        idx = set(int(i) for i in np.argpartition(logits, -k)[-k:])
        if include is not None:
            idx.add(int(include))
        idx = np.array(sorted(idx))
        sub = logits[idx] - logits[idx].max()
        ex = np.exp(sub)
        ex /= ex.sum()
        return {int(t): float(p) for t, p in zip(idx, ex, strict=True)}

    import os as _os
    import tempfile
    from pathlib import Path

    probe_root = Path(tempfile.gettempdir()) / "aura_npm_probe"
    seen_path = Path(tempfile.gettempdir()) / "aura_npm_seen.json"
    for _f in (
        str(probe_root.with_suffix(".keys.npy")),
        str(probe_root.with_suffix(".meta.json")),
        str(seen_path),
    ):
        if _os.path.exists(_f):
            _os.remove(_f)
    mem = NonParametricMemory(dim=dim, path=str(probe_root), base_lambda=0.4, max_lambda=0.8)
    enc = MLXEncoder(model, tok)
    ing = NonParametricIngestor(mem, dedup_path=str(seen_path))
    positions = sum(ing.ingest_sequence(ctx, ans, enc) for ctx, ans in FACTS)
    print(f"Datastore built: {len(mem)} entries from {len(FACTS)} facts ({positions} positions).\n")

    def evaluate(name, pairs):
        bare_hits = interp_hits = 0
        for ctx, ans in pairs:
            gold = _gold_token(tok, ans)
            _, logits = hidden_and_logits(ctx)
            qkey = enc.encode_hidden(ctx)  # normalized, matches the datastore keys
            bare = topk_probs(logits, include=gold)
            blended = mem.interpolate(bare, qkey, k=4, temperature=2.0, phi=0.5, free_energy=0.9)
            bare_hits += int(max(bare, key=bare.get) == gold)
            interp_hits += int(max(blended, key=blended.get) == gold)
            print(f"  {'✓' if max(blended,key=blended.get)==gold else '✗'} [{name}] '{ans}': "
                  f"gold_p {bare.get(gold,0):.3f}→{blended.get(gold,0):.3f}")
        print(f"  → {name}: bare top1 {bare_hits}/{len(pairs)}, interpolated {interp_hits}/{len(pairs)}\n")
        return bare_hits, interp_hits, len(pairs)

    print("=== NEXT-TOKEN, EXACT contexts (lookup floor) ===")
    e_bare, e_int, e_n = evaluate("exact", FACTS)
    print("=== NEXT-TOKEN, PARAPHRASED contexts (semantic recall) ===")
    p_bare, p_int, p_n = evaluate("para", PARAPHRASES)

    print("=== GENERATION (end-to-end causal: does it GENERATE the fictional answer?) ===")
    gen_bare = gen_mem = 0
    gen_pairs = FACTS[:5]
    for ctx, ans in gen_pairs:
        bare = generate_with_memory(model, tok, ctx, mem, max_tokens=6, use_memory=False)
        withm = generate_with_memory(model, tok, ctx, mem, max_tokens=6, use_memory=True, phi=0.5, free_energy=0.9)
        gm = ans.lower() in withm.lower()
        gen_bare += int(ans.lower() in bare.lower())
        gen_mem += int(gm)
        print(f"  {'✓' if gm else '✗'} [{ans}] bare={bare!r}  mem={withm!r}")
    print(f"  → generation: bare {gen_bare}/{len(gen_pairs)}, +memory {gen_mem}/{len(gen_pairs)}\n")

    print("=== CONTROL (model already knows; memory must NOT corrupt) ===")
    gold = _gold_token(tok, CONTROL[1])
    _, logits = hidden_and_logits(CONTROL[0])
    qkey = enc.encode_hidden(CONTROL[0])
    bare = topk_probs(logits, include=gold)
    blended = mem.interpolate(bare, qkey, k=4, temperature=2.0, phi=0.5, free_energy=0.9)
    ctrl_ok = max(blended, key=blended.get) == gold
    print(f"  control preserved={ctrl_ok}\n")

    print("================= RESULT =================")
    print(f"next-token exact:      bare {e_bare}/{e_n}  →  +memory {e_int}/{e_n}")
    print(f"next-token paraphrase: bare {p_bare}/{p_n}  →  +memory {p_int}/{p_n}")
    print(f"GENERATION:            bare {gen_bare}/{len(gen_pairs)}  →  +memory {gen_mem}/{len(gen_pairs)}")
    print(f"control preserved: {ctrl_ok}")
    print("==========================================")
    return 0


def main() -> int:
    from core.runtime.model_lane_control import standalone_model_lane

    model_path = sys.argv[1] if len(sys.argv) > 1 else "models/Qwen2.5-7B-Instruct-4bit"
    with standalone_model_lane(
        owner_id="nonparametric-memory-probe",
        model_path=model_path,
        purpose="benchmark",
        preemptible=False,
        metadata={"tool": "aura_bench.nonparametric_probe"},
    ):
        return _run_probe(model_path)


if __name__ == "__main__":
    raise SystemExit(main())
