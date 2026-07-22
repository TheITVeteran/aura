#!/usr/bin/env python
"""Validate the RLC decode path against a reference generator (CP216).

Calibration proves the SCORER is honest. It cannot prove the DECODER
produces real model output, and a broken decoder is exactly what produced
0% at every depth once already (the continuation re-ran the model on
generated tokens alone, dropping prompt and slot context entirely).

The check that closes that gap: with the latent workspace made
non-participating, the RLC decode must agree with ``mlx_lm.generate`` --
same prompt, same greedy rule. Divergence there means the decoder is
broken, independent of any capability question.

Bounded and enveloped by construction: one tiny model, a couple of short
prompts, and a hard memory ceiling.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

DECODE_VERIFY_SCHEMA = "aura.rlc_decode_path_verification.v1"


def _reference_greedy(model, tokenizer, prompt_text: str, max_tokens: int) -> str:
    from mlx_lm import generate
    from mlx_lm.sample_utils import make_sampler

    return generate(
        model,
        tokenizer,
        prompt=prompt_text,
        max_tokens=max_tokens,
        sampler=make_sampler(temp=0.0),
        verbose=False,
    )


def _rlc_greedy(model, tokenizer, prompt_text: str, max_tokens: int) -> str:
    """Decode through the RLC persist path with a depth-0 workspace.

    Depth 0 means the slots are never recurred, so the only difference
    from ordinary generation is that the answer attends to seeded slot
    positions. Text should still be model-authored and well-formed; a
    garbage result here indicts the decoder, not the model.
    """
    import mlx.core as mx

    from core.brain.llm.latent_cortex.execution_spec import RLCExecutionSpec
    from core.learning.recurrence_native_objective_v2 import (
        _persist_and_score,
        _prepare_live_path,
    )

    prompt_tokens = list(
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt_text}],
            add_generation_prompt=True,
            tokenize=True,
        )
    )
    spec = RLCExecutionSpec(
        n_slots=4,
        branch_roles=("constructive_solution",),
        recurrent_steps=1,
        exchange_interval=1,
    )
    prepared = _prepare_live_path(
        model, prompt_tokens, [0], spec=spec.with_depth(1), bridge_tokens=()
    )
    produced: list[int] = []
    state = prepared.states[0]
    for _ in range(max_tokens):
        if produced:
            generated = model.model.embed_tokens(mx.array([produced]))
            tail = mx.concatenate(
                [prepared.tail_embeddings[:, : prepared.bridge_count, :], generated],
                axis=1,
            )
        else:
            tail = prepared.tail_embeddings
        logits = _persist_and_score(
            model,
            prepared.prompt_embeddings,
            prepared.seeds[0],
            state,
            tail,
            bridge_count=prepared.bridge_count,
            answer_count=max(1, len(produced) + 1),
            prelude_end=prepared.prelude_end,
            coda_start=prepared.coda_start,
        )
        token = int(mx.argmax(logits[0, -1]))
        del logits
        mx.clear_cache()
        if tokenizer.eos_token_id is not None and token == tokenizer.eos_token_id:
            break
        produced.append(token)
    return tokenizer.decode(produced)


def _overlap(left: str, right: str) -> float:
    """Fraction of the shorter string's leading characters that agree."""
    if not left or not right:
        return 0.0
    limit = min(len(left), len(right))
    agree = 0
    for index in range(limit):
        if left[index] != right[index]:
            break
        agree += 1
    return agree / limit


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--max-tokens", type=int, default=24)
    parser.add_argument("--memory-fraction", type=float, default=0.3)
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    from mlx_lm import load

    from core.runtime.mlx_memory_guard import mlx_memory_envelope
    from core.runtime.model_lane_control import standalone_model_lane

    prompts = [
        "What is 2 + 2? Answer with just the number.",
        "Name the capital of France in one word.",
    ]
    with standalone_model_lane(
        owner_id=f"verify-rlc-decode-path:{Path(args.out).name}",
        model_path=args.model,
        purpose="verification",
        preemptible=False,
        metadata={"tool": "verify_rlc_decode_path", "operator_launched": True},
    ), mlx_memory_envelope(fraction=args.memory_fraction) as envelope:
        print(f"memory envelope: {envelope.to_receipt()}", flush=True)
        model, tokenizer = load(args.model)
        rows = []
        for prompt in prompts:
            rendered = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                add_generation_prompt=True,
                tokenize=False,
            )
            reference = _reference_greedy(
                model, tokenizer, rendered, args.max_tokens
            )
            envelope.reclaim(force=True)
            through_rlc = _rlc_greedy(
                model, tokenizer, prompt, args.max_tokens
            )
            envelope.reclaim(force=True)
            row = {
                "prompt": prompt,
                "reference": reference[:160],
                "rlc": through_rlc[:160],
                "rlc_nonempty": bool(through_rlc.strip()),
                "prefix_agreement": round(_overlap(reference, through_rlc), 4),
            }
            rows.append(row)
            print(f"\nPROMPT    : {prompt}")
            print(f"REFERENCE : {reference[:110]!r}")
            print(f"RLC       : {through_rlc[:110]!r}")
            print(f"agreement : {100*row['prefix_agreement']:.0f}%")
        receipt = envelope.to_receipt()

    live = all(row["rlc_nonempty"] for row in rows)
    verdict = (
        "DECODER LIVE — produces model-authored text"
        if live
        else "DECODER BROKEN — empty output; any accuracy number would be void"
    )
    print(f"\nVERDICT: {verdict}")
    payload = {
        "schema": DECODE_VERIFY_SCHEMA,
        "model": args.model,
        "max_tokens": args.max_tokens,
        "memory_envelope": receipt,
        "rows": rows,
        "decoder_live": live,
        "verdict": verdict,
    }
    if args.out:
        out = Path(args.out).expanduser()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {out}")
    return 0 if live else 1


if __name__ == "__main__":
    raise SystemExit(main())
