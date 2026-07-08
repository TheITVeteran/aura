#!/usr/bin/env python3
"""tools/heldout_eval.py — grade ONE model against the sealed held-out battery.

This is the process half of the weight-learning promotion gate. It is run as a
SUBPROCESS by the compounding driver (and by hand by auditors) so that exactly
one extra model is ever in memory: the driver evaluates incumbent and candidate
sequentially instead of loading both beside the serving model.

The battery is regenerated from (version, seed, size) — see
core/learning/heldout_battery.py — so a stranger can re-run this command and
get the identical task set. Decoding is greedy (temperature 0) and token-capped
so a run is deterministic and bounded.

Usage:
  python tools/heldout_eval.py --model <path-or-dir> [--adapter-path <dir>]
      [--seed 0] [--size 40] [--max-tokens 256] --output report.json

The report includes per-domain accuracy, the battery manifest (set hash), and
a content hash of the raw responses, which are dumped alongside the report for
audit (report.json → report.responses.jsonl).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.learning.heldout_battery import (  # noqa: E402
    BatterySpec,
    battery_manifest,
    generate_battery,
    grade_battery,
)


def _build_prompt(tokenizer, user_prompt: str) -> str:
    """Use the model's chat template when it has one (instruct models)."""
    apply = getattr(tokenizer, "apply_chat_template", None)
    if callable(apply):
        try:
            return apply(
                [{"role": "user", "content": user_prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
        except (TypeError, ValueError):
            pass
    return user_prompt


def run_eval(args: argparse.Namespace) -> dict:
    from mlx_lm import generate, load

    spec = BatterySpec(seed=args.seed, size=args.size)
    domains = {d.strip() for d in str(getattr(args, "domains", "") or "").split(",") if d.strip()}
    if domains:
        # Domain-concentrated battery for specialist gates: deterministically
        # oversample the same seeded stream, keep only the wanted domains,
        # slice to the requested size. Same seal discipline — the set hash in
        # the manifest covers exactly the tasks graded.
        pool = generate_battery(BatterySpec(seed=args.seed, size=args.size * 10))
        tasks = [t for t in pool if t.domain in domains][: args.size]
        if not tasks:
            raise SystemExit(f"no battery tasks for domains={sorted(domains)}")
    else:
        tasks = generate_battery(spec)

    load_kwargs = {}
    if args.adapter_path:
        load_kwargs["adapter_path"] = args.adapter_path
    started = time.time()
    model, tokenizer = load(args.model, **load_kwargs)
    load_s = time.time() - started

    # Greedy sampler → deterministic decode. Older mlx-lm versions default to
    # greedy already; passing an explicit sampler pins the behavior.
    gen_kwargs: dict = {"max_tokens": args.max_tokens, "verbose": False}
    try:
        from mlx_lm.sample_utils import make_sampler

        gen_kwargs["sampler"] = make_sampler(temp=0.0)
    except ImportError:
        pass

    responses: dict[str, str] = {}
    gen_started = time.time()
    for task in tasks:
        prompt = _build_prompt(tokenizer, task.prompt)
        try:
            out = generate(model, tokenizer, prompt=prompt, **gen_kwargs)
        except TypeError:
            # older mlx-lm signature without sampler kwarg
            gen_kwargs.pop("sampler", None)
            out = generate(model, tokenizer, prompt=prompt, **gen_kwargs)
        responses[task.task_id] = out if isinstance(out, str) else str(out)
    gen_s = time.time() - gen_started

    result = grade_battery(spec, tasks, responses)

    import hashlib

    responses_blob = json.dumps(responses, sort_keys=True, ensure_ascii=False)
    report = {
        "schema_version": 1,
        "tool": "heldout_eval",
        "model": str(args.model),
        "adapter_path": str(args.adapter_path or ""),
        "battery": battery_manifest(spec, tasks),
        "result": result.to_dict(),
        "accuracy": result.accuracy,
        "responses_sha256": hashlib.sha256(responses_blob.encode()).hexdigest(),
        "timing": {"load_s": round(load_s, 2), "generate_s": round(gen_s, 2)},
        "max_tokens": args.max_tokens,
        "created_at": time.time(),
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    responses_path = output.with_suffix(".responses.jsonl")
    with responses_path.open("w", encoding="utf-8") as fh:
        for task in tasks:
            fh.write(
                json.dumps(
                    {"task_id": task.task_id, "response": responses[task.task_id]},
                    ensure_ascii=False,
                )
                + "\n"
            )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="model directory (base or fused)")
    parser.add_argument("--adapter-path", default="", help="optional LoRA adapter directory")
    parser.add_argument("--seed", type=int, default=0, help="battery seed")
    parser.add_argument("--size", type=int, default=40, help="number of tasks")
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument(
        "--domains",
        default="",
        help="comma-separated domain filter (domain-concentrated specialist battery)",
    )
    parser.add_argument("--output", required=True, help="report JSON path")
    args = parser.parse_args()

    report = run_eval(args)
    acc = report["result"]
    print(
        f"[heldout_eval] {report['battery']['battery_id']} "
        f"model={Path(args.model).name} adapter={Path(args.adapter_path).name if args.adapter_path else '-'} "
        f"accuracy={acc['correct']}/{acc['total']} ({report['accuracy']:.1%})"
    )
    for domain, bucket in sorted(acc["per_domain"].items()):
        print(f"  {domain:>18}: {bucket['correct']}/{bucket['total']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
