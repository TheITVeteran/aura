#!/usr/bin/env python
"""Frozen-checkpoint execution-spec sweep for the Recursive Latent Cortex.

The 2026-08-06 directional campaign scored base vanilla 13/28 and base RLC
5/28 on the same frozen weights. That eight-point deficit exists before any
optimizer runs, so no adapter can be evaluated honestly until it is explained.
Two candidate causes were identified from the retained journal, and they are
separable only by execution, not by more analysis:

  1. Terminal-disposition language. Every recurrent episode injected a block
     ahead of the answer decode. On 24 of 28 tasks that block read "give only
     the best bounded answer ... disclose the unresolved part" -- an
     instruction to answer partially, delivered to the recurrent arms while
     the vanilla control received nothing at all.
  2. The recurrent rewrite itself perturbing the residual stream.

This sweep crosses those two factors on frozen weights. It performs no
optimizer update and awards no claim. It answers one question: can the
recurrent execution path reach parity with an ordinary decode on this
checkpoint? Until it can, training against that path is training into a hole.

All five arms share one model load -- they differ only in configuration --
which is what makes the whole sweep affordable.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

SWEEP_SCHEMA = "aura.rlc_reconciliation_sweep.v1"

# (arm, recurrent_steps or None for vanilla, terminal_instruction_policy)
ARMS: tuple[tuple[str, int | None, str], ...] = (
    ("vanilla", None, "applied"),
    ("rlc_asrun", 4, "applied"),
    ("rlc_nodisp", 4, "suppressed"),
    ("rlc_shallow", 1, "applied"),
    ("rlc_shallow_nodisp", 1, "suppressed"),
)


def _now() -> float:
    return time.time()


def _atomic_write(path: Path, payload: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(payload, encoding="utf-8")
    os.replace(tmp, path)


class Journal:
    """Append-only cell journal. Resumption replays it and skips committed work."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.done: set[tuple[str, str]] = set()
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    # A torn final line from a hard kill is not evidence.
                    continue
                if record.get("event") == "CELL":
                    self.done.add((record["arm"], record["task_id"]))

    def append(self, record: dict[str, Any]) -> None:
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    def cells(self) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        if not self.path.exists():
            return out
        for line in self.path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("event") == "CELL":
                out.append(record)
        return out


def _status(out_dir: Path, **fields: Any) -> None:
    path = out_dir / "status.json"
    body: dict[str, Any] = {}
    if path.exists():
        try:
            body = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            body = {}
    body.update(fields)
    body["heartbeat_unix"] = _now()
    body["pid"] = os.getpid()
    _atomic_write(path, json.dumps(body, indent=1, sort_keys=True) + "\n")


def _build_config(
    steps: int,
    n_slots: int,
    policy: str,
    max_tokens: int,
    decode_contract: str = "final_answer_v1",
):
    from core.brain.llm.latent_cortex.types import (
        BranchConfig,
        CortexConfig,
        RecurrenceConfig,
        WorkspaceConfig,
    )

    return CortexConfig(
        workspace=WorkspaceConfig(n_slots=n_slots, seed=0),
        recurrence=RecurrenceConfig(max_steps=steps, min_steps=steps),
        branches=BranchConfig(n_branches=2, exchange_interval=1),
        prelude_frac=0.25,
        coda_frac=0.25,
        decode_max_tokens=max_tokens,
        decode_temperature=0.0,
        decode_top_p=1.0,
        decode_bridge_policy="none",
        decode_incumbent_policy="latent",
        decode_contract=decode_contract,
        decode_contract_grace_tokens=320 if decode_contract != "none" else 0,
        terminal_instruction_policy=policy,
    )


def _render_prompt(tokenizer, task) -> list[int]:
    content = task.public.prompt
    return list(
        tokenizer.apply_chat_template(
            [{"role": "user", "content": content}],
            add_generation_prompt=True,
            tokenize=True,
        )
    )


def _render_prompt_text(tokenizer, task) -> str:
    content = task.public.prompt
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": content}],
        add_generation_prompt=True,
        tokenize=False,
    )


def _run_vanilla(model, tokenizer, rendered: str, max_tokens: int) -> str:
    """Ordinary greedy decode -- the control the recurrent arms must beat.

    It stops on the same rule the recurrent arms stop on
    (``decode_contract=final_answer_v1``). Letting the control run past its
    first complete answer is not neutral: in the 2026-08-06 campaign the
    free-running vanilla arm emitted a second FINAL_ANSWER marker on 12 of 28
    tasks and was format-rejected for it, while every recurrent arm was
    protected by the contract stop.
    """
    from mlx_lm import stream_generate

    from core.brain.llm.latent_cortex.answer_contract import is_contract_complete

    pieces: list[str] = []
    for response in stream_generate(
        model, tokenizer, prompt=rendered, max_tokens=max_tokens
    ):
        pieces.append(response.text)
        if "}" in response.text and is_contract_complete("".join(pieces)):
            break
    return "".join(pieces)


def _run_rlc(model, config, prompt_tokens: list[int], tokenizer) -> tuple[str, dict[str, Any]]:
    from core.brain.llm.latent_cortex.engine import LatentCortexEngine

    engine = LatentCortexEngine(model, config=config, tokenizer=tokenizer)
    result = engine.reason(token_ids=prompt_tokens)
    receipt = result.receipt.to_dict()
    text = getattr(result, "text", "") or ""
    if not text and tokenizer is not None:
        text = tokenizer.decode(list(result.tokens))
    return text, receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seed", type=int, default=20260807)
    parser.add_argument("--per-domain", type=int, default=4)
    parser.add_argument("--n-slots", type=int, default=16)
    parser.add_argument("--max-tokens", type=int, default=320)
    parser.add_argument("--memory-fraction", type=float, default=0.40)
    parser.add_argument("--max-wall-s", type=float, default=64_800.0)
    parser.add_argument(
        "--arms",
        default=",".join(a[0] for a in ARMS),
        help="comma-separated subset of arms to execute",
    )
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    from core.brain.llm.latent_cortex import frontier_tasks as ft

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    selected = [a for a in ARMS if a[0] in {v.strip() for v in args.arms.split(",")}]
    if not selected:
        print("no arms selected", file=sys.stderr)
        return 2

    # Tasks are generated from a committed seed so a resumed process, and any
    # independent replay, reconstructs exactly the same battery.
    seeds = [args.seed + i for i in range(args.per_domain)]
    tasks = ft.generate_task_battery(seeds, difficulty=2)
    manifest = ft.build_task_manifest(tasks)
    commitment = ft.build_task_commitment(manifest)
    _atomic_write(
        out_dir / "task_commitment.json",
        json.dumps(
            {
                "schema": SWEEP_SCHEMA,
                "seed": args.seed,
                "per_domain": args.per_domain,
                "task_count": len(tasks),
                "commitment_sha256": commitment.commitment_sha256,
                "registry_version": ft.REGISTRY_VERSION,
                "domains": list(ft.FRONTIER_DOMAINS),
            },
            indent=1,
            sort_keys=True,
        )
        + "\n",
    )
    print(f"{len(tasks)} tasks, commitment {commitment.commitment_sha256[:16]}", flush=True)

    if args.self_test:
        print(json.dumps({"tasks": len(tasks), "arms": [a[0] for a in selected]}, indent=2))
        return 0

    journal = Journal(out_dir / "journal.jsonl")
    planned = len(selected) * len(tasks)
    print(f"planned cells {planned}, already committed {len(journal.done)}", flush=True)

    from mlx_lm import load

    from core.runtime.mlx_memory_guard import mlx_memory_envelope
    from core.runtime.model_lane_control import standalone_model_lane

    started = time.monotonic()
    _status(
        out_dir,
        phase="loading_model",
        planned_cells=planned,
        committed_cells=len(journal.done),
        started_unix=_now(),
    )

    with standalone_model_lane(
        owner_id=f"rlc-reconciliation-sweep:{os.getpid()}",
        model_path=args.model,
        purpose="evaluation",
        preemptible=False,
        metadata={"tool": "run_rlc_reconciliation_sweep", "operator_launched": True},
    ), mlx_memory_envelope(fraction=args.memory_fraction) as envelope:
        print(f"memory envelope: {envelope.to_receipt()}", flush=True)
        model, tokenizer = load(args.model)
        print("model loaded", flush=True)

        for arm, steps, policy in selected:
            config = (
                None
                if steps is None
                else _build_config(steps, args.n_slots, policy, args.max_tokens)
            )
            for index, task in enumerate(tasks):
                key = (arm, task.task_id)
                if key in journal.done:
                    continue
                if time.monotonic() - started > args.max_wall_s:
                    _status(out_dir, phase="wall_budget_reached", arm=arm)
                    print("wall budget reached; exiting for clean resume", flush=True)
                    return 3
                cell_started = time.monotonic()
                error = ""
                receipt: dict[str, Any] = {}
                text = ""
                try:
                    if config is None:
                        text = _run_vanilla(
                            model,
                            tokenizer,
                            _render_prompt_text(tokenizer, task),
                            args.max_tokens,
                        )
                    else:
                        text, receipt = _run_rlc(
                            model, config, _render_prompt(tokenizer, task), tokenizer
                        )
                except Exception as exc:  # noqa: BLE001 - recorded, never silent
                    # A harness fault must be visible as a fault. It is never
                    # scored as a wrong answer.
                    error = f"{type(exc).__name__}: {exc}"
                    print(f"  !! {arm} {task.domain} {error}", flush=True)
                journal.append(
                    {
                        "event": "CELL",
                        "arm": arm,
                        "task_id": task.task_id,
                        "domain": task.domain,
                        "recurrent_steps": steps,
                        "terminal_instruction_policy": policy,
                        "text": text,
                        "error": error,
                        "latency_s": time.monotonic() - cell_started,
                        "decode_prefix_token_count": receipt.get("decode_prefix_token_count"),
                        "decode_prefix_composition": receipt.get("decode_prefix_composition"),
                        "decode_termination": receipt.get("decode_termination"),
                        "decode_generated_tokens": receipt.get("decode_generated_tokens"),
                        "halting_reason": receipt.get("halting_reason"),
                        "committed_unix": _now(),
                    }
                )
                journal.done.add(key)
                # A twelve-hour run reclaims or it dies. The envelope refused
                # an unguarded eval that reached 103GB once already.
                envelope.reclaim(force=True)
                _status(
                    out_dir,
                    phase="executing",
                    arm=arm,
                    arm_progress=f"{index + 1}/{len(tasks)}",
                    planned_cells=planned,
                    committed_cells=len(journal.done),
                    elapsed_s=time.monotonic() - started,
                )
                print(
                    f"  {arm} {index + 1}/{len(tasks)} {task.domain} "
                    f"{time.monotonic() - cell_started:.0f}s",
                    flush=True,
                )

    _status(out_dir, phase="grading", committed_cells=len(journal.done))
    verdict = grade(out_dir, tasks)
    _status(out_dir, phase="complete", verdict=verdict["decision"])
    print(json.dumps(verdict, indent=2))
    return 0


def grade(out_dir: Path, tasks) -> dict[str, Any]:
    """Score every committed cell and decide whether the path can be trained."""
    from core.brain.llm.latent_cortex import frontier_tasks as ft

    by_id = {t.task_id: t for t in tasks}
    journal = Journal(out_dir / "journal.jsonl")
    arms: dict[str, dict[str, Any]] = {}
    for cell in journal.cells():
        arm = cell["arm"]
        bucket = arms.setdefault(
            arm,
            {
                "correct": 0,
                "total": 0,
                "errors": 0,
                "reasons": {},
                "generated_tokens": [],
                "prefix_tokens": [],
            },
        )
        bucket["total"] += 1
        if cell.get("error"):
            bucket["errors"] += 1
            continue
        task = by_id.get(cell["task_id"])
        if task is None:
            continue
        result = ft.score_task(task, cell["text"])
        bucket["correct"] += int(result.correct)
        reason = result.reason or "correct"
        bucket["reasons"][reason] = bucket["reasons"].get(reason, 0) + 1
        if cell.get("decode_generated_tokens") is not None:
            bucket["generated_tokens"].append(cell["decode_generated_tokens"])
        if cell.get("decode_prefix_token_count") is not None:
            bucket["prefix_tokens"].append(cell["decode_prefix_token_count"])

    vanilla = arms.get("vanilla", {}).get("correct", 0)
    best_rlc_name, best_rlc = "", -1
    for name, bucket in arms.items():
        if name == "vanilla":
            continue
        if bucket["correct"] > best_rlc:
            best_rlc_name, best_rlc = name, bucket["correct"]

    reaches_parity = best_rlc >= vanilla and best_rlc >= 0
    decision = (
        "proceed_to_checkpoint_phase"
        if reaches_parity
        else "recurrent_path_below_ordinary_decode"
    )
    verdict = {
        "schema": SWEEP_SCHEMA,
        "arms": arms,
        "vanilla_correct": vanilla,
        "best_recurrent_arm": best_rlc_name,
        "best_recurrent_correct": best_rlc,
        "reaches_parity_with_ordinary_decode": reaches_parity,
        "decision": decision,
        "claims": {
            "reasoning_gain_proven": False,
            "fusion_authorized": False,
            "frontier_level_proven": False,
        },
        "graded_unix": _now(),
    }
    _atomic_write(
        out_dir / "verdict.json", json.dumps(verdict, indent=1, sort_keys=True) + "\n"
    )
    return verdict


if __name__ == "__main__":
    raise SystemExit(main())
