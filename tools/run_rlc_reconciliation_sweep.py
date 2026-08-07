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
import hashlib
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


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def decode_fingerprint(
    *,
    model: str,
    n_slots: int,
    max_tokens: int,
    episode_wall_s: float,
    seed: int,
    per_domain: int,
) -> str:
    """Identity of the decode configuration every cell in a run must share.

    Twice now a resumed run reused cells produced under an older configuration
    -- once the recurrent arms, once the control -- and both times the effect
    was to compare arms that had been decoded under different rules. That is
    precisely the confound this sweep exists to remove, so the check cannot be
    a habit of whoever restarts the run. Cells carry their configuration and a
    mismatched cell is treated as absent.
    """
    body = json.dumps(
        {
            "contract": "rlc_reconciliation_decode.v1",
            "episode_wall_s": float(episode_wall_s),
            "max_tokens": int(max_tokens),
            "model": str(model),
            "n_slots": int(n_slots),
            "per_domain": int(per_domain),
            "seed": int(seed),
        },
        sort_keys=True,
    )
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


class Journal:
    """Append-only cell journal. Resumption replays it and skips committed work.

    Only cells matching the current decode fingerprint count as committed; a
    cell from a superseded configuration is discarded and re-run.
    """

    def __init__(self, path: Path, fingerprint: str | None = None) -> None:
        self.path = path
        self.fingerprint = fingerprint
        self.done: set[tuple[str, str]] = set()
        self.superseded = 0
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
                if record.get("event") != "CELL":
                    continue
                if not self._current(record):
                    self.superseded += 1
                    continue
                self.done.add((record["arm"], record["task_id"]))

    def _current(self, record: dict[str, Any]) -> bool:
        if self.fingerprint is None:
            return True
        return record.get("decode_fingerprint") == self.fingerprint

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
            if record.get("event") == "CELL" and self._current(record):
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
        # The 2026-08-06 campaign ran with these; without them a probe cell
        # decoded 640 tokens of "to to to to". Leaving the defaults would mean
        # rlc_asrun is not reproducing the arm it is supposed to reproduce, and
        # every recurrent arm would be measured against a decode discipline the
        # original never had.
        decode_repetition_penalty=1.25,
        decode_repetition_window=72,
        decode_bridge_policy="none",
        decode_incumbent_policy="latent",
        # Serving-side answer replacement is a live-product safeguard: it
        # abstains rather than emit a candidate it cannot bound. In a research
        # arm that abstention destroys the observation -- the episode returns
        # no text and the cell becomes a fault instead of a measurement. The
        # same thing aborted CP420S12. Research measures the mechanism, so the
        # raw recurrent answer is retained and graded on its own terms.
        answer_replacement_enabled=False,
        local_repair_enabled=False,
        # A degraded episode that quietly serves an ordinary decode would make
        # this arm a second copy of the vanilla control wearing the recurrent
        # arm's label -- the worst possible failure here, because it looks like
        # a result. Let it fault visibly instead.
        allow_vanilla_fallback=False,
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


class EpisodeFault(RuntimeError):
    """Infrastructure failed. Never scored as a wrong answer."""


# A model that cannot finish its answer inside the token budget has failed to
# answer -- that is a policy observation and it is scored incorrect, exactly as
# CP420S12 established. Infrastructure failures are different in kind and must
# never be graded, because grading them reports a broken harness as a reasoning
# result. The 2026-08-06 campaign's own base_rlc arm carried nine such policy
# failures out of 28, so excluding them would flatter the recurrent path.
_POLICY_FAILURE_MARKERS: tuple[str, ...] = (
    "decode_incomplete",
    "contract_irrecoverable",
    "token_limit",
    "budget_exhausted",
    "answer_replacement_abstained",
    "confidence_bound_abstention",
    "wall_reserve",
)


def _is_policy_failure(reason: str, termination: str) -> bool:
    haystack = f"{reason} {termination}".lower()
    if "latent_phase" in haystack or "worker" in haystack or "invariant" in haystack:
        return False
    return any(marker in haystack for marker in _POLICY_FAILURE_MARKERS)


def _run_rlc(
    model,
    config,
    prompt_tokens: list[int],
    tokenizer,
    *,
    wall_clock_s: float = 720.0,
) -> tuple[str, dict[str, Any]]:
    from core.brain.llm.latent_cortex.engine import LatentCortexEngine
    from core.brain.llm.latent_cortex.types import ComputeBudget

    engine = LatentCortexEngine(model, config=config, tokenizer=tokenizer)
    # The default 120s episode wall clock is smaller than these episodes take:
    # the 2026-08-06 campaign's median recurrent episode ran 298s. Left at the
    # default, every recurrent cell terminates budget_exhausted with no text --
    # which would have been graded as a wrong answer, making the whole arm a
    # measurement of the budget rather than of recurrence.
    budget = ComputeBudget(wall_clock_s=wall_clock_s)
    result = engine.reason(token_ids=prompt_tokens, budget=budget)
    receipt = result.receipt.to_dict()
    text = getattr(result, "text", "") or ""
    if not text and tokenizer is not None and getattr(result, "tokens", None):
        text = tokenizer.decode(list(result.tokens))
    reason = str(getattr(result, "reason", "") or "")
    termination = str(receipt.get("decode_termination") or "")
    if not result.ok or not text.strip():
        if not _is_policy_failure(reason, termination):
            raise EpisodeFault(
                f"episode produced no answer: ok={result.ok} "
                f"reason={reason!r} termination={termination!r}"
            )
        # A policy failure is a real observation: the arm did not answer.
        # It is graded on exactly the text it managed to emit.
    # Belt and braces on the same hazard: if any fallback did serve an ordinary
    # decode, this is not an observation of the recurrent path.
    flags = [str(flag) for flag in (receipt.get("honest_flags") or [])]
    if any("fallback" in flag or "vanilla" in flag for flag in flags):
        raise EpisodeFault(f"episode degraded to an ordinary decode: flags={flags}")
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
    # Per-episode wall clock. The engine default is 120s; the campaign's median
    # recurrent episode was 298s, so the default silently starves every arm.
    parser.add_argument("--episode-wall-s", type=float, default=720.0)
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

    fingerprint = decode_fingerprint(
        model=args.model,
        n_slots=args.n_slots,
        max_tokens=args.max_tokens,
        episode_wall_s=args.episode_wall_s,
        seed=args.seed,
        per_domain=args.per_domain,
    )
    _atomic_write(
        out_dir / "decode_fingerprint.json",
        json.dumps({"decode_fingerprint": fingerprint}, indent=1, sort_keys=True) + "\n",
    )
    journal = Journal(out_dir / "journal.jsonl", fingerprint)
    planned = len(selected) * len(tasks)
    print(f"planned cells {planned}, already committed {len(journal.done)}", flush=True)
    if journal.superseded:
        # Loud, because silently re-running them looks identical to a slow start.
        print(
            f"discarded {journal.superseded} cells from a superseded decode "
            f"configuration; they will be re-run under {fingerprint[:16]}",
            flush=True,
        )

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
                            model,
                            config,
                            _render_prompt(tokenizer, task),
                            tokenizer,
                            wall_clock_s=args.episode_wall_s,
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
                        "decode_fingerprint": fingerprint,
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
    # Grading is bound to the same configuration the cells were produced under.
    # Absent the record (an older run, or a unit test), every cell is admitted.
    recorded = _read_json(out_dir / "decode_fingerprint.json")
    journal = Journal(
        out_dir / "journal.jsonl",
        (recorded or {}).get("decode_fingerprint"),
    )
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

    # An arm carrying harness faults has not been measured. Concluding either
    # way from it would report a starved budget as a reasoning result.
    faulted = {name: b["errors"] for name, b in arms.items() if b["errors"]}
    complete = not faulted
    # A battery the ordinary decode cannot score on has not measured the
    # recurrent path either: 0 >= 0 satisfies every inequality below, so mutual
    # failure would otherwise be published as parity and promote a model that
    # answered nothing. Parity is a claim about a baseline, and with no solved
    # control task there is no baseline to be at parity with. The floor is
    # structural (a baseline exists / does not), not a tuned threshold.
    informative = vanilla > 0
    reaches_parity = complete and informative and best_rlc >= vanilla
    if not complete:
        decision = "inconclusive_arms_carry_harness_faults"
    elif not informative:
        decision = "inconclusive_battery_uninformative_ordinary_decode_scored_zero"
    elif reaches_parity:
        decision = "proceed_to_checkpoint_phase"
    else:
        decision = "recurrent_path_below_ordinary_decode"
    verdict = {
        "schema": SWEEP_SCHEMA,
        "arms": arms,
        "vanilla_correct": vanilla,
        "best_recurrent_arm": best_rlc_name,
        "best_recurrent_correct": best_rlc,
        "arms_complete": complete,
        "faulted_arms": faulted,
        "battery_informative": informative,
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
