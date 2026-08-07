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

A third candidate cause surfaced from the sweep's own first cells and is
crossed here too: the control is completion-limited. At the campaign's decode
budget the ordinary decode emitted a terminal FINAL_ANSWER on 43% of tasks
against 96% for an arm instructed to answer briefly, so part of the deficit
may be a token budget rather than a reasoning result.

All seven arms share one model load -- they differ only in configuration --
which is what makes the whole sweep affordable.
"""
from __future__ import annotations

import argparse
import dataclasses
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

# (name, recurrent steps or None for ordinary decode, terminal-instruction
# policy, decode token budget or None for the campaign default).
#
# The first five reproduce the 2026-08-06 campaign at its own budget of 512.
# The last two exist because the control turned out to be completion-limited
# there: at 512 the ordinary decode emitted a terminal FINAL_ANSWER on only
# 43% of tasks, while an arm carrying "give only the best bounded answer"
# finished 96% of the time. A deficit measured against a control that mostly
# runs out of tokens is a statement about budget, not about recurrence, so the
# budget is varied directly rather than argued about.
@dataclasses.dataclass(frozen=True)
class Arm:
    """One measured configuration.

    ``profile`` selects how much of the built stack is switched on:

    ``ordinary``   plain decode -- the control.
    ``mechanism``  recurrence, branches and slots only, with the internal
                   proxy verifiers. This is what the 2026-08-06 campaign and
                   every reconciliation run before 2026-08-07 measured. It is
                   an ABLATION, not the system.
    ``full``       every pillar the program actually built: adaptive halting,
                   hidden-state optimization, temporary fast weights, local
                   repair and the evidence-bound acceptance rule, on top of
                   the verifier mesh. This is the claim under test -- that
                   reasoning is a unified system rather than one mechanism.
    ``full_oracle`` ``full`` plus a ground-truth verifier. Not a capability
                   claim and never promotable; it is the verifier ablation,
                   and it separates a generation ceiling from a selection
                   ceiling.
    """

    name: str
    steps: int | None
    policy: str
    max_tokens: int | None
    profile: str


ARMS: tuple[Arm, ...] = (
    Arm("vanilla", None, "applied", None, "ordinary"),
    Arm("full_stack", 8, "applied", None, "full"),
    Arm("full_stack_oracle", 8, "applied", None, "full_oracle"),
    # Ablations, meaningful only underneath the full arm.
    Arm("rlc_mechanism", 4, "suppressed", None, "mechanism"),
    Arm("vanilla_long", None, "applied", 1024, "ordinary"),
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
    arm: str = "",
    adapter: str = "",
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
            # An attached adapter is a different model. Without it in the
            # identity, a candidate's cells could resume as frozen-base cells.
            "adapter": str(adapter),
            "arm": str(arm),
            "contract": "rlc_reconciliation_decode.v3",
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

    def __init__(
        self, path: Path, fingerprint: str | dict[str, str] | None = None
    ) -> None:
        self.path = path
        # Arms may differ in decode budget, so identity is per arm. A bare
        # string applies to every arm; None admits everything.
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
        if isinstance(self.fingerprint, str):
            return record.get("decode_fingerprint") == self.fingerprint
        expected = self.fingerprint.get(record.get("arm", ""))
        # An arm absent from the current configuration was not asked for, so
        # its cells are not evidence for this run either.
        return expected is not None and record.get("decode_fingerprint") == expected

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
    profile: str = "mechanism",
):
    from core.brain.llm.latent_cortex.types import (
        BranchConfig,
        CortexConfig,
        FastWeightsConfig,
        LatentOptConfig,
        RecurrenceConfig,
        WorkspaceConfig,
    )

    full = profile in {"full", "full_oracle"}
    # Adaptive halting is the program's own latency lever and its own
    # capability claim: easy problems converge in two steps and stop paying,
    # hard ones are allowed to keep going. Pinning min == max, as every run
    # before 2026-08-07 did, forces the deep path onto every question and
    # removes the mechanism that makes depth affordable at all.
    min_steps = 2 if full else steps
    recurrence = RecurrenceConfig(
        max_steps=steps,
        min_steps=min_steps,
        alpha=0.5,
        alpha_schedule="cosine" if full else "constant",
        convergence_eps=0.02,
        fixed_depth=not full,
    )
    return CortexConfig(
        workspace=WorkspaceConfig(n_slots=n_slots, seed=0),
        recurrence=recurrence,
        branches=BranchConfig(
            n_branches=2,
            exchange_interval=4 if full else 1,
            # Branches must not isolate for longer than the run is allowed to
            # last; a depth-1 ablation would otherwise never exchange at all.
            isolation_steps=max(1, min(2, steps)),
        ),
        # Anima Rationis pillars 5 and 6. Off in every prior run, which is
        # why "the recurrent path" had never actually been measured: without
        # these the second pass is the same computation as the first.
        latent_opt=LatentOptConfig(enabled=full, steps=4, lr=0.05),
        fast_weights=FastWeightsConfig(enabled=full, rank=2, opt_steps=4),
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
        # The Spark's acceptance rule: replace an answer only when the
        # evidence for the new one clearly exceeds the old. Disabling it in a
        # research arm was how "confidently wrong" got measured and reported
        # as a property of recurrence.
        answer_replacement_enabled=full,
        local_repair_enabled=full,
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


def _episode_verifier(task):
    """The admitted, deployable verifier for a full-stack episode.

    ``EpisodeTaskVerifier`` already implements the whole admission contract:
    it is deterministic, it separates correct from incorrect arithmetic and
    parseable from unparseable code (so it passes the blind-review decoy
    preflight, which an answer-key lookup cannot), and it exposes the
    ``fast_weight_learning_evidence`` provider that fast-weight attachment
    requires. It scores a candidate by re-deriving what the candidate itself
    asserts, never by consulting the expected answer -- which is what makes an
    episode guided by it a measurement of the reasoning rather than of the
    answer key.
    """
    from core.brain.llm.latent_cortex.task_verifiers import EpisodeTaskVerifier

    return EpisodeTaskVerifier(task.public.prompt)


def _oracle_verifier(task):
    """Ground-truth scorer for the verifier ablation.

    The Spark's verifier ladder runs from no verifier through self-rating,
    learned, process and executable verifiers up to an oracle. The oracle rung
    exists to answer one question that no other rung can: when the system
    fails, is it because it could not GENERATE a correct trajectory, or
    because it could not SELECT the correct one it already had? Those two
    failures have opposite remedies, and guessing between them is how a
    program spends weeks on the wrong half.

    An arm using this is a ceiling, never a capability claim, and never
    promotable -- which is why it is bound to an arm name carrying "oracle".
    """
    from core.brain.llm.latent_cortex import frontier_tasks as ft

    def _score(candidate: str) -> float:
        try:
            return 1.0 if ft.score_task(task, candidate).correct else 0.0
        except Exception:  # noqa: BLE001 - a scorer fault is not a signal
            return 0.0

    return _score


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
    verifier=None,
    objective: str = "",
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
    episode_started = time.monotonic()
    kwargs: dict[str, Any] = {}
    if verifier is not None:
        kwargs["verifier"] = verifier
    if objective:
        # The engine derives its verification objective from prompt/messages.
        # Passing token_ids alone -- which this harness did to control the
        # chat template exactly -- leaves the objective empty, and BOTH the
        # generative and counterfactual verifiers then refuse to run with
        # "verification_objective_unavailable" no matter how well the verifier
        # itself was admitted. The messages form carries the objective while
        # keeping the same rendered tokens.
        kwargs["messages"] = [{"role": "user", "content": objective}]
        result = engine.reason(budget=budget, **kwargs)
    else:
        result = engine.reason(token_ids=prompt_tokens, budget=budget, **kwargs)
    receipt = result.receipt.to_dict()
    # Latency has to be attributable, or "make it faster" is guesswork. The
    # engine's own phase accounting is preferred; the wall time is the floor.
    receipt.setdefault("episode_wall_s", time.monotonic() - episode_started)
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
        default=",".join(a.name for a in ARMS),
        help="comma-separated subset of arms to execute",
    )
    parser.add_argument(
        "--adapter",
        default="",
        help=(
            "Optional recurrence adapter to ATTACH (not fuse) before the run. "
            "Attachment preserves ScopedLoRALinear slot scoping, which is the "
            "form the adapter was trained in; fusing folds the delta into the "
            "linear weights and changes ordinary decode at every token."
        ),
    )
    parser.add_argument("--adapter-manifest", default="")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    from core.brain.llm.latent_cortex import frontier_tasks as ft

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    selected = [a for a in ARMS if a.name in {v.strip() for v in args.arms.split(",")}]
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

    arm_tokens = {
        a.name: (args.max_tokens if a.max_tokens is None else a.max_tokens)
        for a in selected
    }
    fingerprints = {
        name: decode_fingerprint(
            model=args.model,
            n_slots=args.n_slots,
            max_tokens=tokens,
            episode_wall_s=args.episode_wall_s,
            seed=args.seed,
            per_domain=args.per_domain,
            arm=name,
            adapter=args.adapter,
        )
        for name, tokens in arm_tokens.items()
    }
    _atomic_write(
        out_dir / "decode_fingerprint.json",
        json.dumps(
            {"decode_fingerprint": fingerprints, "arm_max_tokens": arm_tokens},
            indent=1,
            sort_keys=True,
        )
        + "\n",
    )
    journal = Journal(out_dir / "journal.jsonl", fingerprints)
    planned = len(selected) * len(tasks)
    print(f"planned cells {planned}, already committed {len(journal.done)}", flush=True)
    if journal.superseded:
        # Loud, because silently re-running them looks identical to a slow start.
        print(
            f"discarded {journal.superseded} cells from a superseded decode "
            f"configuration; they will be re-run under the current arm budgets "
            f"{arm_tokens}",
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
        if args.adapter:
            from core.brain.llm.latent_cortex.resident_adapter_loader import (
                load_resident_adapter,
            )

            manifest_path = Path(args.adapter_manifest) if args.adapter_manifest else None
            if manifest_path is None or not manifest_path.exists():
                found = list(Path(args.adapter).parent.rglob(
                    "recurrence_adapter_manifest.json"
                ))
                manifest_path = found[0] if found else None
            if manifest_path is None:
                print("adapter requested but no manifest found", file=sys.stderr)
                return 2
            projections = load_resident_adapter(
                model,
                Path(args.adapter),
                json.loads(manifest_path.read_text(encoding="utf-8")),
            )
            print(
                f"adapter attached: {args.adapter} ({projections} projections, "
                f"slot scoping preserved)",
                flush=True,
            )

        for spec in selected:
            arm, steps, policy = spec.name, spec.steps, spec.policy
            tokens = arm_tokens[arm]
            config = (
                None
                if steps is None
                else _build_config(
                    steps, args.n_slots, policy, tokens, profile=spec.profile
                )
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
                            tokens,
                        )
                    else:
                        # The verifier ablation. An oracle arm is a
                        # diagnostic ceiling that separates a generation
                        # limit from a selection limit; it is never a
                        # capability claim and never promotable, which the
                        # arm name carries so no downstream reader can lose
                        # track of which one produced a number.
                        verifier = None
                        if spec.profile == "full":
                            verifier = _episode_verifier(task)
                        elif spec.profile == "full_oracle":
                            verifier = _oracle_verifier(task)
                        text, receipt = _run_rlc(
                            model,
                            config,
                            _render_prompt(tokenizer, task),
                            tokenizer,
                            wall_clock_s=args.episode_wall_s,
                            verifier=verifier,
                            objective=(
                                task.public.prompt
                                if spec.profile in {"full", "full_oracle"}
                                else ""
                            ),
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
                        "decode_fingerprint": fingerprints[arm],
                        "decode_max_tokens": tokens,
                        "recurrent_steps": steps,
                        "arm_profile": spec.profile,
                        "terminal_instruction_policy": policy,
                        # Latency is a first-class result, not a footnote:
                        # a unified system that answers better but takes ten
                        # minutes has not been shown to be deployable.
                        "steps_taken": receipt.get("steps_taken"),
                        "halted_early": receipt.get("halted_early"),
                        "phase_latency_s": receipt.get("phase_latency_s"),
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
                "latency_s": [],
                "steps_taken": [],
                "halted_early": 0,
            },
        )
        bucket["total"] += 1
        if cell.get("latency_s") is not None:
            bucket["latency_s"].append(float(cell["latency_s"]))
        if cell.get("steps_taken") is not None:
            bucket["steps_taken"].append(cell["steps_taken"])
        if cell.get("halted_early"):
            bucket["halted_early"] += 1
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

    # Accuracy without cost is not a deployability result. The program's own
    # standard requires equal-latency AND equal-compute comparisons, so every
    # arm publishes what its answers cost next to what they were worth.
    for bucket in arms.values():
        lat = sorted(bucket.pop("latency_s", []) or [])
        steps = [x for x in bucket.pop("steps_taken", []) if x is not None]
        bucket["latency_median_s"] = (
            round(lat[len(lat) // 2], 1) if lat else None
        )
        bucket["latency_p90_s"] = (
            round(lat[max(0, int(len(lat) * 0.9) - 1)], 1) if lat else None
        )
        bucket["steps_median"] = (
            sorted(steps)[len(steps) // 2] if steps else None
        )
        bucket["halted_early_fraction"] = (
            round(bucket["halted_early"] / bucket["total"], 2)
            if bucket.get("total")
            else None
        )

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
    # No recurrent arm ran, so nothing about the recurrent path was observed.
    # The sentinel -1 is smaller than any vanilla score, which would otherwise
    # publish "below ordinary decode" as a finding drawn from no data at all.
    measured_recurrence = best_rlc >= 0
    reaches_parity = (
        complete and informative and measured_recurrence and best_rlc >= vanilla
    )
    if not complete:
        decision = "inconclusive_arms_carry_harness_faults"
    elif not measured_recurrence:
        decision = "inconclusive_no_recurrent_arm_measured"
    elif not informative:
        decision = "inconclusive_battery_uninformative_ordinary_decode_scored_zero"
    elif reaches_parity:
        decision = "proceed_to_checkpoint_phase"
    else:
        decision = "recurrent_path_below_ordinary_decode"
    vanilla_latency = arms.get("vanilla", {}).get("latency_median_s") or 0.0
    verdict = {
        "schema": SWEEP_SCHEMA,
        "arms": arms,
        "vanilla_correct": vanilla,
        "best_recurrent_arm": best_rlc_name,
        "best_recurrent_correct": best_rlc,
        "arms_complete": complete,
        "faulted_arms": faulted,
        "battery_informative": informative,
        "latency_ratio_vs_ordinary_decode": {
            name: (
                round(b["latency_median_s"] / vanilla_latency, 1)
                if b.get("latency_median_s") and vanilla_latency
                else None
            )
            for name, b in arms.items()
        },
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
