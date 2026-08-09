#!/usr/bin/env python
"""Frozen-checkpoint complete-engine sweep for the Recursive Latent Cortex.

The earlier directional campaigns measured a recurrence mechanism in
isolation and repeatedly described the result as a property of the RLC. That
was the wrong experimental object. The architecture's claim is the integrated
system: persistent workspace, independent branches, controlled recurrence,
latent optimization, temporary fast weights, verifier-guided local repair,
adaptive compute, and evidence-bound answer promotion.

This sweep measures that complete engine against two retained controls:
ordinary greedy decode and a preliminary best-of-three textual control.
Ordinary decode remains the per-task incumbent, so an unpromoted full-stack
answer must be byte-identical to it. Diagnostic ablations can explain a result
but can never win the experiment. The sweep performs no optimizer update and
awards no reasoning, fusion, frontier, or production claim by itself.

Every cell is bound to the exact model, task commitment, decode configuration,
adapter, and implementation source. A partial, stale, mixed-source, or
runtime-unmeasured battery is inconclusive rather than a negative result.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Final

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

SWEEP_SCHEMA = "aura.rlc_reconciliation_sweep.v1"
EVIDENCE_MANIFEST_SCHEMA = "aura.rlc.reconciliation_evidence_manifest.v1"


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
    ``full``       the complete neural RLC engine: adaptive halting,
                   hidden-state optimization, temporary fast weights, local
                   repair and evidence-bound acceptance on top of the verifier
                   mesh. It does not claim the service-side acquisition and
                   reasoning-amplifier layers.
    ``complete_closed_book`` the neural RLC plus its bounded deterministic
                   acquisition continuation, the production reasoning
                   amplifier, and a final evidence-bound promotion gate. It
                   is sealed from memory, RAG, web, and answer keys so the arm
                   measures same-information system reasoning.
    ``full_oracle`` ``full`` plus a ground-truth verifier. Not a capability
                   claim and never deployable; its research-only arbitration
                   may expose a correct generated candidate so the verifier
                   ablation can separate a generation ceiling from a
                   selection ceiling.
    """

    name: str
    steps: int | None
    policy: str
    max_tokens: int | None
    profile: str


ARMS: tuple[Arm, ...] = (
    Arm("vanilla", None, "applied", None, "ordinary"),
    # Historical preliminary control. Three independent textual samples are a
    # useful stronger-than-greedy baseline, but they are not resource matched
    # to the adaptive complete system. Claim-grade parity is established only
    # by the digest-bound operation ledger in the paired campaign.
    Arm("vanilla_equal_compute", None, "applied", None, "ordinary_best_of_3"),
    Arm(
        "complete_system_closed_book",
        8,
        "suppressed",
        None,
        "complete_closed_book",
    ),
    # The product answer starts from the same clean prompt root as vanilla.
    # Terminal-disposition text changes the decode before any evidence gate and
    # therefore cannot coexist with a byte-identical incumbent guarantee.
    Arm("full_stack", 8, "suppressed", None, "full"),
    Arm("full_stack_disposition", 8, "applied", None, "full"),
    Arm("full_stack_oracle", 8, "suppressed", None, "full_oracle"),
    # Ablations, meaningful only underneath the full arm.
    Arm("rlc_mechanism", 4, "suppressed", None, "mechanism"),
    Arm("vanilla_long", None, "applied", 1024, "ordinary"),
)

CONTROL_ARM_NAMES: Final[tuple[str, ...]] = ("vanilla", "vanilla_equal_compute")
CLAIM_ARM_NAMES: Final[frozenset[str]] = frozenset({"complete_system_closed_book", "full_stack"})
DIAGNOSTIC_ARM_NAMES: Final[frozenset[str]] = frozenset(
    {"full_stack_disposition", "full_stack_oracle", "rlc_mechanism", "vanilla_long"}
)
RUNTIME_STACK_ARM_NAMES: Final[frozenset[str]] = frozenset(
    arm.name for arm in ARMS if arm.profile in {"complete_closed_book", "full", "full_oracle"}
)


def _expand_requested_arms(requested_names: set[str]) -> list[Arm]:
    """Return an executable experiment, never a treatment without controls.

    A subset restart used to write a manifest containing only ``full_stack``.
    That made the 56 exact baseline cells already in the journal invisible to
    grading, and a fresh directory would have run no controls at all. Any
    request that measures a non-control arm now brings both preregistered
    controls with it. Exact matching journal cells are skipped normally.
    """

    by_name = {arm.name: arm for arm in ARMS}
    unknown = requested_names - set(by_name)
    if unknown:
        raise ValueError(f"unknown arms requested: {sorted(unknown)}")
    expanded = set(requested_names)
    if expanded - set(CONTROL_ARM_NAMES):
        expanded.update(CONTROL_ARM_NAMES)
    return [arm for arm in ARMS if arm.name in expanded]


def _now() -> float:
    return time.time()


def _atomic_write(path: Path, payload: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(payload, encoding="utf-8")
    os.replace(tmp, path)


def _persist_runtime_receipt(
    out_dir: Path,
    *,
    arm: str,
    task_id: str,
    receipt: dict[str, Any],
) -> tuple[str, str]:
    """Persist the complete public receipt outside the append-only journal."""

    canonical = json.dumps(
        receipt,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    key = hashlib.sha256(f"{arm}:{task_id}".encode()).hexdigest()
    receipt_dir = out_dir / "runtime_receipts"
    receipt_dir.mkdir(parents=True, exist_ok=True)
    path = receipt_dir / f"{key}.json"
    _atomic_write(path, json.dumps(receipt, indent=1, sort_keys=True) + "\n")
    return str(path.relative_to(out_dir)), digest


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def _implementation_manifest() -> dict[str, str]:
    """Hash every executable source file that defines this experiment.

    A git commit alone is insufficient in a shared, potentially dirty worktree.
    The cell identity therefore commits to the bytes actually imported by the
    RLC plus this runner/grader. Any implementation change retires old cells.
    """

    latent_root = REPO_ROOT / "core/brain/llm/latent_cortex"
    verifier_root = REPO_ROOT / "core/brain/verifiers"
    paths = (
        Path(__file__).resolve(),
        REPO_ROOT / "tools/rlc_complete_system_closed_book.py",
        REPO_ROOT / "tools/rlc_reconciliation_evidence.py",
        *sorted(latent_root.glob("*.py")),
        *sorted(verifier_root.glob("*.py")),
        REPO_ROOT / "core/brain/calibration_gate.py",
        REPO_ROOT / "core/brain/cortex_compute_acquisition.py",
        REPO_ROOT / "core/brain/courtroom.py",
        REPO_ROOT / "core/brain/reasoning_amplifier.py",
        REPO_ROOT / "core/brain/reasoning_amplifier_v2.py",
        REPO_ROOT / "core/brain/symbolic_sandbox.py",
    )
    return {
        str(path.relative_to(REPO_ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in paths
    }


def _implementation_sha256(manifest: dict[str, str] | None = None) -> str:
    canonical = json.dumps(
        manifest if manifest is not None else _implementation_manifest(),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def decode_fingerprint(
    *,
    model: str,
    n_slots: int,
    max_tokens: int,
    episode_wall_s: float,
    seed: int,
    per_domain: int,
    difficulty: int = 2,
    arm: str = "",
    adapter: str = "",
    implementation_sha256: str | None = None,
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
            "contract": "rlc_reconciliation_decode.v4",
            "difficulty": int(difficulty),
            "episode_wall_s": float(episode_wall_s),
            "implementation_sha256": (implementation_sha256 or _implementation_sha256()),
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

    def __init__(self, path: Path, fingerprint: str | dict[str, str] | None = None) -> None:
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


YIELD_SENTINEL: Final = "YIELD"


def yield_requested(out_dir: Path) -> bool:
    """True when the operator wants the GPU back.

    The host cannot hold two 32B models, so a long campaign and the live
    instance are strictly exclusive. That does not mean the campaign needs
    long contiguous blocks -- only that it must be able to stand up and leave
    on request. Touching ``YIELD`` in the run directory stops the sweep at the
    next cell boundary, which costs at most one cell, and every committed cell
    stays valid because configuration identity travels with it.
    """
    return (out_dir / YIELD_SENTINEL).exists()


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
    # NOTE: the product arm overrides decode_contract to "none" below, which
    # is what the deployed system runs. Contract enforcement inside the engine
    # does not merely stop generation the way the ordinary control does -- it
    # can discard the produced answer entirely (measured: 576 generated tokens
    # returned as an empty string under token_limit_contract_incomplete). An
    # unfinished answer is a policy observation to be scored, never something
    # to blank, and blanking it puts the arm below the floor by construction.
):
    from core.brain.llm.latent_cortex.types import (
        BranchConfig,
        CortexConfig,
        FastWeightsConfig,
        LatentOptConfig,
        RecurrenceConfig,
        WorkspaceConfig,
    )

    full = profile in {"complete_closed_book", "full", "full_oracle"}
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
        # Raw latent proposals often preserve the exact greedy text while
        # reducing the answer-independent proxy. Rejecting those ties means
        # every proposal is reverted before it can accumulate into a changed
        # decode. The optimizer already requires BOTH semantic non-regression
        # and strict proxy descent; enable that conservative continuation rule
        # for the complete-engine arm so latent search can actually move.
        verifier_accept_non_regression=full,
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
        # The deployed system decodes at penalty 1.0; so does the ordinary
        # control. 1.25 was added here to stop a degenerate "to to to to"
        # probe, and it is actively hostile to this task family: step-by-step
        # arithmetic repeats phrasing, variable names and digits by
        # construction, so penalizing repetition penalizes the reasoning. It
        # also meant the incumbent decode was not ordinary decode, which
        # breaks the floor the whole design depends on.
        decode_repetition_penalty=1.0 if full else 1.25,
        decode_repetition_window=72,
        # The protected public answer remains the byte-identical vanilla
        # incumbent. This bridge is consumed only by latent candidate probes,
        # where the 32B otherwise spends its 256-token evidence budget on
        # preamble and never reaches the requested terminal answer contract.
        decode_bridge_policy=("assistant_answer_v4" if full else "none"),
        # The deployed system runs "vanilla_incumbent": every subsystem still
        # executes and is receipted, but the public answer decodes from the
        # clean prompt root, and a latent answer only takes over when an
        # independent gain gate promotes it. That is monotonic by
        # construction -- the system cannot score below ordinary decode.
        #
        # "latent" hands the answer to the recurrent path unconditionally,
        # which means ordinary decode's answer is never a candidate at all.
        # That is correct for the mechanism ablation (a degraded episode must
        # not silently serve a vanilla answer and be read as a recurrent
        # result) and flatly wrong for the arm that is supposed to BE the
        # product. Carried over from the ablation config, it is why a stack
        # with more verification scored HALF of plain greedy decode: the
        # verifiers were selecting the best of several equally corrupted
        # latent candidates, and the good answer was not in the pool.
        # Selection cannot exceed the best candidate it is given.
        decode_incumbent_policy=("vanilla_incumbent" if full else "latent"),
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
        # Both private branches must have a chance to become gradeable. A
        # one-attempt ceiling made an invalid first branch consume the entire
        # repair budget while the second branch remained permanently outside
        # the verifier and promotion pools.
        local_repair_max_attempts=2 if full else 1,
        # A repair regenerates from the failed atom to the END of the answer,
        # and it must terminate on the contract or it is discarded. At the
        # default 128 tokens it never can: these answers run ~450 tokens, so
        # every repair returned generation_contract_invalid and was thrown
        # away. Measured on the 32B: refuted=1, repair requests=1,
        # transactions=[('repair_not_executed', 'generation_contract_invalid')]
        # -- the chain ran end to end and then dropped its only candidate,
        # which is why the arm could tie ordinary decode but never beat it.
        # The repair therefore gets the same room as the answer it replaces.
        local_repair_max_tokens=(max(32, min(512, max_tokens)) if full else 128),
        # A degraded episode that quietly serves an ordinary decode would make
        # this arm a second copy of the vanilla control wearing the recurrent
        # arm's label -- the worst possible failure here, because it looks like
        # a result. Let it fault visibly instead.
        allow_vanilla_fallback=False,
        # Verify the answer, not a preview of it.
        #
        # The default probe budget is 48 tokens. Every verification surface --
        # atomic decomposition, the deterministic router, the disagreement
        # graph, local repair, and therefore answer promotion -- runs on these
        # branch probes rather than on the decoded answer. Measured on the
        # 32B: a 1500-character answer decomposed to ONE atom covering 48
        # characters, "To solve this combinatorics problem, we need to ", and
        # every route came back unknown because a preamble contains nothing
        # checkable. Zero refutations means zero repair requests means zero
        # promotion candidates, whatever else is fixed upstream.
        #
        # The probe budget is therefore sized to decompose, not to reproduce.
        # At 48 tokens a 1500-character answer yielded ONE atom -- a preamble
        # with nothing checkable. At the full 512 it yielded 19 atoms and found
        # the refutation, but cost 394-591s per cell against a 150s budget,
        # because every branch pays it on every episode. Half the decode budget
        # keeps roughly ten atoms -- an order of magnitude more checkable
        # surface than the failure mode, and far past the one atom that made
        # routing impossible -- at half the price. Verification needs enough of
        # the answer to find a claim, not all of it.
        verifier_probe_max_tokens=max(48, min(256, max_tokens // 2)) if full else 48,
        decode_contract="none" if full else decode_contract,
        # Candidate verification needs a complete, gradeable object even when
        # the public answer remains the byte-identical vanilla incumbent.
        # This contract is deliberately separate from decode_contract so the
        # evidence lane cannot move the product floor.
        verifier_probe_contract="final_answer_v1" if full else "none",
        decode_contract_grace_tokens=(0 if full else (320 if decode_contract != "none" else 0)),
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


def _equal_compute_seed(campaign_seed: int, task_id: str, sample_index: int) -> int:
    """Bind every control draw to its task instead of reusing one RNG stream."""

    material = f"{campaign_seed}:{task_id}:equal-compute:{sample_index}"
    return int.from_bytes(hashlib.sha256(material.encode("utf-8")).digest()[:4], "big")


def _run_vanilla_best_of(
    model,
    tokenizer,
    rendered: str,
    max_tokens: int,
    samples: int,
    *,
    campaign_seed: int,
    task_id: str,
) -> str:
    """Preliminary N-sample textual control with a self-consistency vote.

    This historical arm predates operation-level accounting. It must not be
    called equal-compute merely because an older wall-clock observation made
    three samples look similar in price. A unified stack that beats this arm
    has earned a pilot signal, not a compute-matched architectural-gain claim.

    Selection is a majority vote over the extracted FINAL_ANSWER payload, not
    a verifier score. Giving this arm the verifier would make it a different
    system rather than the compute-matched control.
    """
    import mlx.core as mx
    from mlx_lm import stream_generate
    from mlx_lm.sample_utils import make_sampler

    from core.brain.llm.latent_cortex.answer_contract import is_contract_complete

    candidates: list[str] = []
    for index in range(max(1, samples)):
        mx.random.seed(_equal_compute_seed(campaign_seed, task_id, index))
        pieces: list[str] = []
        for response in stream_generate(
            model,
            tokenizer,
            prompt=rendered,
            max_tokens=max_tokens,
            sampler=make_sampler(temp=0.7, top_p=0.95),
        ):
            pieces.append(response.text)
            if "}" in response.text and is_contract_complete("".join(pieces)):
                break
        candidates.append("".join(pieces))

    payloads: dict[str, list[str]] = {}
    for text in candidates:
        marker = text.rfind("FINAL_ANSWER:")
        key = text[marker:].strip() if marker >= 0 else ""
        if key:
            payloads.setdefault(key, []).append(text)
    if not payloads:
        return candidates[0]
    # Ties break on first appearance, which is deterministic given the seeds.
    winner = max(payloads.items(), key=lambda kv: len(kv[1]))
    return winner[1][0]


def _run_vanilla(
    model,
    tokenizer,
    prompt_tokens: list[int],
    max_tokens: int,
) -> tuple[str, list[int], str]:
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
    output_tokens: list[int] = []
    termination = "token_limit"
    for response in stream_generate(model, tokenizer, prompt=prompt_tokens, max_tokens=max_tokens):
        pieces.append(response.text)
        # MLX emits one final response for both EOS and length termination.
        # Its EOS response carries the stop-token id but deliberately exposes
        # no stop-token text. The engine's native decoder likewise excludes
        # EOS from its public token sequence, so binding that id would make a
        # truthful streamed answer fail its own token/text identity check.
        if response.finish_reason != "stop":
            output_tokens.append(int(response.token))
        if response.finish_reason == "stop":
            termination = "eos"
        elif response.finish_reason == "length":
            termination = "token_limit"
        if "}" in response.text and is_contract_complete("".join(pieces)):
            termination = "contract_complete"
            break
    text = tokenizer.decode(output_tokens)
    if text != "".join(pieces):
        raise RuntimeError("ordinary decode token/text round trip differs")
    return text, output_tokens, termination


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

    return EpisodeTaskVerifier(
        task.public.prompt,
        response_contract=task.public.response_contract,
    )


def _contract_neutral_score(task, text: str):
    """Score model-generated values after deterministic representation repair.

    This diagnostic cannot affect the strict serving grade. It only separates
    a wrong answer from one correct payload wrapped in the wrong transport
    syntax. The parser may preserve and re-encode generated values; it cannot
    invent, replace, or choose answer values.
    """

    from core.brain.llm.latent_cortex import frontier_tasks as ft
    from core.brain.llm.latent_cortex.contract_repair import (
        parse_contract_repair_generation,
    )

    try:
        normalized = parse_contract_repair_generation(
            text,
            response_contract=task.public.response_contract,
        )
    except (TypeError, ValueError):
        return ft.score_task(task, text), False
    return ft.score_task(task, normalized), normalized != text.strip()


def _route_counts(receipt: dict[str, Any]) -> dict[str, int]:
    """Aggregate deterministic-route outcomes across branches for one cell."""
    selection = receipt.get("diagnostic_action_selection") or {}
    routes = selection.get("candidate_routes") or {}
    totals: dict[str, int] = {}
    for envelope in routes.values():
        for key, value in (envelope.get("counts") or {}).items():
            totals[key] = totals.get(key, 0) + int(value)
    return totals


def _full_stack_evidence(receipt: dict[str, Any]) -> dict[str, Any]:
    from tools.rlc_reconciliation_evidence import full_stack_evidence

    return full_stack_evidence(receipt)


def _complete_system_evidence(receipt: dict[str, Any]) -> dict[str, Any]:
    from tools.rlc_complete_system_closed_book import (
        _complete_system_evidence as summarize,
    )

    return summarize(receipt, engine_evidence=_full_stack_evidence(receipt))


def _runtime_receipt_issues(
    out_dir: Path,
    cell: dict[str, Any],
) -> list[str]:
    """Verify that a compact cell summary reconstructs from its full receipt."""

    relative = cell.get("runtime_receipt_path")
    expected_sha = cell.get("runtime_receipt_sha256")
    if not isinstance(relative, str) or not relative:
        return ["runtime_receipt_absent"]
    candidate = (out_dir / relative).resolve()
    receipt_root = (out_dir / "runtime_receipts").resolve()
    if receipt_root not in candidate.parents or not candidate.is_file():
        return ["runtime_receipt_path_invalid"]
    receipt = _read_json(candidate)
    if not isinstance(receipt, dict):
        return ["runtime_receipt_unreadable"]
    canonical = json.dumps(
        receipt,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )
    observed_sha = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    issues: list[str] = []
    if observed_sha != expected_sha:
        issues.append("runtime_receipt_digest_mismatch")
    if cell.get("full_stack_evidence") != _full_stack_evidence(receipt):
        issues.append("runtime_receipt_summary_mismatch")
    expected_complete = cell.get("complete_system_evidence")
    if expected_complete is not None and expected_complete != _complete_system_evidence(receipt):
        issues.append("complete_system_receipt_summary_mismatch")
    if expected_complete is not None:
        observed_text_sha256 = hashlib.sha256(
            str(cell.get("text") or "").encode("utf-8")
        ).hexdigest()
        if expected_complete.get("final_text_sha256") != observed_text_sha256:
            issues.append("complete_system_final_text_mismatch")
    return issues


class _OracleTaskVerifier:
    """Answer-key diagnostic with an independently calibrated review surface.

    The hidden answer key is only consulted for complete task answers. Blind
    admission controls deliberately contain no FINAL_ANSWER contract, so they
    are evaluated by the same deterministic candidate-local verifier used by
    the deployable arm. This proves the scorer can discriminate before the
    answer key receives authority; an answer-key-only closure scores every
    decoy zero and is correctly rejected.
    """

    def __init__(self, task) -> None:
        from core.brain.llm.latent_cortex import frontier_tasks as ft
        from core.brain.llm.latent_cortex.task_verifiers import EpisodeTaskVerifier

        self.task = task
        self.response_contract = task.public.response_contract
        self._local = EpisodeTaskVerifier(
            task.public.prompt,
            response_contract=self.response_contract,
        )
        self.evaluations = self._local.evaluations
        self._scorer_source_sha256 = hashlib.sha256(Path(ft.__file__).read_bytes()).hexdigest()

    def __call__(self, candidate: str) -> float:
        local_score = self._local(candidate)
        if "FINAL_ANSWER:" not in candidate:
            return local_score
        from core.brain.llm.latent_cortex import frontier_tasks as ft

        try:
            return 1.0 if ft.score_task(self.task, candidate).correct else 0.0
        except Exception:  # noqa: BLE001 - a scorer fault grants no authority
            return 0.0

    def fast_weight_learning_evidence(self, *args, **kwargs):
        return self._local.fast_weight_learning_evidence(*args, **kwargs)

    def latent_state_score(self, text: str) -> float:
        """Use only candidate-local semantics; hidden answers remain excluded."""

        return self._local.latent_state_score(text)

    def to_receipt(self, *args, **kwargs):
        return self._local.to_receipt(*args, **kwargs)

    def research_oracle_assessment(self, candidate: str) -> dict[str, Any]:
        """Return a hidden-answer verdict carrying research authority only."""

        from core.brain.llm.latent_cortex import frontier_tasks as ft
        from core.brain.llm.latent_cortex.research_oracle_arbitration import (
            build_research_oracle_assessment,
        )

        result = ft.score_task(self.task, candidate)
        public = self.task.public
        return build_research_oracle_assessment(
            candidate=candidate,
            task_id=public.task_id,
            task_payload_sha256=public.task_payload_sha256,
            answer_commitment_sha256=public.answer_commitment_sha256,
            scorer_id=public.scorer_id,
            scorer_version=public.scorer_version,
            scorer_source_sha256=self._scorer_source_sha256,
            parsed=result.parsed,
            correct=result.correct,
            reason=result.reason,
            normalized_answer_sha256=result.normalized_answer_sha256,
        )


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
    deployable. Its output may be promoted only inside the hash-bound
    research-oracle arbitration receipt so the measured arm can expose a
    generated answer the deployable selector missed.
    """
    return _OracleTaskVerifier(task)


class EpisodeFault(RuntimeError):  # noqa: N818 - domain term distinguishes a cell fault
    """Infrastructure failed. Never scored as a wrong answer."""

    def __init__(self, message: str, *, receipt: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.receipt = dict(receipt or {})


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


def _manifest_integrity_issues(
    recorded: dict[str, Any] | None,
    tasks: list[Any] | tuple[Any, ...],
) -> list[str]:
    """Validate the evidence envelope before any contained cell is credited."""

    if recorded is None:
        # Historical artifacts and small unit fixtures predate the envelope.
        return []
    if recorded.get("schema") != EVIDENCE_MANIFEST_SCHEMA:
        return []

    from core.brain.llm.latent_cortex import frontier_tasks as ft

    issues: list[str] = []
    expected_ids = [task.task_id for task in tasks]
    recorded_ids = recorded.get("expected_task_ids")
    if recorded_ids != expected_ids or len(set(recorded_ids or [])) != len(expected_ids):
        issues.append("task_set_mismatch")

    task_manifest = ft.build_task_manifest(tasks)
    commitment = ft.build_task_commitment(task_manifest).commitment_sha256
    if recorded.get("task_commitment_sha256") != commitment:
        issues.append("task_commitment_mismatch")

    required = recorded.get("required_arms")
    requested = recorded.get("requested_arms")
    arm_names = {arm.name for arm in ARMS}
    if (
        not isinstance(required, list)
        or not required
        or len(set(required)) != len(required)
        or any(name not in arm_names for name in required)
    ):
        issues.append("required_arms_invalid")
        required_set: set[str] = set()
    else:
        required_set = set(required)
    if (
        not isinstance(requested, list)
        or not requested
        or any(name not in required_set for name in requested)
    ):
        issues.append("requested_arms_invalid")
    if required_set - set(CONTROL_ARM_NAMES) and not set(CONTROL_ARM_NAMES).issubset(required_set):
        issues.append("treatment_controls_missing")

    fingerprints = recorded.get("decode_fingerprint")
    arm_tokens = recorded.get("arm_max_tokens")
    if not isinstance(fingerprints, dict) or set(fingerprints) != required_set:
        issues.append("decode_fingerprint_arm_set_mismatch")
    elif any(
        not isinstance(value, str)
        or len(value) != 64
        or any(ch not in "0123456789abcdef" for ch in value)
        for value in fingerprints.values()
    ):
        issues.append("decode_fingerprint_invalid")
    if not isinstance(arm_tokens, dict) or set(arm_tokens) != required_set:
        issues.append("arm_token_budget_set_mismatch")

    implementation = _implementation_manifest()
    implementation_digest = _implementation_sha256(implementation)
    if recorded.get("implementation_files") != implementation:
        issues.append("implementation_file_manifest_mismatch")
    if recorded.get("implementation_sha256") != implementation_digest:
        issues.append("implementation_identity_mismatch")
    return sorted(set(issues))


def _bind_sweep_runtime_identity(
    receipt: dict[str, Any],
    *,
    config,
    budget,
    wall_clock_s: float,
    verifier,
    objective: str,
    domain: str,
    incumbent_artifact,
    worker_identity: dict[str, Any] | None,
    runtime_identity: dict[str, Any] | None,
) -> None:
    if not isinstance(worker_identity, dict) or not worker_identity:
        raise EpisodeFault("sweep worker identity is absent", receipt=receipt)
    if not isinstance(runtime_identity, dict) or not runtime_identity:
        raise EpisodeFault("sweep source/runtime identity is absent", receipt=receipt)
    for field in (
        "worker_boot_id",
        "worker_pid",
        "worker_model_path",
        "worker_model_parameter_count",
        "worker_model_stored_parameter_element_count",
        "worker_model_parameter_count_basis",
        "worker_source_sha256",
        "worker_affective_steering_active",
        "worker_affective_steering_alpha",
    ):
        receipt[field] = worker_identity[field]
    receipt["worker_identity"] = dict(worker_identity)
    receipt["runtime_identity"] = dict(runtime_identity)
    from core.brain.llm.latent_cortex.runtime_identity import (
        latent_request_payload_sha256,
    )

    request_messages = [{"role": "user", "content": objective}] if objective else None
    receipt["request_payload_sha256"] = latent_request_payload_sha256(
        prompt=None,
        messages=request_messages,
        domain=domain,
        config=dataclasses.asdict(config),
        budget={
            "max_layer_apps": budget.max_layer_apps,
            "wall_clock_s": wall_clock_s,
        },
        runtime_controls={
            "surface": "rlc_reconciliation_sweep",
            "incumbent_receipt_sha256": str(
                (getattr(incumbent_artifact, "receipt", {}) or {}).get(
                    "receipt_sha256",
                    "",
                )
            ),
        },
        verifier_guidance=True if verifier is not None else None,
    )
    from core.brain.llm.latent_cortex.runtime_integrity import (
        bind_worker_runtime_integrity,
        runtime_integrity_safe,
    )

    receipt["runtime_integrity"] = bind_worker_runtime_integrity(
        receipt.get("runtime_integrity") or {},
        worker_identity=worker_identity,
    )
    if not runtime_integrity_safe(
        receipt["runtime_integrity"],
        require_worker=True,
        expected_episode_id=receipt["episode_id"],
        expected_input_tokens_sha256=receipt["input_tokens_sha256"],
        expected_worker_identity=worker_identity,
        expected_fast_weights_applied=receipt.get("fast_weights_applied") is True,
        expected_checkpoint_fingerprint=receipt["checkpoint_fingerprint"],
        expected_checkpoint_method=receipt["checkpoint_fingerprint_method"],
        expected_checkpoint_file_count=receipt["checkpoint_file_count"],
    ):
        raise EpisodeFault("sweep worker runtime integrity is unproven", receipt=receipt)
    from core.brain.llm.latent_cortex.causal_receipt import (
        build_causal_receipt,
        validate_causal_receipt,
    )

    receipt["causal_receipt"] = build_causal_receipt(receipt)
    try:
        validate_causal_receipt(
            receipt["causal_receipt"],
            worker_receipt=receipt,
            require_complete=True,
        )
    except (TypeError, ValueError) as exc:
        raise EpisodeFault(
            f"sweep causal receipt is incomplete: {exc}",
            receipt=receipt,
        ) from exc
    if runtime_identity.get("identity_bound") is not True:
        raise EpisodeFault("sweep source/runtime identity is unbound", receipt=receipt)


def _run_rlc(
    model,
    config,
    prompt_tokens: list[int],
    tokenizer,
    *,
    wall_clock_s: float = 720.0,
    verifier=None,
    objective: str = "",
    model_path: str = "",
    incumbent_artifact=None,
    worker_identity: dict[str, Any] | None = None,
    runtime_identity: dict[str, Any] | None = None,
    domain: str = "general",
    cognitive_context: list[dict[str, Any]] | None = None,
) -> tuple[str, dict[str, Any]]:
    from core.brain.llm.latent_cortex.engine import LatentCortexEngine
    from core.brain.llm.latent_cortex.types import ComputeBudget

    engine = LatentCortexEngine(
        model,
        config=config,
        tokenizer=tokenizer,
        model_path=model_path or None,
    )
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
    # The ordinary control stops at exactly max_tokens. The engine default adds
    # a sentence-completion grace window, which made a retained incumbent longer
    # than its supposedly identical control on truncated answers.
    kwargs["decode_sentence_grace_tokens"] = 0
    if incumbent_artifact is not None:
        kwargs["incumbent_artifact"] = incumbent_artifact
    kwargs["domain"] = str(domain or "general")
    if cognitive_context:
        kwargs["cognitive_context"] = cognitive_context
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
    if model_path:
        _bind_sweep_runtime_identity(
            receipt,
            config=config,
            budget=budget,
            wall_clock_s=wall_clock_s,
            verifier=verifier,
            objective=objective,
            domain=domain,
            incumbent_artifact=incumbent_artifact,
            worker_identity=worker_identity,
            runtime_identity=runtime_identity,
        )
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
                f"reason={reason!r} termination={termination!r}",
                receipt=receipt,
            )
        # A policy failure is a real observation: the arm did not answer.
        # It is graded on exactly the text it managed to emit.
    # Belt and braces on the same hazard: if any fallback did serve an ordinary
    # decode, this is not an observation of the recurrent path.
    flags = [str(flag) for flag in (receipt.get("honest_flags") or [])]
    if any("fallback" in flag or "vanilla" in flag for flag in flags):
        raise EpisodeFault(
            f"episode degraded to an ordinary decode: flags={flags}",
            receipt=receipt,
        )
    return text, receipt


def _promotion_assessment(**kwargs: Any) -> tuple[str, dict[str, Any]]:
    from tools.rlc_complete_system_closed_book import _promotion_assessment as assess

    return assess(**kwargs)


def _run_complete_system_closed_book(*args: Any, **kwargs: Any) -> tuple[str, dict[str, Any]]:
    from tools.rlc_complete_system_closed_book import (
        _run_complete_system_closed_book as run,
    )

    return run(*args, **kwargs)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seed", type=int, default=20260807)
    parser.add_argument("--per-domain", type=int, default=4)
    parser.add_argument(
        "--difficulty",
        type=int,
        choices=(1, 2, 3),
        default=2,
        help=(
            "Committed task-generator difficulty. Calibrate with control-only "
            "seeds, then evaluate treatment on disjoint held-out seeds."
        ),
    )
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

    requested_names = {v.strip() for v in args.arms.split(",") if v.strip()}
    try:
        selected = _expand_requested_arms(requested_names)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    if not selected:
        print("no arms selected", file=sys.stderr)
        return 2

    # Tasks are generated from a committed seed so a resumed process, and any
    # independent replay, reconstructs exactly the same battery.
    seeds = [args.seed + i for i in range(args.per_domain)]
    tasks = ft.generate_task_battery(seeds, difficulty=args.difficulty)
    manifest = ft.build_task_manifest(tasks)
    commitment = ft.build_task_commitment(manifest)
    _atomic_write(
        out_dir / "task_commitment.json",
        json.dumps(
            {
                "schema": SWEEP_SCHEMA,
                "seed": args.seed,
                "per_domain": args.per_domain,
                "difficulty": args.difficulty,
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
        print(
            json.dumps(
                {
                    "tasks": len(tasks),
                    "requested_arms": sorted(requested_names),
                    "execution_arms": [a.name for a in selected],
                },
                indent=2,
            )
        )
        return 0

    arm_tokens = {
        a.name: (args.max_tokens if a.max_tokens is None else a.max_tokens) for a in selected
    }
    implementation_files = _implementation_manifest()
    implementation_sha256 = _implementation_sha256(implementation_files)
    fingerprints = {
        name: decode_fingerprint(
            model=args.model,
            n_slots=args.n_slots,
            max_tokens=tokens,
            episode_wall_s=args.episode_wall_s,
            seed=args.seed,
            per_domain=args.per_domain,
            difficulty=args.difficulty,
            arm=name,
            adapter=args.adapter,
            implementation_sha256=implementation_sha256,
        )
        for name, tokens in arm_tokens.items()
    }
    _atomic_write(
        out_dir / "decode_fingerprint.json",
        json.dumps(
            {
                "schema": EVIDENCE_MANIFEST_SCHEMA,
                "decode_fingerprint": fingerprints,
                "arm_max_tokens": arm_tokens,
                "difficulty": args.difficulty,
                "implementation_files": implementation_files,
                "implementation_sha256": implementation_sha256,
                "requested_arms": sorted(requested_names),
                "required_arms": [a.name for a in selected],
                "expected_task_ids": [task.task_id for task in tasks],
                "task_commitment_sha256": commitment.commitment_sha256,
            },
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

    with (
        standalone_model_lane(
            owner_id=f"rlc-reconciliation-sweep:{os.getpid()}",
            model_path=args.model,
            purpose="evaluation",
            preemptible=False,
            metadata={"tool": "run_rlc_reconciliation_sweep", "operator_launched": True},
        ),
        mlx_memory_envelope(fraction=args.memory_fraction) as envelope,
    ):
        print(f"memory envelope: {envelope.to_receipt()}", flush=True)
        model, tokenizer = load(args.model)
        print("model loaded", flush=True)
        from core.brain.llm.latent_cortex.governance import (
            checkpoint_file_fingerprint,
        )
        from core.brain.llm.latent_cortex.incumbent_artifact import (
            build_incumbent_artifact,
            incumbent_artifact_from_value,
            incumbent_artifact_to_value,
            validate_incumbent_artifact,
        )

        checkpoint = checkpoint_file_fingerprint(args.model)
        if (
            checkpoint.get("method") != "sha256"
            or not isinstance(checkpoint.get("fingerprint"), str)
            or len(checkpoint["fingerprint"]) != 64
        ):
            print(
                "resident checkpoint lacks a cryptographic file fingerprint",
                file=sys.stderr,
            )
            return 2
        tasks_by_id = {task.task_id: task for task in tasks}
        incumbent_by_task: dict[str, Any] = {}
        vanilla_latency_by_task: dict[str, float] = {}
        for cell in journal.cells():
            if cell.get("arm") != "vanilla" or cell.get("error"):
                continue
            task = tasks_by_id.get(str(cell.get("task_id") or ""))
            if task is None:
                continue
            artifact = incumbent_artifact_from_value(cell.get("incumbent_artifact") or {})
            incumbent_by_task[task.task_id] = validate_incumbent_artifact(
                artifact,
                input_tokens=_render_prompt(tokenizer, task),
                checkpoint_fingerprint=checkpoint["fingerprint"],
                checkpoint_fingerprint_method=checkpoint["method"],
                max_tokens=int(cell.get("decode_max_tokens") or args.max_tokens),
                n_layers=len(model.model.layers),
                decode=lambda values: tokenizer.decode(list(values)),
            )
            vanilla_latency_by_task[task.task_id] = float(cell.get("latency_s") or 0.0)
        if args.adapter:
            from core.brain.llm.latent_cortex.resident_adapter_loader import (
                load_resident_adapter,
            )

            manifest_path = Path(args.adapter_manifest) if args.adapter_manifest else None
            if manifest_path is None or not manifest_path.exists():
                found = list(Path(args.adapter).parent.rglob("recurrence_adapter_manifest.json"))
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

        from core.brain.llm.latent_cortex.runtime_identity import (
            build_worker_identity,
            collect_latent_runtime_identity,
        )
        from core.brain.llm.latent_cortex.worker_capture_identity import (
            build_worker_capture_identity,
        )

        signing_identity = build_worker_capture_identity(worker_boot_id=uuid.uuid4().hex)
        worker_identity = build_worker_identity(
            model,
            model_path=args.model,
            worker_boot_id=signing_identity.public_identity["worker_boot_id"],
            worker_source_path=Path(__file__).resolve(),
            worker_action_capture_identity=signing_identity.public_identity,
            tokenizer=tokenizer,
        )
        runtime_identity = collect_latent_runtime_identity(REPO_ROOT)

        for spec in selected:
            arm, steps, policy = spec.name, spec.steps, spec.policy
            tokens = arm_tokens[arm]
            config = (
                None
                if steps is None
                else _build_config(steps, args.n_slots, policy, tokens, profile=spec.profile)
            )
            for index, task in enumerate(tasks):
                key = (arm, task.task_id)
                if key in journal.done:
                    continue
                if time.monotonic() - started > args.max_wall_s:
                    _status(out_dir, phase="wall_budget_reached", arm=arm)
                    print("wall budget reached; exiting for clean resume", flush=True)
                    return 3
                if yield_requested(out_dir):
                    _status(out_dir, phase="yielded", arm=arm)
                    print(
                        f"yield requested: released the model after "
                        f"{len(journal.done)} committed cells. Re-running the "
                        f"launch script resumes here; delete the YIELD file "
                        f"first.",
                        flush=True,
                    )
                    return 4
                cell_started = time.monotonic()
                error = ""
                receipt: dict[str, Any] = {}
                text = ""
                cell_incumbent = None
                try:
                    if config is None and spec.profile == "ordinary_best_of_3":
                        text = _run_vanilla_best_of(
                            model,
                            tokenizer,
                            _render_prompt_text(tokenizer, task),
                            tokens,
                            samples=3,
                            campaign_seed=args.seed,
                            task_id=task.task_id,
                        )
                    elif config is None:
                        prompt_tokens = _render_prompt(tokenizer, task)
                        text, output_tokens, termination = _run_vanilla(
                            model,
                            tokenizer,
                            prompt_tokens,
                            tokens,
                        )
                        if arm == "vanilla":
                            cell_incumbent = build_incumbent_artifact(
                                input_tokens=prompt_tokens,
                                output_tokens=output_tokens,
                                output_text=text,
                                checkpoint_fingerprint=checkpoint["fingerprint"],
                                checkpoint_fingerprint_method=checkpoint["method"],
                                max_tokens=tokens,
                                n_layers=len(model.model.layers),
                                termination=termination,
                            )
                            incumbent_by_task[task.task_id] = cell_incumbent
                    else:
                        # The verifier ablation. An oracle arm is a
                        # diagnostic ceiling that separates a generation
                        # limit from a selection limit; it is never a
                        # capability claim and never promotable, which the
                        # arm name carries so no downstream reader can lose
                        # track of which one produced a number.
                        verifier = None
                        if spec.profile in {"complete_closed_book", "full"}:
                            verifier = _episode_verifier(task)
                        elif spec.profile == "full_oracle":
                            verifier = _oracle_verifier(task)
                        incumbent = (
                            incumbent_by_task.get(task.task_id)
                            if spec.profile in {"complete_closed_book", "full", "full_oracle"}
                            else None
                        )
                        if (
                            spec.profile in {"complete_closed_book", "full", "full_oracle"}
                            and incumbent is None
                        ):
                            raise EpisodeFault(
                                "paired canonical ordinary-decode incumbent is absent"
                            )
                        if spec.profile == "complete_closed_book":
                            text, receipt = _run_complete_system_closed_book(
                                model,
                                config,
                                _render_prompt(tokenizer, task),
                                tokenizer,
                                task=task,
                                max_tokens=tokens,
                                wall_clock_s=args.episode_wall_s,
                                model_path=args.model,
                                incumbent_artifact=incumbent,
                                worker_identity=worker_identity,
                                runtime_identity=runtime_identity,
                                campaign_seed=args.seed,
                            )
                        else:
                            text, receipt = _run_rlc(
                                model,
                                config,
                                _render_prompt(tokenizer, task),
                                tokenizer,
                                wall_clock_s=args.episode_wall_s,
                                verifier=verifier,
                                model_path=args.model,
                                incumbent_artifact=incumbent,
                                worker_identity=worker_identity,
                                runtime_identity=runtime_identity,
                                domain=task.domain,
                                objective=(
                                    task.public.prompt
                                    if spec.profile in {"full", "full_oracle"}
                                    else ""
                                ),
                            )
                except Exception as exc:  # noqa: BLE001 - recorded, never silent
                    # A harness fault must be visible as a fault. It is never
                    # scored as a wrong answer.
                    if isinstance(exc, EpisodeFault):
                        receipt = exc.receipt
                    error = f"{type(exc).__name__}: {exc}"
                    print(f"  !! {arm} {task.domain} {error}", flush=True)
                receipt_path = ""
                receipt_sha256 = ""
                if receipt:
                    receipt_path, receipt_sha256 = _persist_runtime_receipt(
                        out_dir,
                        arm=arm,
                        task_id=task.task_id,
                        receipt=receipt,
                    )
                incremental_latency_s = time.monotonic() - cell_started
                if arm == "vanilla" and not error:
                    vanilla_latency_by_task[task.task_id] = incremental_latency_s
                paired_incumbent_latency_s = (
                    vanilla_latency_by_task.get(task.task_id, 0.0)
                    if spec.profile in {"complete_closed_book", "full", "full_oracle"}
                    else 0.0
                )
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
                        "runtime_receipt_path": receipt_path,
                        "runtime_receipt_sha256": receipt_sha256,
                        "terminal_instruction_policy": policy,
                        # Latency is a first-class result, not a footnote:
                        # a unified system that answers better but takes ten
                        # minutes has not been shown to be deployable.
                        "steps_taken": receipt.get("steps_taken"),
                        "halted_early": receipt.get("halted_early"),
                        "phase_latency_s": receipt.get("phase_latency_s"),
                        # Whether the promotion chain actually fired, recorded
                        # per cell so the battery answers it directly. Every
                        # link in that chain has been zero at some point
                        # tonight -- probe budget, routing, verifier
                        # allowlists, dispute-only repair -- and diagnosing it
                        # meant separate 32B probe runs each time.
                        "route_counts": _route_counts(receipt),
                        "repair_requests": len(
                            (receipt.get("local_repair") or {}).get("requests") or []
                        ),
                        "answer_replacement_decision": (
                            (
                                (receipt.get("complete_system_closed_book") or {})
                                .get("promotion", {})
                                .get("decision")
                            )
                            if spec.profile == "complete_closed_book"
                            else (receipt.get("answer_replacement") or {}).get("decision")
                        ),
                        "answer_replacement_reason": (
                            (
                                (receipt.get("complete_system_closed_book") or {})
                                .get("promotion", {})
                                .get("reason")
                            )
                            if spec.profile == "complete_closed_book"
                            else (receipt.get("answer_replacement") or {}).get("reason")
                        ),
                        "full_stack_evidence": (
                            _full_stack_evidence(receipt)
                            if spec.profile in {"complete_closed_book", "full", "full_oracle"}
                            else None
                        ),
                        "complete_system_evidence": (
                            _complete_system_evidence(receipt)
                            if spec.profile == "complete_closed_book"
                            else None
                        ),
                        "incumbent_artifact": (
                            incumbent_artifact_to_value(cell_incumbent)
                            if cell_incumbent is not None
                            else None
                        ),
                        "text": text,
                        "error": error,
                        # A product request pays for the ordinary incumbent and
                        # the incremental full-stack search. The experiment
                        # reuses the exact paired artifact for causal identity,
                        # but does not pretend that generating it was free.
                        "latency_s": (incremental_latency_s + paired_incumbent_latency_s),
                        "incremental_latency_s": incremental_latency_s,
                        "paired_incumbent_latency_s": paired_incumbent_latency_s,
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
    """Grade a complete, paired experiment without letting controls win it.

    The product invariant is stronger than aggregate non-inferiority: ordinary
    decode owns every task until independently verified evidence promotes a
    candidate. Therefore a vanilla-right/full-stack-wrong pair is a contract
    failure, not a scientific finding that recurrence happened to lose.
    """
    from core.brain.llm.latent_cortex import frontier_tasks as ft

    tasks = tuple(tasks)
    by_id = {t.task_id: t for t in tasks}
    # Grading is bound to the same configuration the cells were produced under.
    # Absent the record (an older run, or a unit test), every cell is admitted.
    recorded = _read_json(out_dir / "decode_fingerprint.json")
    manifest_issues = _manifest_integrity_issues(recorded, tasks)
    journal = Journal(
        out_dir / "journal.jsonl",
        (recorded or {}).get("decode_fingerprint"),
    )
    cells = journal.cells()
    duplicate_cells: list[str] = []
    unique_cells: dict[tuple[str, str], dict[str, Any]] = {}
    for cell in cells:
        key = (str(cell.get("arm") or ""), str(cell.get("task_id") or ""))
        if key in unique_cells:
            duplicate_cells.append(f"{key[0]}:{key[1]}")
            continue
        unique_cells[key] = cell

    arms: dict[str, dict[str, Any]] = {}
    scored: dict[str, dict[str, dict[str, Any]]] = {}
    unknown_task_cells: list[str] = []
    for cell in unique_cells.values():
        arm = cell["arm"]
        bucket = arms.setdefault(
            arm,
            {
                "correct": 0,
                "contract_neutral_correct": 0,
                "contract_normalizations_admitted": 0,
                "total": 0,
                "errors": 0,
                "reasons": {},
                "contract_neutral_reasons": {},
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
            unknown_task_cells.append(f"{arm}:{cell['task_id']}")
            continue
        result = ft.score_task(task, cell["text"])
        neutral_result, normalized = _contract_neutral_score(task, cell["text"])
        bucket["correct"] += int(result.correct)
        bucket["contract_neutral_correct"] += int(neutral_result.correct)
        bucket["contract_normalizations_admitted"] += int(normalized)
        scored.setdefault(arm, {})[cell["task_id"]] = {
            "correct": bool(result.correct),
            "contract_neutral_correct": bool(neutral_result.correct),
            "text": str(cell.get("text") or ""),
            "replacement_decision": str(cell.get("answer_replacement_decision") or ""),
        }
        reason = result.reason or "correct"
        bucket["reasons"][reason] = bucket["reasons"].get(reason, 0) + 1
        neutral_reason = neutral_result.reason or "correct"
        bucket["contract_neutral_reasons"][neutral_reason] = (
            bucket["contract_neutral_reasons"].get(neutral_reason, 0) + 1
        )
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
        bucket["latency_median_s"] = round(lat[len(lat) // 2], 1) if lat else None
        bucket["latency_p90_s"] = round(lat[max(0, int(len(lat) * 0.9) - 1)], 1) if lat else None
        bucket["steps_median"] = sorted(steps)[len(steps) // 2] if steps else None
        bucket["halted_early_fraction"] = (
            round(bucket["halted_early"] / bucket["total"], 2) if bucket.get("total") else None
        )

    expected_task_ids = set((recorded or {}).get("expected_task_ids") or [])
    required_arms = list((recorded or {}).get("required_arms") or [])
    missing_cells: dict[str, list[str]] = {}
    if expected_task_ids and required_arms:
        for arm in required_arms:
            present = {task_id for candidate_arm, task_id in unique_cells if candidate_arm == arm}
            missing = sorted(expected_task_ids - present)
            if missing:
                missing_cells[arm] = missing

    complete_system_measured = "complete_system_closed_book" in arms

    def claim_eligible(name: str) -> bool:
        # Once the complete closed-book treatment is present, the narrower
        # neural-engine arm becomes an ablation. This prevents a stronger
        # campaign from being declared positive because an older, incomplete
        # experimental object happened to score higher. Historical manifests
        # without the new arm retain their original full_stack interpretation.
        if complete_system_measured:
            return name == "complete_system_closed_book"
        if name in CLAIM_ARM_NAMES:
            return True
        # Compatibility for historical fixtures and evidence. Controls and
        # explicit ablations can never become a claimant.
        return bool(name.startswith("rlc_") and name != "rlc_mechanism" and "oracle" not in name)

    vanilla = arms.get("vanilla", {}).get("correct", 0)
    vanilla_contract_neutral = arms.get("vanilla", {}).get(
        "contract_neutral_correct",
        0,
    )
    equal_compute = arms.get("vanilla_equal_compute", {}).get("correct")
    equal_compute_contract_neutral = arms.get("vanilla_equal_compute", {}).get(
        "contract_neutral_correct"
    )
    best_rlc_name, best_rlc = "", -1
    best_rlc_contract_neutral = -1
    for name, bucket in arms.items():
        if not claim_eligible(name):
            continue
        if bucket["correct"] > best_rlc:
            best_rlc_name, best_rlc = name, bucket["correct"]
        best_rlc_contract_neutral = max(
            best_rlc_contract_neutral,
            int(bucket["contract_neutral_correct"]),
        )

    floor_violations: list[str] = []
    incumbent_byte_violations: list[str] = []
    vanilla_rows = scored.get("vanilla", {})
    for arm, rows in scored.items():
        if not claim_eligible(arm):
            continue
        for task_id, candidate in rows.items():
            baseline = vanilla_rows.get(task_id)
            if baseline is None:
                continue
            if baseline["correct"] and not candidate["correct"]:
                floor_violations.append(f"{arm}:{task_id}")
            if (
                candidate["replacement_decision"] not in {"replace", ""}
                and candidate["text"] != baseline["text"]
            ):
                incumbent_byte_violations.append(f"{arm}:{task_id}")

    mechanism_issues: dict[str, dict[str, list[str]]] = {}
    enforce_runtime_evidence = bool(recorded and recorded.get("schema") == EVIDENCE_MANIFEST_SCHEMA)
    if enforce_runtime_evidence:
        for (arm, task_id), cell in unique_cells.items():
            if arm not in RUNTIME_STACK_ARM_NAMES:
                continue
            evidence = (
                cell.get("complete_system_evidence")
                if arm == "complete_system_closed_book"
                else cell.get("full_stack_evidence")
            )
            issues = (
                list(evidence.get("issues") or [])
                if isinstance(evidence, dict)
                else [
                    "complete_system_runtime_evidence_absent"
                    if arm == "complete_system_closed_book"
                    else "full_stack_runtime_evidence_absent"
                ]
            )
            issues.extend(_runtime_receipt_issues(out_dir, cell))
            if issues:
                mechanism_issues.setdefault(arm, {})[task_id] = sorted(set(issues))

    # An arm carrying harness faults has not been measured. Concluding either
    # way from it would report a starved budget as a reasoning result.
    faulted = {name: b["errors"] for name, b in arms.items() if b["errors"]}
    coverage_complete = not (
        manifest_issues or missing_cells or duplicate_cells or unknown_task_cells
    )
    mechanism_complete = not mechanism_issues
    complete = not faulted and coverage_complete and mechanism_complete
    # A battery the ordinary decode cannot score on has not measured the
    # recurrent path either: 0 >= 0 satisfies every inequality below, so mutual
    # failure would otherwise be published as parity and promote a model that
    # answered nothing. Parity is a claim about a baseline, and with no solved
    # control task there is no baseline to be at parity with. The floor is
    # structural (a baseline exists / does not), not a tuned threshold.
    informative = vanilla > 0
    contract_neutral_informative = vanilla_contract_neutral > 0
    # No recurrent arm ran, so nothing about the recurrent path was observed.
    # The sentinel -1 is smaller than any vanilla score, which would otherwise
    # publish "below ordinary decode" as a finding drawn from no data at all.
    measured_recurrence = best_rlc >= 0
    floor_holds = not floor_violations and not incumbent_byte_violations
    reaches_parity = bool(
        complete and informative and measured_recurrence and floor_holds and best_rlc >= vanilla
    )
    # ``vanilla_equal_compute`` is retained as an artifact-compatible arm name,
    # but its implementation is fixed best-of-three. The CP080 real-checkpoint
    # canary measured a 24x complete-system/vanilla latency ratio while this arm
    # cost only 3.8x. Without per-cell ResourceLedger comparison certificates it
    # has no equal-compute claim authority, regardless of observed accuracy.
    resource_matched_control_proven = False
    outscored_preliminary_best_of_three = bool(
        reaches_parity and equal_compute is not None and best_rlc > int(equal_compute)
    )
    beats_equal_compute = bool(
        resource_matched_control_proven and outscored_preliminary_best_of_three
    )
    contract_neutral_reaches_parity = bool(
        complete
        and contract_neutral_informative
        and measured_recurrence
        and floor_holds
        and best_rlc_contract_neutral >= vanilla_contract_neutral
    )
    contract_neutral_outscored_best_of_three = bool(
        contract_neutral_reaches_parity
        and equal_compute_contract_neutral is not None
        and best_rlc_contract_neutral > int(equal_compute_contract_neutral)
    )
    contract_neutral_beats_equal_compute = bool(
        resource_matched_control_proven and contract_neutral_outscored_best_of_three
    )
    if not complete:
        if manifest_issues:
            decision = "inconclusive_evidence_manifest_invalid"
        elif faulted:
            decision = "inconclusive_arms_carry_harness_faults"
        elif not coverage_complete:
            decision = "inconclusive_campaign_incomplete"
        else:
            decision = "inconclusive_full_stack_runtime_unmeasured"
    elif not floor_holds:
        decision = "invalid_full_stack_violated_vanilla_incumbent"
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
        "vanilla_equal_compute_correct": equal_compute,
        "control_contracts": {
            "vanilla_equal_compute": {
                "artifact_compatible_name": True,
                "selection": "best_of_3_self_consistency",
                "resource_matched": resource_matched_control_proven,
                "claim_authority": "preliminary_only",
                "required_claim_successor": "digest_bound_paired_resource_certificate",
            }
        },
        "contract_neutral_diagnostic": {
            "authority": "diagnostic_only_no_serving_fusion_or_claim_authority",
            "vanilla_correct": vanilla_contract_neutral,
            "vanilla_equal_compute_correct": equal_compute_contract_neutral,
            "best_recurrent_correct": best_rlc_contract_neutral,
            "battery_informative": contract_neutral_informative,
            "reaches_parity_with_ordinary_decode": (contract_neutral_reaches_parity),
            "beats_equal_compute_control": contract_neutral_beats_equal_compute,
            "outscored_preliminary_best_of_3": (contract_neutral_outscored_best_of_three),
        },
        "best_recurrent_arm": best_rlc_name,
        "best_recurrent_correct": best_rlc,
        "arms_complete": complete,
        "faulted_arms": faulted,
        "coverage_complete": coverage_complete,
        "evidence_manifest_valid": not manifest_issues,
        "evidence_manifest_issues": manifest_issues,
        "missing_cells": missing_cells,
        "duplicate_cells": sorted(duplicate_cells),
        "unknown_task_cells": sorted(unknown_task_cells),
        "full_stack_runtime_measured": mechanism_complete,
        "full_stack_runtime_issues": mechanism_issues,
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
        "beats_equal_compute_control": beats_equal_compute,
        "outscored_preliminary_best_of_3": outscored_preliminary_best_of_three,
        "resource_matched_control_proven": resource_matched_control_proven,
        "paired_vanilla_floor": {
            "holds": floor_holds,
            "right_to_wrong_regressions": sorted(floor_violations),
            "unpromoted_byte_divergences": sorted(incumbent_byte_violations),
        },
        "decision": decision,
        "claims": {
            "reasoning_gain_proven": False,
            "fusion_authorized": False,
            "frontier_level_proven": False,
        },
        "graded_unix": _now(),
    }
    _atomic_write(out_dir / "verdict.json", json.dumps(verdict, indent=1, sort_keys=True) + "\n")
    return verdict


if __name__ == "__main__":
    raise SystemExit(main())
