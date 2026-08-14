#!/usr/bin/env python3
"""Run the frozen 1.5B parent/treatment/process-lesion transfer canary."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, Final

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.brain.llm.latent_cortex.answer_contract import (  # noqa: E402
    ContractDecodeDisposition,
    contract_decode_disposition,
)
from core.brain.llm.latent_cortex.frontier_tasks import FrontierTask  # noqa: E402
from core.brain.llm.unified_recurrent_transfer_canary import (  # noqa: E402
    ARMS,
    seal_transfer_canary_plan,
    seal_transfer_canary_result,
)
from core.brain.llm.unified_recurrent_transfer_decode import (  # noqa: E402
    decode_base_greedy_tokens,
    decode_typed_process_tokens,
)
from tools.evaluate_unified_intrinsic_checkpoint import (  # noqa: E402
    _evaluation_layout,
    unified_evaluation_context,
)
from tools.run_unified_recurrent_broad_canary import (  # noqa: E402
    _append_private,
    _canonical_bytes,
    _create_or_verify,
    _ensure_private_directory,
    _issuer,
    _prompt_tokens,
    _read_lines,
    _task_identity,
    _tasks,
)
from tools.unified_intrinsic_checkpoint import (  # noqa: E402
    resolve_checkpoint_generation,
)

JOURNAL_SCHEMA: Final = "aura.unified_intrinsic.transfer_canary_journal.v1"
RUN_SCHEMA: Final = "aura.unified_intrinsic.transfer_canary_run.v1"
SOURCE_PATHS: Final = (
    "core/brain/llm/latent_cortex/frontier_tasks.py",
    "core/brain/llm/unified_recurrent_transfer_canary.py",
    "core/brain/llm/unified_recurrent_transfer_decode.py",
    "core/learning/frontier_process_supervision.py",
    "core/learning/unified_intrinsic_objective.py",
    "core/learning/unified_intrinsic_recurrence.py",
    "core/runtime/mlx_memory_guard.py",
    "tools/evaluate_unified_intrinsic_checkpoint.py",
    "tools/run_unified_recurrent_broad_canary.py",
    "tools/run_unified_recurrent_transfer_canary.py",
    "tools/unified_intrinsic_checkpoint.py",
)


class TransferCanaryRunnerError(RuntimeError):
    """The transfer canary could not retain a source-bound resumable run."""


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_binding(source_commit: str) -> dict[str, Any]:
    if (
        not isinstance(source_commit, str)
        or len(source_commit) != 40
        or any(character not in "0123456789abcdef" for character in source_commit)
    ):
        raise TransferCanaryRunnerError("transfer canary source commit is invalid")
    return {
        "git_commit": source_commit,
        "implementation_sha256s": {
            relative: _file_sha256(REPO_ROOT / relative)
            for relative in SOURCE_PATHS
        },
    }


def _contract_complete(tokenizer: Any, token_ids: Sequence[int]) -> bool:
    text = tokenizer.decode(list(token_ids), skip_special_tokens=True)
    return contract_decode_disposition(text) in {
        ContractDecodeDisposition.COMPLETE,
        ContractDecodeDisposition.INVALID,
    }


def _arm_order(task_id: str) -> tuple[str, ...]:
    offset = int(hashlib.sha256(task_id.encode()).hexdigest()[:8], 16) % len(ARMS)
    return ARMS[offset:] + ARMS[:offset]


def _progress_callback(
    *,
    task: FrontierTask,
    arm: str,
    maximum_tokens: int,
) -> Callable[[int], None]:
    def report(generated_tokens: int) -> None:
        print(
            json.dumps(
                {
                    "event": "token_generated",
                    "task_id": task.task_id,
                    "domain": task.domain,
                    "arm": arm,
                    "token_step": generated_tokens,
                    "maximum_tokens": maximum_tokens,
                },
                sort_keys=True,
            ),
            flush=True,
        )

    return report


def _candidate_rows(path: Path, *, plan_sha256: str) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for envelope in _read_lines(path):
        candidate = envelope.get("candidate")
        if (
            set(envelope)
            != {"schema", "plan_sha256", "candidate", "raw_text", "score"}
            or envelope.get("schema") != JOURNAL_SCHEMA
            or envelope.get("plan_sha256") != plan_sha256
            or not isinstance(candidate, dict)
            or not isinstance(envelope.get("raw_text"), str)
            or not isinstance(envelope.get("score"), dict)
        ):
            raise TransferCanaryRunnerError("transfer canary journal identity differs")
        key = (str(candidate.get("task_id")), str(candidate.get("arm")))
        if key in seen:
            raise TransferCanaryRunnerError("transfer canary journal contains a duplicate arm")
        seen.add(key)
        candidates.append(candidate)
    return candidates


def _run_loaded(
    *,
    output_dir: Path,
    plan: Mapping[str, Any],
    tasks: Sequence[FrontierTask],
    model: Any,
    tokenizer: Any,
    spec: Any,
    parent_controller: Any,
    treatment_controller: Any,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    journal_path = output_dir / "candidates.jsonl"
    candidates = _candidate_rows(journal_path, plan_sha256=str(plan["plan_sha256"]))
    completed = {(row["task_id"], row["arm"]) for row in candidates}
    task_by_id = {task.task_id: task for task in tasks}
    if set(task_by_id) != {row["task_id"] for row in plan["tasks"]}:
        raise TransferCanaryRunnerError("transfer canary private task reconstruction differs")
    if parent_controller.parameter_sha256() != plan["parent_controller_sha256"]:
        raise TransferCanaryRunnerError("transfer canary parent controller differs")
    if treatment_controller.parameter_sha256() != plan["treatment_controller_sha256"]:
        raise TransferCanaryRunnerError("transfer canary treatment controller differs")
    total = len(tasks) * len(ARMS)
    for task in tasks:
        public_tokens = _prompt_tokens(tokenizer, task)
        for arm in _arm_order(task.task_id):
            key = (task.task_id, arm)
            if key in completed:
                continue
            maximum_tokens = int(plan["max_tokens"])
            print(
                json.dumps(
                    {
                        "event": "arm_started",
                        "task_id": task.task_id,
                        "domain": task.domain,
                        "arm": arm,
                        "completed_arms": len(completed),
                        "total_arms": total,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            progress = _progress_callback(
                task=task,
                arm=arm,
                maximum_tokens=maximum_tokens,
            )
            def complete(ids: tuple[int, ...]) -> bool:
                return _contract_complete(tokenizer, ids)
            if arm == "base_greedy":
                generated, stopped, latency_ms = decode_base_greedy_tokens(
                    model,
                    public_tokens,
                    eos_token_id=tokenizer.eos_token_id,
                    max_tokens=maximum_tokens,
                    completion_check=complete,
                    progress=progress,
                )
            else:
                controller = (
                    parent_controller if arm == "parent_typed" else treatment_controller
                )
                generated, stopped, latency_ms = decode_typed_process_tokens(
                    model,
                    controller,
                    spec,
                    public_tokens,
                    recurrence_depth=int(plan["recurrence_depth"]),
                    eos_token_id=tokenizer.eos_token_id,
                    max_tokens=maximum_tokens,
                    typed_action_lesion=arm == "action_lesion",
                    completion_check=complete,
                    progress=progress,
                )
            text = tokenizer.decode(list(generated), skip_special_tokens=True)
            score = task.score(text)
            candidate = {
                "task_id": task.task_id,
                "domain": task.domain,
                "arm": arm,
                "correct": bool(score.correct),
                "parsed": bool(score.parsed),
                "response_sha256": hashlib.sha256(text.encode()).hexdigest(),
                "generated_tokens": len(generated),
                "stopped": stopped,
                "latency_ms": latency_ms,
            }
            _append_private(
                journal_path,
                {
                    "schema": JOURNAL_SCHEMA,
                    "plan_sha256": plan["plan_sha256"],
                    "candidate": candidate,
                    "raw_text": text,
                    "score": score.to_dict(),
                },
            )
            candidates.append(candidate)
            completed.add(key)
            print(
                json.dumps(
                    {
                        "progress": f"{len(completed)}/{total}",
                        "task_id": task.task_id,
                        "domain": task.domain,
                        "arm": arm,
                        "correct": bool(score.correct),
                        "parsed": bool(score.parsed),
                        "latency_ms": latency_ms,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    result = seal_transfer_canary_result(plan, candidates)
    return candidates, result


def run_canary(
    *,
    campaign_dir: Path,
    stem: str,
    output_dir: Path,
    seeds: Sequence[int],
    difficulty: int,
    recurrence_depth: int,
    max_tokens: int,
    source_commit: str,
    memory_limit_gb: float,
    cache_limit_gb: float,
    wired_limit_gb: float,
    pressure_broker_vm_stat_path: Path | None = None,
    pressure_broker_swapusage_path: Path | None = None,
) -> dict[str, Any]:
    output_dir = _ensure_private_directory(output_dir)
    issuer = _issuer(output_dir, seeds, difficulty)
    tasks = _tasks(issuer)
    layout = _evaluation_layout(campaign_dir)
    resolved = resolve_checkpoint_generation(
        layout.checkpoint_dir,
        stem=stem,
        required=True,
    )
    if resolved is None:  # pragma: no cover - required=True is exhaustive
        raise TransferCanaryRunnerError("transfer canary treatment checkpoint is absent")
    started = time.time()
    with unified_evaluation_context(
        campaign_dir,
        stem=stem,
        memory_limit_gb=memory_limit_gb,
        cache_limit_gb=cache_limit_gb,
        wired_limit_gb=wired_limit_gb,
        pressure_broker_vm_stat_path=pressure_broker_vm_stat_path,
        pressure_broker_swapusage_path=pressure_broker_swapusage_path,
    ) as loaded:
        bundle, parent_controller, tokenizer, spec, identity, _envelope, _guard = loaded
        bootstrap = identity.get("bootstrap")
        if not isinstance(bootstrap, dict):
            raise TransferCanaryRunnerError("transfer canary has no imported parent")
        parent_sha256 = parent_controller.parameter_sha256()
        treatment_sha256 = bundle.controller.parameter_sha256()
        if identity.get("initial_controller_sha256") != parent_sha256:
            raise TransferCanaryRunnerError("transfer canary parent identity differs")
        plan = seal_transfer_canary_plan(
            campaign_identity_sha256=str(identity["identity_sha256"]),
            parent_checkpoint_sha256=str(bootstrap["parent_checkpoint_sha256"]),
            parent_controller_sha256=parent_sha256,
            treatment_checkpoint_sha256=str(resolved.receipt["checkpoint_sha256"]),
            treatment_controller_sha256=treatment_sha256,
            recurrence_depth=recurrence_depth,
            max_tokens=max_tokens,
            tasks=[_task_identity(task) for task in tasks],
            source_binding=_source_binding(source_commit),
        )
        _create_or_verify(output_dir / "plan.json", _canonical_bytes(plan), mode=0o400)
        candidates, verdict = _run_loaded(
            output_dir=output_dir,
            plan=plan,
            tasks=tasks,
            model=bundle.model,
            tokenizer=tokenizer,
            spec=spec,
            parent_controller=parent_controller,
            treatment_controller=bundle.controller,
        )
    body = {
        "schema": RUN_SCHEMA,
        "plan_sha256": plan["plan_sha256"],
        "candidate_count": len(candidates),
        "verdict": verdict,
        "started_at_unix": started,
        "completed_at_unix": time.time(),
        "runner_sha256": _file_sha256(Path(__file__).resolve()),
        "serving_authority": False,
    }
    result = {**body, "run_sha256": _sha(body)}
    _create_or_verify(
        output_dir / "run-complete.json",
        _canonical_bytes(result),
        mode=0o400,
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-dir", type=Path, required=True)
    parser.add_argument("--stem", default="checkpoint_latest")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, action="append", required=True)
    parser.add_argument("--difficulty", type=int, default=2, choices=(1, 2, 3))
    parser.add_argument("--recurrence-depth", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--memory-limit-gb", type=float, default=40.0)
    parser.add_argument("--cache-limit-gb", type=float, default=2.0)
    parser.add_argument("--wired-limit-gb", type=float, default=48.0)
    parser.add_argument("--pressure-broker-vm-stat-path", type=Path)
    parser.add_argument("--pressure-broker-swapusage-path", type=Path)
    arguments = parser.parse_args()
    if len(set(arguments.seed)) != len(arguments.seed):
        parser.error("--seed values must be unique")
    result = run_canary(
        campaign_dir=arguments.campaign_dir.expanduser().resolve(strict=True),
        stem=arguments.stem,
        output_dir=arguments.output_dir.expanduser(),
        seeds=tuple(arguments.seed),
        difficulty=arguments.difficulty,
        recurrence_depth=arguments.recurrence_depth,
        max_tokens=arguments.max_tokens,
        source_commit=arguments.source_commit,
        memory_limit_gb=arguments.memory_limit_gb,
        cache_limit_gb=arguments.cache_limit_gb,
        wired_limit_gb=arguments.wired_limit_gb,
        pressure_broker_vm_stat_path=arguments.pressure_broker_vm_stat_path,
        pressure_broker_swapusage_path=arguments.pressure_broker_swapusage_path,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["verdict"]["supported"] is True else 2


if __name__ == "__main__":
    raise SystemExit(main())
