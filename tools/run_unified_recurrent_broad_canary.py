#!/usr/bin/env python3
"""Run a resumable broad canary on certified resident recurrent tissue."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import secrets
import stat
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import mlx.core as mx  # noqa: E402

from core.brain.llm.latent_cortex.answer_contract import (  # noqa: E402
    ContractDecodeDisposition,
    contract_decode_disposition,
)
from core.brain.llm.latent_cortex.frontier_tasks import (  # noqa: E402
    CONTAMINATION_SAFE_REGISTRY_VERSION,
    FRONTIER_DOMAINS,
    FrontierTask,
    generate_task_battery,
    reblind_frontier_task,
)
from core.brain.llm.unified_recurrent_broad_canary import (  # noqa: E402
    ARMS,
    broad_canary_plan_errors,
    seal_broad_canary_plan,
    seal_broad_canary_result,
)
from core.brain.llm.unified_recurrent_qualified_activation_store import (  # noqa: E402
    read_qualified_activation,
)
from core.brain.llm.unified_recurrent_shadow import (  # noqa: E402
    inspect_shadow_package,
    load_unified_recurrent_shadow,
)
from core.learning.unified_intrinsic_recurrence import (  # noqa: E402
    UnifiedRecurrentController,
)
from core.runtime.model_lane_control import standalone_model_lane  # noqa: E402

ISSUER_SCHEMA: Final = "aura.unified_intrinsic.broad_canary_issuer.v1"
JOURNAL_SCHEMA: Final = "aura.unified_intrinsic.broad_canary_journal.v1"
RUN_SCHEMA: Final = "aura.unified_intrinsic.broad_canary_run.v1"
SOURCE_PATHS: Final = (
    "core/brain/llm/latent_cortex/frontier_tasks.py",
    "core/brain/llm/unified_recurrent_broad_canary.py",
    "core/brain/llm/unified_recurrent_shadow.py",
    "core/learning/unified_intrinsic_recurrence.py",
    "tools/run_unified_recurrent_broad_canary.py",
)


class BroadCanaryRunnerError(RuntimeError):
    """The broad canary could not retain a source-bound resumable run."""


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _ensure_private_directory(path: Path) -> Path:
    target = path.expanduser().absolute()
    target.mkdir(mode=0o700, parents=True, exist_ok=True)
    current = Path(target.anchor)
    for part in target.parts[1:]:
        current /= part
        metadata = current.lstat()
        if stat.S_ISLNK(metadata.st_mode):
            raise BroadCanaryRunnerError("broad canary path contains a symlink")
    metadata = target.stat()
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or metadata.st_uid != os.geteuid()
        or stat.S_IMODE(metadata.st_mode) & 0o077
    ):
        raise BroadCanaryRunnerError("broad canary directory custody differs")
    return target


def _create_or_verify(path: Path, payload: bytes, *, mode: int = 0o400) -> None:
    if path.exists() or path.is_symlink():
        if path.is_symlink() or path.read_bytes() != payload:
            raise BroadCanaryRunnerError(
                f"broad canary immutable artifact differs: {path.name}"
            )
        return
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0),
        mode,
    )
    try:
        if os.write(descriptor, payload) != len(payload):
            raise BroadCanaryRunnerError(
                f"broad canary artifact write was short: {path.name}"
            )
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _append_private(path: Path, value: Mapping[str, Any]) -> None:
    payload = _canonical_bytes(dict(value)) + b"\n"
    descriptor = os.open(
        path,
        os.O_WRONLY
        | os.O_APPEND
        | os.O_CREAT
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        if os.write(descriptor, payload) != len(payload):
            raise BroadCanaryRunnerError(
                f"broad canary journal write was short: {path.name}"
            )
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _read_lines(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    if path.is_symlink():
        raise BroadCanaryRunnerError("broad canary journal is a symlink")
    rows: list[dict[str, Any]] = []
    for line in path.read_bytes().splitlines():
        try:
            row = json.loads(line.decode("ascii"))
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise BroadCanaryRunnerError("broad canary journal is invalid") from exc
        if not isinstance(row, dict) or _canonical_bytes(row) != line:
            raise BroadCanaryRunnerError("broad canary journal is non-canonical")
        rows.append(row)
    return rows


def _issuer(output_dir: Path, seeds: Sequence[int], difficulty: int) -> dict[str, Any]:
    path = output_dir / "issuer-private.json"
    if path.exists() or path.is_symlink():
        try:
            value = json.loads(path.read_text(encoding="ascii"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise BroadCanaryRunnerError("broad canary issuer is invalid") from exc
        body = {key: item for key, item in value.items() if key != "issuer_sha256"}
        if (
            not isinstance(value, dict)
            or value.get("schema") != ISSUER_SCHEMA
            or value.get("seeds") != list(seeds)
            or value.get("difficulty") != difficulty
            or value.get("registry_version") != CONTAMINATION_SAFE_REGISTRY_VERSION
            or value.get("domains") != list(FRONTIER_DOMAINS)
            or value.get("issuer_sha256") != _sha(body)
            or len(value.get("blind_nonces", []))
            != len(seeds) * len(FRONTIER_DOMAINS)
        ):
            raise BroadCanaryRunnerError("broad canary issuer identity differs")
        return value
    body = {
        "schema": ISSUER_SCHEMA,
        "seeds": list(seeds),
        "difficulty": difficulty,
        "registry_version": CONTAMINATION_SAFE_REGISTRY_VERSION,
        "domains": list(FRONTIER_DOMAINS),
        "blind_nonces": [
            secrets.token_bytes(32).hex()
            for _index in range(len(seeds) * len(FRONTIER_DOMAINS))
        ],
    }
    value = {**body, "issuer_sha256": _sha(body)}
    _create_or_verify(path, _canonical_bytes(value), mode=0o400)
    return value


def _tasks(issuer: Mapping[str, Any]) -> tuple[FrontierTask, ...]:
    tasks = generate_task_battery(
        issuer["seeds"],
        domains=FRONTIER_DOMAINS,
        difficulty=int(issuer["difficulty"]),
        registry_version=str(issuer["registry_version"]),
    )
    nonces = issuer["blind_nonces"]
    return tuple(
        reblind_frontier_task(task, blind_nonce=bytes.fromhex(str(nonces[index])))
        for index, task in enumerate(tasks)
    )


def _task_identity(task: FrontierTask) -> dict[str, Any]:
    public = task.public
    return {
        "task_id": task.task_id,
        "domain": task.domain,
        "task_payload_sha256": public.task_payload_sha256,
        "answer_commitment_sha256": public.answer_commitment_sha256,
        "prompt_sha256": hashlib.sha256(public.prompt.encode()).hexdigest(),
    }


def _source_binding(source_commit: str) -> dict[str, Any]:
    if (
        not isinstance(source_commit, str)
        or len(source_commit) != 40
        or any(character not in "0123456789abcdef" for character in source_commit)
    ):
        raise BroadCanaryRunnerError("broad canary source commit is invalid")
    return {
        "git_commit": source_commit,
        "implementation_sha256s": {
            path: _file_sha256(REPO_ROOT / path) for path in SOURCE_PATHS
        },
    }


def _prompt_tokens(tokenizer: Any, task: FrontierTask) -> tuple[int, ...]:
    value = tokenizer.apply_chat_template(
        [{"role": "user", "content": task.public.prompt}],
        add_generation_prompt=True,
        tokenize=True,
    )
    if hasattr(value, "tolist"):
        value = value.tolist()
    if (
        not isinstance(value, list)
        or not value
        or any(type(token_id) is not int or token_id < 0 for token_id in value)
    ):
        raise BroadCanaryRunnerError("broad canary prompt tokenization differs")
    return tuple(value)


def _contract_complete(tokenizer: Any, token_ids: Sequence[int]) -> bool:
    text = tokenizer.decode(list(token_ids), skip_special_tokens=True)
    return contract_decode_disposition(text) in {
        ContractDecodeDisposition.COMPLETE,
        ContractDecodeDisposition.INVALID,
    }


def _base_decode(
    model: Any,
    tokenizer: Any,
    public_tokens: Sequence[int],
    *,
    max_tokens: int,
) -> tuple[tuple[int, ...], bool, int]:
    tokens = mx.array([list(public_tokens)], dtype=mx.int32)
    generated: list[int] = []
    stopped = False
    started = time.perf_counter()
    for _index in range(max_tokens):
        output = model(tokens)
        logits = (
            output.logits
            if hasattr(output, "logits")
            else output[0]
            if isinstance(output, tuple)
            else output
        )
        token_id = int(mx.argmax(logits[0, -1]).item())
        generated.append(token_id)
        if tokenizer.eos_token_id is not None and token_id == tokenizer.eos_token_id:
            stopped = True
            break
        if _contract_complete(tokenizer, generated):
            stopped = True
            break
        tokens = mx.concatenate(
            [tokens, mx.array([[token_id]], dtype=tokens.dtype)],
            axis=1,
        )
    elapsed_ms = max(0, int(round((time.perf_counter() - started) * 1000.0)))
    return tuple(generated), stopped, elapsed_ms


def _arm_order(task_id: str) -> tuple[str, ...]:
    offset = int(hashlib.sha256(task_id.encode()).hexdigest()[:8], 16) % len(ARMS)
    return ARMS[offset:] + ARMS[:offset]


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
            raise BroadCanaryRunnerError("broad canary journal identity differs")
        key = (str(candidate.get("task_id")), str(candidate.get("arm")))
        if key in seen:
            raise BroadCanaryRunnerError("broad canary journal contains a duplicate arm")
        seen.add(key)
        candidates.append(candidate)
    return candidates


def _run_loaded(
    *,
    output_dir: Path,
    plan: Mapping[str, Any],
    tasks: Sequence[FrontierTask],
    loaded: Any,
    model: Any,
    tokenizer: Any,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    journal_path = output_dir / "candidates.jsonl"
    candidates = _candidate_rows(journal_path, plan_sha256=str(plan["plan_sha256"]))
    completed = {(row["task_id"], row["arm"]) for row in candidates}
    task_by_id = {task.task_id: task for task in tasks}
    if set(task_by_id) != {row["task_id"] for row in plan["tasks"]}:
        raise BroadCanaryRunnerError("broad canary private task reconstruction differs")
    initial = UnifiedRecurrentController(loaded.controller.config)
    mx.eval(initial.parameters())
    if initial.parameter_sha256() == loaded.controller.parameter_sha256():
        raise BroadCanaryRunnerError("trained controller equals initialization control")
    total = len(tasks) * len(ARMS)
    for task in tasks:
        public_tokens = _prompt_tokens(tokenizer, task)
        for arm in _arm_order(task.task_id):
            key = (task.task_id, arm)
            if key in completed:
                continue
            if arm == "base_greedy":
                generated, stopped, latency_ms = _base_decode(
                    model,
                    tokenizer,
                    public_tokens,
                    max_tokens=int(plan["max_tokens"]),
                )
            else:
                controller = initial if arm == "initial_t4" else loaded.controller
                depth = 1 if arm == "trained_t1" else int(plan["recurrence_depth"])
                generated, stopped, latency_ms = loaded.decode_general_recurrent_tokens(
                    model,
                    public_tokens,
                    max_tokens=int(plan["max_tokens"]),
                    recurrence_depth=depth,
                    controller=controller,
                    completion_check=lambda ids: _contract_complete(tokenizer, ids),
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
    result = seal_broad_canary_result(plan, candidates)
    return candidates, result


def run_canary(
    *,
    package: Path,
    model_path: Path,
    output_dir: Path,
    seeds: Sequence[int],
    difficulty: int,
    max_tokens: int,
    source_commit: str,
) -> dict[str, Any]:
    output_dir = _ensure_private_directory(output_dir)
    issuer = _issuer(output_dir, seeds, difficulty)
    tasks = _tasks(issuer)
    verified = inspect_shadow_package(package)
    manifest = verified["manifest"]
    activation = read_qualified_activation()
    if (
        activation.get("package_id") != manifest.get("package_id")
        or activation.get("manifest_sha256") != manifest.get("manifest_sha256")
        or activation.get("mode") != "qualified_typed_only"
        or activation.get("serving_authority") is not True
    ):
        raise BroadCanaryRunnerError(
            "broad canary package lacks matching qualified activation"
        )
    plan = seal_broad_canary_plan(
        package_id=str(manifest["package_id"]),
        manifest_sha256=str(manifest["manifest_sha256"]),
        controller_sha256=str(activation["controller_sha256"]),
        model_manifest_sha256=str(manifest["model_manifest_sha256"]),
        recurrence_depth=int(manifest["domain_contract"]["recurrence_depth"]),
        max_tokens=max_tokens,
        tasks=[_task_identity(task) for task in tasks],
        source_binding=_source_binding(source_commit),
    )
    if broad_canary_plan_errors(plan):
        raise BroadCanaryRunnerError("broad canary plan failed reopening")
    _create_or_verify(output_dir / "plan.json", _canonical_bytes(plan), mode=0o400)
    started = time.time()
    from mlx_lm import load

    with standalone_model_lane(
        owner_id=f"unified-broad-canary:{os.getpid()}",
        model_path=str(model_path),
        purpose="benchmark",
        preemptible=False,
        metadata={"tool": "run_unified_recurrent_broad_canary"},
    ):
        model, tokenizer = load(str(model_path))
        loaded = load_unified_recurrent_shadow(
            package,
            model=model,
            tokenizer=tokenizer,
            model_path=model_path,
        )
        if (
            loaded.receipt["package_id"] != plan["package_id"]
            or loaded.receipt["controller_sha256"] != plan["controller_sha256"]
        ):
            raise BroadCanaryRunnerError("loaded broad canary controller differs")
        candidates, verdict = _run_loaded(
            output_dir=output_dir,
            plan=plan,
            tasks=tasks,
            loaded=loaded,
            model=model,
            tokenizer=tokenizer,
        )
    body = {
        "schema": RUN_SCHEMA,
        "plan_sha256": plan["plan_sha256"],
        "package_id": plan["package_id"],
        "controller_sha256": plan["controller_sha256"],
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
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, action="append", required=True)
    parser.add_argument("--difficulty", type=int, default=2, choices=(1, 2, 3))
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--source-commit", required=True)
    arguments = parser.parse_args()
    if len(set(arguments.seed)) != len(arguments.seed):
        parser.error("--seed values must be unique")
    result = run_canary(
        package=arguments.package.expanduser().resolve(strict=True),
        model_path=arguments.model.expanduser().resolve(strict=True),
        output_dir=arguments.output_dir,
        seeds=tuple(arguments.seed),
        difficulty=arguments.difficulty,
        max_tokens=arguments.max_tokens,
        source_commit=arguments.source_commit,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["verdict"]["supported"] is True else 2


if __name__ == "__main__":
    raise SystemExit(main())
