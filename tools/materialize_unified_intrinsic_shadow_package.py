#!/usr/bin/env python3
"""Materialize a verified unified-recurrence controller as shadow-only tissue."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import stat
import sys
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final, Never

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.brain.llm.unified_recurrent_shadow_battery import (  # noqa: E402
    seal_shadow_canary_battery,
    validate_shadow_canary_battery,
)
from core.runtime.atomic_writer import atomic_write_bytes  # noqa: E402
from tools import adjudicate_unified_intrinsic_resident_replication as replication  # noqa: E402
from tools import launch_unified_intrinsic_resident_evaluation as launcher  # noqa: E402
from tools.unified_intrinsic_checkpoint import (  # noqa: E402
    resolve_checkpoint_generation,
)
from tools.unified_intrinsic_resident_identity import (  # noqa: E402
    canonical_bytes,
    canonical_sha256,
)

PACKAGE_SCHEMA: Final = "aura.unified_intrinsic.shadow_package.v2"
COMPLETE_SCHEMA: Final = "aura.unified_intrinsic.shadow_package_complete.v2"
_PACKAGE_ID = re.compile(r"[a-z0-9][a-z0-9._-]{0,119}")
_MAX_COPY_BYTES: Final = 64 * 1024 * 1024 * 1024


class UnifiedIntrinsicShadowPackageError(RuntimeError):
    """Shadow package evidence or storage is incomplete or inconsistent."""


def _fail(message: str) -> Never:
    raise UnifiedIntrinsicShadowPackageError(message)


def _private_directory(path: Path, *, create: bool) -> Path:
    lexical = path.expanduser().absolute()
    if create:
        lexical.mkdir(parents=True, exist_ok=True, mode=0o700)
        lexical.chmod(0o700)
    current = Path(lexical.anchor)
    for part in lexical.parts[1:]:
        current /= part
        try:
            observed = current.lstat()
        except OSError as exc:
            raise UnifiedIntrinsicShadowPackageError(
                f"shadow package directory is unavailable: {current}"
            ) from exc
        if stat.S_ISLNK(observed.st_mode):
            _fail(f"shadow package directory is a symlink: {current}")
    resolved = lexical.resolve(strict=True)
    observed = resolved.stat()
    if (
        not stat.S_ISDIR(observed.st_mode)
        or observed.st_uid != os.geteuid()
        or stat.S_IMODE(observed.st_mode) != 0o700
    ):
        _fail(f"shadow package directory custody differs: {resolved}")
    return resolved


def _stable_copy(source: Path, destination: Path) -> dict[str, Any]:
    if source.is_symlink() or destination.exists() or destination.is_symlink():
        _fail("shadow package artifact path is unsafe")
    digest = hashlib.sha256()
    try:
        source_fd = os.open(
            source,
            os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
        )
        try:
            before = os.fstat(source_fd)
            if (
                not stat.S_ISREG(before.st_mode)
                or before.st_uid != os.geteuid()
                or not 0 < before.st_size <= _MAX_COPY_BYTES
            ):
                _fail("shadow package source artifact identity differs")
            destination_fd = os.open(
                destination,
                os.O_WRONLY
                | os.O_CREAT
                | os.O_EXCL
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0),
                0o400,
            )
            try:
                remaining = before.st_size
                while remaining:
                    chunk = os.read(source_fd, min(8 * 1024 * 1024, remaining))
                    if not chunk:
                        break
                    digest.update(chunk)
                    view = memoryview(chunk)
                    while view:
                        written = os.write(destination_fd, view)
                        if written <= 0:
                            _fail("shadow package artifact write was short")
                        view = view[written:]
                    remaining -= len(chunk)
                os.fsync(destination_fd)
            finally:
                os.close(destination_fd)
            after = os.fstat(source_fd)
        finally:
            os.close(source_fd)
    except OSError as exc:
        destination.unlink(missing_ok=True)
        raise UnifiedIntrinsicShadowPackageError("shadow package artifact copy failed") from exc
    if (
        remaining
        or (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
        or destination.stat().st_size != before.st_size
    ):
        destination.unlink(missing_ok=True)
        _fail("shadow package source changed while copying")
    destination.chmod(0o400)
    return {
        "path": destination.name,
        "sha256": digest.hexdigest(),
        "size_bytes": before.st_size,
    }


def _write_document(path: Path, value: Mapping[str, Any]) -> dict[str, Any]:
    payload = canonical_bytes(dict(value)) + b"\n"
    atomic_write_bytes(path, payload, mode=0o400)
    return {
        "path": path.name,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size_bytes": len(payload),
    }


def _stable_identity(path: Path, *, maximum: int = _MAX_COPY_BYTES) -> dict[str, Any]:
    if path.is_symlink():
        _fail("shadow package artifact is a symlink")
    digest = hashlib.sha256()
    try:
        descriptor = os.open(
            path,
            os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
        )
        try:
            before = os.fstat(descriptor)
            if (
                not stat.S_ISREG(before.st_mode)
                or before.st_uid != os.geteuid()
                or stat.S_IMODE(before.st_mode) != 0o400
                or not 0 < before.st_size <= maximum
            ):
                _fail("shadow package artifact custody differs")
            remaining = before.st_size
            while remaining:
                chunk = os.read(descriptor, min(8 * 1024 * 1024, remaining))
                if not chunk:
                    break
                digest.update(chunk)
                remaining -= len(chunk)
            after = os.fstat(descriptor)
        finally:
            os.close(descriptor)
    except OSError as exc:
        raise UnifiedIntrinsicShadowPackageError("shadow package artifact is unreadable") from exc
    if remaining or (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    ) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ):
        _fail("shadow package artifact changed while reading")
    return {"path": path.name, "sha256": digest.hexdigest(), "size_bytes": before.st_size}


def _read_document(path: Path) -> dict[str, Any]:
    identity = _stable_identity(path, maximum=1024 * 1024 * 1024)
    try:
        payload = path.read_bytes()
        value = json.loads(payload.decode("ascii"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise UnifiedIntrinsicShadowPackageError("shadow package document is invalid") from exc
    if (
        not isinstance(value, dict)
        or payload != canonical_bytes(value) + b"\n"
        or len(payload) != identity["size_bytes"]
        or hashlib.sha256(payload).hexdigest() != identity["sha256"]
    ):
        _fail("shadow package document is not canonical")
    return value


def _verified_evidence(
    campaign: Path,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    list[dict[str, Any]],
]:
    config, completion = launcher._terminal_campaign(campaign / "campaign.json")  # noqa: SLF001
    arguments = argparse.Namespace(
        campaign=campaign,
        output=None,
        verdict_output=None,
    )
    _campaign, _config, plan = replication._load_plan(arguments)  # noqa: SLF001
    verdict_path = campaign / "resident-replication" / "replication-verdict.json"
    verdict = replication._read_canonical(verdict_path)  # noqa: SLF001
    recomputed = replication.adjudicate(arguments)
    if verdict != recomputed or verdict.get("supported") is not True:
        _fail("shadow package requires a supported recomputed replication verdict")
    reports: list[dict[str, Any]] = []
    for row in plan["evaluations"]:
        status = launcher.status(
            replication._evaluation_arguments(campaign, plan, row)  # noqa: SLF001
        )
        report = status.get("report")
        if status.get("state") != "completed" or not isinstance(report, dict):
            _fail("shadow package replication report is incomplete")
        reports.append(report)
    if len(reports) != len(plan["seeds"]):
        _fail("shadow package replication report inventory differs")
    return config, completion, plan, verdict, reports


_CANARY_GENERATOR_SOURCES: Final = (
    "core/learning/recurrence_curriculum.py",
    "tools/evaluate_unified_intrinsic_checkpoint.py",
    "tools/train_intrinsic_recurrence.py",
)


def _fresh_canary_battery(
    config: Mapping[str, Any],
    identity: Mapping[str, Any],
    plan: Mapping[str, Any],
    verdict: Mapping[str, Any],
    reports: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Generate a tokenizer-bound battery disjoint from all frozen evidence."""

    from mlx_lm.utils import load_tokenizer

    from tools.evaluate_unified_intrinsic_checkpoint import _fresh_tasks
    from tools.train_intrinsic_recurrence import encode_example
    from tools.unified_intrinsic_tokenization_contract import load_source_dataset

    paths = config.get("paths")
    model = identity.get("model")
    if not isinstance(paths, Mapping) or not isinstance(model, Mapping):
        _fail("shadow package canary source identity is unavailable")
    try:
        training, holdout = load_source_dataset(Path(str(paths["dataset"])))
    except (KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
        raise UnifiedIntrinsicShadowPackageError(
            "shadow package frozen dataset is unavailable"
        ) from exc
    excluded_task_ids = {str(task.task_id) for task in (*training, *holdout)}
    excluded_prompt_sha256s = {
        hashlib.sha256(str(task.prompt).encode()).hexdigest()
        for task in (*training, *holdout)
    }
    for report in reports:
        candidates = report.get("candidates")
        if not isinstance(candidates, list) or not candidates:
            _fail("shadow package canary replication candidates are unavailable")
        for candidate in candidates:
            if (
                not isinstance(candidate, Mapping)
                or not isinstance(candidate.get("task_id"), str)
                or not candidate["task_id"]
                or not isinstance(candidate.get("prompt_sha256"), str)
                or len(candidate["prompt_sha256"]) != 64
                or any(
                    character not in "0123456789abcdef"
                    for character in candidate["prompt_sha256"]
                )
            ):
                _fail("shadow package canary replication identity differs")
            excluded_task_ids.add(candidate["task_id"])
            excluded_prompt_sha256s.add(candidate["prompt_sha256"])
    verdict_sha256 = verdict.get("verdict_sha256")
    plan_sha256 = plan.get("plan_sha256")
    if not isinstance(verdict_sha256, str) or not isinstance(plan_sha256, str):
        _fail("shadow package canary evidence commitments are unavailable")
    base_seed = int(
        hashlib.sha256(f"{verdict_sha256}:live-shadow-canary".encode()).hexdigest()[:15],
        16,
    )
    task_depths = tuple(int(value) for value in identity.get("task_depths", ()))
    if not task_depths:
        _fail("shadow package canary task depths are unavailable")
    tasks: list[Any] = []
    selected_seed = -1
    for attempt in range(64):
        selected_seed = base_seed + attempt * 10_000_019
        proposed = [
            task
            for depth in task_depths
            for task in _fresh_tasks(
                dict(identity),
                per_cell=1,
                seed=selected_seed + depth * 1_000_003,
                task_depth=depth,
            )
        ]
        proposed_ids = {str(task.task_id) for task in proposed}
        proposed_prompts = {
            hashlib.sha256(str(task.prompt).encode()).hexdigest()
            for task in proposed
        }
        if (
            len(proposed_ids) == len(proposed)
            and len(proposed_prompts) == len(proposed)
            and not proposed_ids & excluded_task_ids
            and not proposed_prompts & excluded_prompt_sha256s
        ):
            tasks = proposed
            break
    if not tasks:
        _fail("shadow package fresh canary could not be made disjoint")
    model_path = model.get("canonical_path")
    if not isinstance(model_path, str) or not model_path:
        _fail("shadow package canary tokenizer path is unavailable")
    try:
        tokenizer = load_tokenizer(model_path)
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise UnifiedIntrinsicShadowPackageError(
            "shadow package canary tokenizer is unavailable"
        ) from exc
    bridge_value = identity.get("bridge")
    if not isinstance(bridge_value, str) or not bridge_value:
        _fail("shadow package canary answer bridge is unavailable")
    bridge = {"assistant_answer": "\n\nFINAL_ANSWER: "}.get(
        bridge_value,
        bridge_value,
    )
    cases: list[dict[str, Any]] = []
    for task in tasks:
        prompt, answer = encode_example(tokenizer, task, bridge)
        public_token_ids = [int(value) for value in prompt[0].tolist()]
        expected_token_ids = [int(value) for value in answer[0].tolist()]
        if not public_token_ids or not expected_token_ids:
            _fail("shadow package canary tokenization is empty")
        cases.append(
            {
                "task_id": str(task.task_id),
                "family": str(task.family),
                "task_depth": int(task.depth),
                "prompt_sha256": hashlib.sha256(str(task.prompt).encode()).hexdigest(),
                "expected_sha256": hashlib.sha256(str(task.answer).encode()).hexdigest(),
                "public_token_ids": public_token_ids,
                "expected_token_ids": expected_token_ids,
                "max_tokens": len(expected_token_ids),
            }
        )
    source_sha256s = {
        relative: hashlib.sha256((REPO_ROOT / relative).read_bytes()).hexdigest()
        for relative in _CANARY_GENERATOR_SOURCES
    }
    return seal_shadow_canary_battery(
        cases,
        seed=selected_seed,
        replication_plan_sha256=plan_sha256,
        replication_verdict_sha256=verdict_sha256,
        excluded_task_ids_sha256=canonical_sha256(sorted(excluded_task_ids)),
        excluded_prompt_sha256s_sha256=canonical_sha256(
            sorted(excluded_prompt_sha256s)
        ),
        generator_source_sha256s=source_sha256s,
    )


def inspect_shadow_package(
    package: Path,
    *,
    expected_package_id: str | None = None,
) -> dict[str, Any]:
    """Reopen every byte and prove a package remains shadow-only and complete."""

    package = _private_directory(package, create=False)
    manifest = _read_document(package / "manifest.json")
    complete = _read_document(package / "PACKAGE_COMPLETE.json")
    manifest_body = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    complete_body = {key: value for key, value in complete.items() if key != "complete_sha256"}
    package_id = manifest.get("package_id")
    domain_contract = manifest.get("domain_contract")
    claims_not_supported = manifest.get("claims_not_supported")
    if (
        manifest.get("schema") != PACKAGE_SCHEMA
        or not isinstance(package_id, str)
        or _PACKAGE_ID.fullmatch(package_id) is None
        or (expected_package_id is not None and package_id != expected_package_id)
        or manifest.get("manifest_sha256") != canonical_sha256(manifest_body)
        or manifest.get("mode") != "shadow_only"
        or manifest.get("serving_authority") is not False
        or not isinstance(domain_contract, dict)
        or domain_contract.get("ordinary_chat_authorized") is not False
        or domain_contract.get("arbitrary_reasoning_authorized") is not False
        or not isinstance(claims_not_supported, list)
        or claims_not_supported.count("global_activation") != 1
        or claims_not_supported.count("static_weight_fusion") != 1
    ):
        _fail("shadow package manifest authority or identity differs")
    manifest_identity = _stable_identity(package / "manifest.json")
    if (
        complete.get("schema") != COMPLETE_SCHEMA
        or complete.get("package_id") != package_id
        or complete.get("manifest_sha256") != manifest["manifest_sha256"]
        or complete.get("manifest_file_sha256") != manifest_identity["sha256"]
        or complete.get("mode") != "shadow_only"
        or complete.get("serving_authority") is not False
        or complete.get("complete_sha256") != canonical_sha256(complete_body)
    ):
        _fail("shadow package completion receipt differs")

    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict):
        _fail("shadow package artifact inventory is unavailable")
    bindings: list[dict[str, Any]] = []
    for role in (
        "controller",
        "checkpoint",
        "campaign_completion",
        "replication_plan",
        "replication_verdict",
        "canary_battery",
    ):
        binding = artifacts.get(role)
        if not isinstance(binding, dict):
            _fail(f"shadow package artifact binding is unavailable: {role}")
        bindings.append(binding)
    reports = artifacts.get("replication_reports")
    if not isinstance(reports, list) or not reports:
        _fail("shadow package replication report inventory is unavailable")
    bindings.extend(reports)

    expected_files = {"manifest.json", "PACKAGE_COMPLETE.json"}
    for binding in bindings:
        if set(binding) != {"path", "sha256", "size_bytes"}:
            _fail("shadow package artifact binding shape differs")
        name = binding.get("path")
        if not isinstance(name, str) or Path(name).name != name or name in expected_files:
            _fail("shadow package artifact binding path is invalid")
        expected_files.add(name)
        observed = _stable_identity(package / name)
        if observed != binding:
            _fail(f"shadow package artifact binding differs: {name}")
    observed_files = {entry.name for entry in package.iterdir()}
    if (
        observed_files != expected_files
        or complete.get("bound_artifact_count") != len(bindings) + 1
    ):
        _fail("shadow package artifact inventory differs")

    verdict = _read_document(package / str(artifacts["replication_verdict"]["path"]))
    checkpoint = _read_document(package / str(artifacts["checkpoint"]["path"]))
    canary_battery = _read_document(package / str(artifacts["canary_battery"]["path"]))
    try:
        validate_shadow_canary_battery(canary_battery)
    except (TypeError, ValueError) as exc:
        raise UnifiedIntrinsicShadowPackageError(
            "shadow package canary battery differs"
        ) from exc
    if (
        verdict.get("supported") is not True
        or verdict.get("verdict_sha256") != manifest.get("replication_verdict_sha256")
        or verdict.get("checkpoint_sha256") != manifest.get("checkpoint_sha256")
        or checkpoint.get("checkpoint_sha256") != manifest.get("checkpoint_sha256")
        or canary_battery.get("battery_sha256")
        != manifest.get("canary_battery_sha256")
        or canary_battery.get("replication_plan_sha256")
        != manifest.get("replication_plan_sha256")
        or canary_battery.get("replication_verdict_sha256")
        != manifest.get("replication_verdict_sha256")
    ):
        _fail("shadow package scientific evidence differs")
    return {
        "schema": COMPLETE_SCHEMA,
        "package": str(package),
        "package_id": package_id,
        "manifest_sha256": manifest["manifest_sha256"],
        "complete_sha256": complete["complete_sha256"],
        "mode": "shadow_only",
        "serving_authority": False,
    }


def materialize(
    campaign: Path,
    *,
    output_root: Path,
    package_id: str,
) -> dict[str, Any]:
    if not isinstance(package_id, str) or _PACKAGE_ID.fullmatch(package_id) is None:
        _fail("shadow package id is invalid")
    campaign = _private_directory(campaign, create=False)
    output_root = _private_directory(output_root, create=True)
    destination = output_root / package_id
    if destination.exists() or destination.is_symlink():
        _fail("shadow package destination already exists")

    config, completion, plan, verdict, reports = _verified_evidence(campaign)
    if verdict.get("supported") is not True:
        _fail("shadow package requires a supported replication verdict")
    checkpoint = resolve_checkpoint_generation(
        Path(config["paths"]["training_output"]),
        stem="checkpoint_answer_bridge_admitted",
        required=True,
    )
    if checkpoint is None:  # pragma: no cover - required=True is exhaustive
        _fail("shadow package admitted checkpoint is unavailable")
    if completion["checkpoint"].get("checkpoint_sha256") != checkpoint.receipt.get(
        "checkpoint_sha256"
    ) or checkpoint.receipt.get("checkpoint_sha256") != verdict.get("checkpoint_sha256"):
        _fail("shadow package checkpoint differs from terminal evidence")

    stage = output_root / f".{package_id}.stage-{uuid.uuid4().hex}"
    stage.mkdir(mode=0o700)
    try:
        artifacts = {
            "controller": _stable_copy(
                checkpoint.weights_path,
                stage / "controller.safetensors",
            ),
            "checkpoint": _write_document(
                stage / "checkpoint-complete.json",
                checkpoint.receipt,
            ),
            "campaign_completion": _write_document(
                stage / "campaign-completion.json",
                completion,
            ),
            "replication_plan": _write_document(
                stage / "replication-plan.json",
                plan,
            ),
            "replication_verdict": _write_document(
                stage / "replication-verdict.json",
                verdict,
            ),
        }
        report_bindings = []
        for index, report in enumerate(reports, start=1):
            report_bindings.append(
                _write_document(
                    stage / f"replication-report-{index:02d}.json",
                    report,
                )
            )

        identity = checkpoint.receipt.get("identity")
        if not isinstance(identity, dict):
            _fail("shadow package checkpoint identity is unavailable")
        canary_battery = _fresh_canary_battery(
            config,
            identity,
            plan,
            verdict,
            reports,
        )
        artifacts["canary_battery"] = _write_document(
            stage / "shadow-canary-battery.json",
            canary_battery,
        )
        body = {
            "schema": PACKAGE_SCHEMA,
            "package_id": package_id,
            "mode": "shadow_only",
            "serving_authority": False,
            "campaign_id": config["campaign_id"],
            "campaign_config_sha256": config["config_sha256"],
            "source_commit": config["source"]["git"]["commit"],
            "checkpoint_sha256": checkpoint.receipt["checkpoint_sha256"],
            "checkpoint_identity_sha256": identity["identity_sha256"],
            "model_manifest_sha256": config["model"]["manifest_sha256"],
            "tokenizer_identity_sha256": identity["tokenizer"]["identity_sha256"],
            "replication_plan_sha256": plan["plan_sha256"],
            "replication_verdict_sha256": verdict["verdict_sha256"],
            "canary_battery_sha256": canary_battery["battery_sha256"],
            "domain_contract": {
                "qualification": "generator_and_grammar_bound",
                "families": list(identity["families"]),
                "task_depths": list(identity["task_depths"]),
                "recurrence_depth": int(plan["recurrence_depths"][0]),
                "answer_emission_contract_sha256": identity["answer_emission_contract"][
                    "contract_sha256"
                ],
                "ordinary_chat_authorized": False,
                "arbitrary_reasoning_authorized": False,
            },
            "artifacts": {**artifacts, "replication_reports": report_bindings},
            "promotion_requirements": [
                "source_bound_worker_loader",
                "shadow_output_equivalence_and_latency_gate",
                "rollback_and_restart_proof",
                "domain_qualified_live_canary",
                "broader_equal_compute_reasoning_replication",
            ],
            "claims_not_supported": [
                "ordinary_chat_gain",
                "broad_reasoning_gain",
                "frontier_performance",
                "global_activation",
                "static_weight_fusion",
                "wow_signal",
            ],
        }
        manifest = {**body, "manifest_sha256": canonical_sha256(body)}
        manifest_binding = _write_document(stage / "manifest.json", manifest)
        complete_body = {
            "schema": COMPLETE_SCHEMA,
            "package_id": package_id,
            "manifest_sha256": manifest["manifest_sha256"],
            "manifest_file_sha256": manifest_binding["sha256"],
            "bound_artifact_count": 7 + len(report_bindings),
            "mode": "shadow_only",
            "serving_authority": False,
        }
        complete = {**complete_body, "complete_sha256": canonical_sha256(complete_body)}
        _write_document(stage / "PACKAGE_COMPLETE.json", complete)
        inspect_shadow_package(stage, expected_package_id=package_id)
        os.replace(stage, destination)
        directory_fd = os.open(output_root, os.O_RDONLY | getattr(os, "O_CLOEXEC", 0))
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    try:
        return inspect_shadow_package(destination, expected_package_id=package_id)
    except Exception:
        shutil.rmtree(destination, ignore_errors=True)
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="action", required=True)
    materialize_parser = subparsers.add_parser("materialize")
    materialize_parser.add_argument("campaign", type=Path)
    materialize_parser.add_argument("--output-root", type=Path, required=True)
    materialize_parser.add_argument("--package-id", required=True)
    inspect_parser = subparsers.add_parser("inspect")
    inspect_parser.add_argument("package", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        if arguments.action == "inspect":
            result = inspect_shadow_package(arguments.package)
        else:
            result = materialize(
                arguments.campaign,
                output_root=arguments.output_root,
                package_id=arguments.package_id,
            )
    except (
        OSError,
        ValueError,
        UnifiedIntrinsicShadowPackageError,
    ) as exc:
        print(f"shadow package materialization failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
