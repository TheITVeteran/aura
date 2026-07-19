#!/usr/bin/env python3
"""Verify one recurrence-native v2 interruption/resume mechanics campaign.

This is a mechanics gate, not an intelligence-gain certificate. It binds the
strict adapter identity, exact training resume state, detached containment,
campaign journal replay, ordinary-generation isolation, and causal recurrent
adapter activation into one deterministic receipt.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Never, cast

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from core.brain.llm.latent_cortex.campaign_journal import (  # noqa: E402
    CampaignJournal,
    CampaignPlan,
    canonical_json_bytes,
)
from core.brain.llm.latent_cortex.paired_campaign import (  # noqa: E402
    ADAPTER_RLC,
    ADAPTER_VANILLA,
    BASE_RLC,
    BASE_VANILLA,
)
from tools import run_detached_step as detached  # noqa: E402
from tools import run_latent_cortex_paired_campaign as campaign_runner  # noqa: E402
from tools.verify_latent_cortex_campaign_resume import _verify_grade  # noqa: E402

SCHEMA = "aura.recurrence_v2_mechanics_verdict.v1"


class SmokeVerificationError(RuntimeError):
    """Stable fail-closed mechanics-verification failure."""


def _fail(reason: str) -> Never:
    raise SmokeVerificationError(reason)


def _strict_json(path: Path, *, max_bytes: int = 64 * 1024 * 1024) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        _fail(f"unsafe_or_missing_artifact:{path}")
    before = path.stat()
    if not 0 < before.st_size <= max_bytes:
        _fail(f"artifact_size_invalid:{path}")
    raw = path.read_bytes()
    after = path.stat()
    if (
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
        _fail(f"artifact_changed_while_reading:{path}")

    def object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                _fail(f"duplicate_json_key:{path}:{key}")
            result[key] = value
        return result

    try:
        value = json.loads(raw, object_pairs_hook=object_pairs)
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as exc:
        raise SmokeVerificationError(f"artifact_json_invalid:{path}") from exc
    if not isinstance(value, dict):
        _fail(f"artifact_not_object:{path}")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_detached_receipt(
    run_dir: Path,
    *,
    expected_returncodes: frozenset[int],
) -> dict[str, Any]:
    receipt = detached._verified_receipt(run_dir / detached.RECEIPT_FILE)
    if (
        receipt.get("returncode") not in expected_returncodes
        or receipt.get("containment_verified") is not True
        or receipt.get("lineage_empty") is not True
        or receipt.get("process_group_empty") is not True
        or receipt.get("timed_out") is not False
        or receipt.get("supervisor_error") is not None
    ):
        _fail(f"detached_receipt_invalid:{run_dir}")
    return cast(dict[str, Any], receipt)


def _activation(value: Any, *, expected_active: bool) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        _fail("recurrence_adapter_activation_missing")
    activation = dict(value)
    if (
        activation.get("schema") != "aura.recurrence_adapter_activation.v1"
        or activation.get("scope") != "latent_slots_only"
        or activation.get("active") is not expected_active
    ):
        _fail("recurrence_adapter_activation_invalid")
    calls = activation.get("calls")
    adapted = activation.get("adapted_positions")
    observed = activation.get("observed_positions")
    if any(type(item) is not int or item < 0 for item in (calls, adapted, observed)):
        _fail("recurrence_adapter_activation_count_invalid")
    calls_value = cast(int, calls)
    adapted_value = cast(int, adapted)
    observed_value = cast(int, observed)
    if expected_active:
        if calls_value <= 0 or adapted_value <= 0 or observed_value < adapted_value:
            _fail("recurrence_adapter_did_not_run")
    elif calls_value != 0 or adapted_value != 0 or observed_value != 0:
        _fail("base_arm_adapter_contaminated")
    return activation


def _verify_training(
    *,
    model: Path,
    adapter: Path,
    adapter_id: str,
    partial_run: Path,
    resume_run: Path,
    expected_steps: int,
    expected_invocations: int,
    expected_generations: int,
) -> dict[str, Any]:
    identity_args = argparse.Namespace(
        model=str(model),
        adapter=str(adapter),
        adapter_id=adapter_id,
        personality_adapter="trained",
    )
    model_identity, adapter_identity = campaign_runner._identity_material(identity_args)
    identity_receipt = adapter_identity.get("identity_receipt")
    if not isinstance(identity_receipt, Mapping) or identity_receipt.get("complete") is not True:
        _fail("strict_adapter_identity_incomplete")

    receipt = _strict_json(adapter / "receipt.json")
    completion = _strict_json(adapter / "training_completion.json")
    latest = _strict_json(adapter / "latest.json")
    if (
        receipt.get("steps") != expected_steps
        or receipt.get("invocation_count") != expected_invocations
        or receipt.get("halt_reason") != "max_steps"
        or receipt.get("complete") is not True
        or completion.get("complete") is not True
        or completion.get("step") != expected_steps
        or completion.get("halt_reason") != "max_steps"
    ):
        _fail("training_completion_invalid")
    checkpoint_root = adapter / "checkpoints"
    if checkpoint_root.is_symlink() or not checkpoint_root.is_dir():
        _fail("checkpoint_root_invalid")
    generations = sorted(path.name for path in checkpoint_root.iterdir() if path.is_dir())
    if len(generations) != expected_generations:
        _fail("checkpoint_generation_count_mismatch")
    final_checkpoint = receipt.get("final_checkpoint")
    if (
        not isinstance(final_checkpoint, str)
        or final_checkpoint != generations[-1]
        or latest.get("checkpoint") != f"checkpoints/{final_checkpoint}"
    ):
        _fail("final_checkpoint_binding_invalid")
    partial = _verify_detached_receipt(partial_run, expected_returncodes=frozenset({75}))
    resumed = _verify_detached_receipt(resume_run, expected_returncodes=frozenset({0}))
    if resumed.get("passed") is not True or partial.get("passed") is not False:
        _fail("detached_training_status_invalid")
    return {
        "adapter_identity_sha256": identity_receipt["composite_identity_sha256"],
        "adapter_sha256": identity_receipt["adapter_sha256"],
        "base_checkpoint_sha256": model_identity["fingerprint"],
        "checkpoint_generations": generations,
        "completion_sha256": _sha256_file(adapter / "training_completion.json"),
        "final_checkpoint": final_checkpoint,
        "invocation_count": receipt["invocation_count"],
        "loss_trail": receipt.get("loss_trail"),
        "partial_detached_receipt_sha256": partial["receipt_sha256"],
        "resume_detached_receipt_sha256": resumed["receipt_sha256"],
        "steps": receipt["steps"],
    }


def _verify_campaign(campaign_dir: Path) -> dict[str, Any]:
    plan = CampaignPlan.from_dict(_strict_json(campaign_dir / "plan.json"))
    with CampaignJournal(campaign_dir / "campaign.jsonl", plan) as journal:
        snapshot = journal.resume()
        records = journal.committed_records()
        manifest = journal.finalize(campaign_dir / "campaign_manifest.json")
    _verify_grade(campaign_dir / "grade.json", manifest["manifest_sha256"])
    if snapshot.runnable_cell_ids or snapshot.incomplete_cell_ids:
        _fail("campaign_not_complete")
    campaign_receipt = _verify_detached_receipt(
        campaign_dir,
        expected_returncodes=frozenset({0, 2}),
    )

    rows: dict[str, dict[str, dict[str, Any]]] = {}
    for record in records:
        definition = record.get("definition")
        result = record.get("result")
        if not isinstance(definition, dict) or not isinstance(result, dict):
            _fail("campaign_record_invalid")
        task_id = definition.get("task_id")
        arm = definition.get("arm")
        if not isinstance(task_id, str) or not isinstance(arm, str):
            _fail("campaign_record_identity_invalid")
        rows.setdefault(task_id, {})[arm] = record
    expected_arms = {BASE_VANILLA, BASE_RLC, ADAPTER_VANILLA, ADAPTER_RLC}
    if not rows or any(set(arms) != expected_arms for arms in rows.values()):
        _fail("campaign_arm_matrix_incomplete")

    causal_digest_changes = 0
    activation_totals = {"calls": 0, "adapted_positions": 0, "observed_positions": 0}
    ordinary_hashes: list[str] = []
    for arms in rows.values():
        base_v = arms[BASE_VANILLA]["result"]
        adapter_v = arms[ADAPTER_VANILLA]["result"]
        base_r = arms[BASE_RLC]["result"]
        adapter_r = arms[ADAPTER_RLC]["result"]
        if (
            base_v.get("text") != adapter_v.get("text")
            or base_v.get("output_sha256") != adapter_v.get("output_sha256")
            or base_v.get("episode_receipt")
            or adapter_v.get("episode_receipt")
        ):
            _fail("ordinary_generation_isolation_failed")
        ordinary_hashes.append(str(base_v["output_sha256"]))
        base_episode = base_r.get("episode_receipt")
        adapter_episode = adapter_r.get("episode_receipt")
        if not isinstance(base_episode, Mapping) or not isinstance(adapter_episode, Mapping):
            _fail("rlc_episode_receipt_missing")
        _activation(base_episode.get("recurrence_adapter"), expected_active=False)
        adapter_activation = _activation(
            adapter_episode.get("recurrence_adapter"), expected_active=True
        )
        for key in activation_totals:
            activation_totals[key] += int(adapter_activation[key])
        if base_episode.get("first_logits_digest") != adapter_episode.get(
            "first_logits_digest"
        ):
            causal_digest_changes += 1
        if (
            base_r.get("adapter_wrapped_projections") != 0
            or base_r.get("runtime_adapter_identity") is not None
            or type(adapter_r.get("adapter_wrapped_projections")) is not int
            or adapter_r["adapter_wrapped_projections"] <= 0
            or not isinstance(adapter_r.get("runtime_adapter_identity"), Mapping)
        ):
            _fail("runtime_adapter_identity_invalid")
    if causal_digest_changes <= 0:
        _fail("adapter_has_no_observed_causal_logit_effect")
    grade = _strict_json(campaign_dir / "grade.json")
    return {
        "activation_totals": activation_totals,
        "campaign_detached_receipt_sha256": campaign_receipt["receipt_sha256"],
        "campaign_manifest_sha256": manifest["manifest_sha256"],
        "causal_logit_digest_changes": causal_digest_changes,
        "committed_cells": len(records),
        "grade_sha256": grade.get("grade_sha256"),
        "grade_verdict": grade.get("verdict"),
        "ordinary_generation_exact_match": True,
        "ordinary_output_sha256": sorted(ordinary_hashes),
        "plan_sha256": plan.plan_sha256,
        "task_count": len(rows),
    }


def _atomic_create_or_verify(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.is_symlink() or path.read_bytes() != payload:
            _fail("existing_verdict_differs")
        return
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    descriptor = os.open(
        temporary,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0),
        0o600,
    )
    try:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                _fail("short_verdict_write")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    try:
        os.link(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--adapter-id", required=True)
    parser.add_argument("--partial-run", required=True)
    parser.add_argument("--resume-run", required=True)
    parser.add_argument("--campaign-dir", required=True)
    parser.add_argument("--expected-steps", type=int, default=2)
    parser.add_argument("--expected-invocations", type=int, default=2)
    parser.add_argument("--expected-generations", type=int, default=2)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    try:
        training = _verify_training(
            model=Path(args.model).expanduser().resolve(strict=True),
            adapter=Path(args.adapter).expanduser().resolve(strict=True),
            adapter_id=args.adapter_id,
            partial_run=Path(args.partial_run).expanduser().resolve(strict=True),
            resume_run=Path(args.resume_run).expanduser().resolve(strict=True),
            expected_steps=args.expected_steps,
            expected_invocations=args.expected_invocations,
            expected_generations=args.expected_generations,
        )
        campaign = _verify_campaign(
            Path(args.campaign_dir).expanduser().resolve(strict=True)
        )
        material = {
            "schema": SCHEMA,
            "claim_scope": "recurrence_v2_mechanics_only",
            "frontier_or_reasoning_gain_proven": False,
            "passed": True,
            "training": training,
            "campaign": campaign,
        }
        verdict = {
            **material,
            "verdict_sha256": hashlib.sha256(canonical_json_bytes(material)).hexdigest(),
        }
        payload = canonical_json_bytes(verdict) + b"\n"
        output = Path(args.output).expanduser().resolve(strict=False)
        _atomic_create_or_verify(output, payload)
    except Exception as exc:
        print(
            f"verify_recurrence_v2_smoke: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 2
    print(json.dumps(verdict, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
