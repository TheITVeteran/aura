#!/usr/bin/env python3
"""Independently verify a sealed semantic neural decode canary."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Final

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.brain.llm.latent_cortex.semantic_neural_decode_context import (  # noqa: E402
    execute_semantic_neural_decode_state,
)
from core.learning.frontier_process_supervision import (  # noqa: E402
    frontier_process_task_battery,
)
from core.runtime.atomic_writer import atomic_write_text  # noqa: E402

CANARY_SCHEMA: Final = "aura.rlc.semantic_neural_decode_canary.v1"
JOURNAL_SCHEMA: Final = "aura.rlc.semantic_neural_decode_journal.v1"
VERIFICATION_SCHEMA: Final = "aura.rlc.semantic_neural_decode_verification.v1"
ARMS: Final = (
    "ordinary_base",
    "matched_wire_base",
    "treatment",
    "coefficient_lesion",
    "matched_wrong_state",
)
DOMAINS: Final = ("coding", "calibration", "misleading_premise")
DIFFICULTIES: Final = (1, 2, 3)
SOURCE_PATHS: Final = (
    "core/brain/llm/latent_cortex/semantic_neural_decode_context.py",
    "core/brain/llm/unified_recurrent_transfer_decode.py",
    "core/learning/frontier_process_supervision.py",
    "core/learning/public_frontier_action_compiler.py",
    "core/learning/semantic_neural_machine.py",
    "tools/run_semantic_neural_decode_canary.py",
)


def _sha(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest()


def _file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_blob_sha(commit: str, path: str) -> str:
    completed = subprocess.run(
        ("git", "show", f"{commit}:{path}"),
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
    )
    return hashlib.sha256(completed.stdout).hexdigest()


def _verify_embedded_receipt(payload: dict[str, Any], field: str) -> None:
    claimed = payload.get(field)
    body = {key: value for key, value in payload.items() if key != field}
    if not isinstance(claimed, str) or claimed != _sha(body):
        raise RuntimeError(f"semantic decode {field} mismatch")


def _expected_tasks(seed: int, per_cell: int):
    return frontier_process_task_battery(
        DOMAINS,
        DIFFICULTIES,
        per_cell,
        seed=seed,
    )


def _paired_one_sided_p(gains: int, regressions: int) -> float:
    discordant = gains + regressions
    if discordant == 0:
        return 1.0
    return sum(math.comb(discordant, value) for value in range(gains, discordant + 1)) / (
        2**discordant
    )


def _verify_journal(
    journal_path: Path,
    *,
    payload: dict[str, Any],
    raw_outputs: list[dict[str, Any]],
) -> dict[str, Any]:
    journal_path = journal_path.expanduser().resolve(strict=True)
    events: list[dict[str, Any]] = []
    previous = "0" * 64
    with journal_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                event = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"semantic decode journal line {line_number} is invalid JSON"
                ) from exc
            if not isinstance(event, dict) or event.get("schema") != JOURNAL_SCHEMA:
                raise RuntimeError(f"semantic decode journal line {line_number} is invalid")
            receipt = event.get("receipt_sha256")
            body = {key: value for key, value in event.items() if key != "receipt_sha256"}
            if event.get("previous_receipt_sha256") != previous or receipt != _sha(body):
                raise RuntimeError(
                    f"semantic decode journal receipt chain broke at line {line_number}"
                )
            previous = receipt
            events.append(event)

    expected_count = len(raw_outputs)
    if len(events) != expected_count + 2:
        raise RuntimeError("semantic decode journal event count mismatch")
    started, *decode_events, completed = events
    expected_start = {
        "event": "campaign_started",
        "source_commit": payload["source_commit"],
        "seed": payload["seed"],
        "tasks_per_difficulty": payload["tasks_per_difficulty"],
        "task_count": payload["task_count"],
        "arm_count": len(ARMS),
    }
    if any(started.get(key) != value for key, value in expected_start.items()):
        raise RuntimeError("semantic decode journal campaign identity mismatch")

    for index, (event, raw_output) in enumerate(
        zip(decode_events, raw_outputs, strict=True), start=1
    ):
        row = event.get("row")
        if (
            event.get("event") != "decode_committed"
            or event.get("completed") != index
            or event.get("total") != expected_count
            or event.get("raw_output") != raw_output
            or not isinstance(row, dict)
            or row.get("task_id") != raw_output.get("task_id")
            or row.get("arm") != raw_output.get("arm")
            or row.get("response_sha256")
            != hashlib.sha256(str(raw_output.get("response", "")).encode()).hexdigest()
        ):
            raise RuntimeError(f"semantic decode journal row {index} mismatch")
    last_decode_receipt = decode_events[-1]["receipt_sha256"]
    if payload.get("journal_last_decode_receipt_sha256") != last_decode_receipt:
        raise RuntimeError("semantic decode final journal decode receipt mismatch")
    if (
        completed.get("event") != "campaign_completed"
        or completed.get("previous_receipt_sha256") != last_decode_receipt
        or completed.get("admitted") is not payload.get("admitted")
        or completed.get("report_receipt_sha256") != payload.get("receipt_sha256")
    ):
        raise RuntimeError("semantic decode journal completion mismatch")
    return {
        "journal_sha256": _file_sha(journal_path),
        "journal_event_count": len(events),
        "journal_decode_count": len(decode_events),
        "journal_final_receipt_sha256": completed["receipt_sha256"],
    }


def verify_canary(
    artifact_path: Path,
    *,
    model_path: Path,
    journal_path: Path | None = None,
) -> dict[str, Any]:
    artifact_path = artifact_path.expanduser().resolve(strict=True)
    model_path = model_path.expanduser().resolve(strict=True)
    raw_bytes = artifact_path.read_bytes()
    payload = json.loads(raw_bytes)
    if not isinstance(payload, dict) or payload.get("schema") != CANARY_SCHEMA:
        raise RuntimeError("semantic decode canary schema mismatch")
    _verify_embedded_receipt(payload, "receipt_sha256")

    source_commit = payload.get("source_commit")
    source_sha256s = payload.get("source_sha256s")
    if not isinstance(source_commit, str) or not isinstance(source_sha256s, dict):
        raise RuntimeError("semantic decode source identity is incomplete")
    if set(source_sha256s) != set(SOURCE_PATHS):
        raise RuntimeError("semantic decode source manifest mismatch")
    for path in SOURCE_PATHS:
        if source_sha256s[path] != _git_blob_sha(source_commit, path):
            raise RuntimeError(f"semantic decode source hash mismatch: {path}")

    model_identity = payload.get("model_identity")
    if not isinstance(model_identity, dict):
        raise RuntimeError("semantic decode model identity is missing")
    if Path(str(model_identity.get("path", ""))).expanduser().resolve() != model_path:
        raise RuntimeError("semantic decode model path mismatch")
    for filename, field in (
        ("config.json", "config_sha256"),
        ("model.safetensors.index.json", "weights_index_sha256"),
    ):
        if model_identity.get(field) != _file_sha(model_path / filename):
            raise RuntimeError(f"semantic decode model identity mismatch: {field}")

    seed = payload.get("seed")
    per_cell = payload.get("tasks_per_difficulty")
    if type(seed) is not int or type(per_cell) is not int or not 2 <= per_cell <= 20:
        raise RuntimeError("semantic decode task identity is invalid")
    tasks = _expected_tasks(seed, per_cell)
    task_by_id = {task.task_id: task for task in tasks}
    if len(task_by_id) != len(tasks) or payload.get("task_count") != len(tasks):
        raise RuntimeError("semantic decode task cohort is invalid")

    raw_outputs = payload.get("raw_outputs")
    if not isinstance(raw_outputs, list):
        raise RuntimeError("semantic decode raw outputs are missing")
    expected_pairs = {(task.task_id, arm) for task in tasks for arm in ARMS}
    observed_pairs: set[tuple[str, str]] = set()
    correctness: dict[str, dict[str, bool]] = {arm: {} for arm in ARMS}
    parsed_counts: Counter[str] = Counter()
    for row in raw_outputs:
        if not isinstance(row, dict):
            raise RuntimeError("semantic decode raw output row is invalid")
        task_id, arm, response = row.get("task_id"), row.get("arm"), row.get("response")
        pair = (task_id, arm)
        if pair not in expected_pairs or pair in observed_pairs or not isinstance(response, str):
            raise RuntimeError("semantic decode raw output matrix is invalid")
        observed_pairs.add(pair)
        verdict = task_by_id[task_id].grade(response)
        if not isinstance(verdict, dict) or type(verdict.get("correct")) is not bool:
            raise RuntimeError("semantic decode independent grader returned invalid output")
        correctness[arm][task_id] = bool(verdict["correct"])
        parsed_counts[arm] += int(verdict.get("parsed") is not None)
    if observed_pairs != expected_pairs:
        raise RuntimeError("semantic decode raw output matrix is incomplete")

    summaries = payload.get("arms")
    if not isinstance(summaries, dict) or set(summaries) != set(ARMS):
        raise RuntimeError("semantic decode arm summary manifest mismatch")
    exact_counts: dict[str, int] = {}
    for arm in ARMS:
        summary = summaries[arm]
        if not isinstance(summary, dict):
            raise RuntimeError(f"semantic decode summary is invalid: {arm}")
        _verify_embedded_receipt(summary, "receipt_sha256")
        exact = sum(correctness[arm].values())
        exact_counts[arm] = exact
        expected = {
            "examples": len(tasks),
            "exact": exact,
            "parsed": parsed_counts[arm],
            "exact_accuracy": round(exact / len(tasks), 6),
            "parsed_accuracy": round(parsed_counts[arm] / len(tasks), 6),
        }
        if any(summary.get(key) != value for key, value in expected.items()):
            raise RuntimeError(f"semantic decode summary disagrees with raw output: {arm}")

    ordinary = correctness["ordinary_base"]
    treatment = correctness["treatment"]
    gains = sorted(task_id for task_id in task_by_id if treatment[task_id] and not ordinary[task_id])
    regressions = sorted(
        task_id for task_id in task_by_id if ordinary[task_id] and not treatment[task_id]
    )
    if payload.get("gain_count") != len(gains) or payload.get("gain_set_sha256") != _sha(gains):
        raise RuntimeError("semantic decode gain set mismatch")
    if payload.get("regression_count") != len(regressions) or payload.get("regression_set_sha256") != _sha(regressions):
        raise RuntimeError("semantic decode regression set mismatch")

    replayed_state_receipts = [
        execute_semantic_neural_decode_state(task.prompt, task.family).receipt()[
            "receipt_sha256"
        ]
        for task in tasks
    ]
    if payload.get("treatment_state_receipt_sha256s") != replayed_state_receipts:
        raise RuntimeError("semantic decode treatment-state replay mismatch")

    admitted = bool(
        exact_counts["treatment"] == len(tasks)
        and gains
        and not regressions
        and all(
            exact_counts[arm] < exact_counts["treatment"]
            for arm in ("matched_wire_base", "coefficient_lesion", "matched_wrong_state")
        )
    )
    if payload.get("admitted") is not admitted or not admitted:
        raise RuntimeError("semantic decode independently derived admission failed")

    journal_verification = (
        _verify_journal(journal_path, payload=payload, raw_outputs=raw_outputs)
        if journal_path is not None
        else {}
    )
    paired_p = _paired_one_sided_p(len(gains), len(regressions))

    body = {
        "schema": VERIFICATION_SCHEMA,
        "verified": True,
        "artifact_sha256": hashlib.sha256(raw_bytes).hexdigest(),
        "artifact_receipt_sha256": payload["receipt_sha256"],
        "source_commit": source_commit,
        "model_identity": model_identity,
        "seed": seed,
        "tasks_per_difficulty": per_cell,
        "task_count": len(tasks),
        "raw_output_count": len(raw_outputs),
        "independent_exact_by_arm": exact_counts,
        "independent_parsed_by_arm": dict(parsed_counts),
        "gain_count": len(gains),
        "regression_count": len(regressions),
        "paired_discordant_count": len(gains) + len(regressions),
        "paired_one_sided_exact_p": paired_p,
        "treatment_state_replay_count": len(replayed_state_receipts),
        **journal_verification,
        "claim_boundary": payload.get("claim_boundary"),
        "verifier_source_sha256": _file_sha(Path(__file__)),
    }
    return {**body, "verification_receipt_sha256": _sha(body)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--journal", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    report = verify_canary(
        args.artifact,
        model_path=args.model,
        journal_path=args.journal,
    )
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.report is not None:
        destination = args.report.expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_text(destination, encoded)
        with destination.open("rb") as handle:
            os.fsync(handle.fileno())
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
