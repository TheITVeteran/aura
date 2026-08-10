#!/usr/bin/env python
"""Prove the public-objective producer before paying for model inference.

This gate gives the producer only the candidate-visible objective, then asks
the separately implemented hidden scorer whether the resulting answer is
correct. It also requires the public verifier to accept every produced answer
and reject a malformed terminal answer for every task. The artifact contains
commitments and verdicts, never private expected payloads.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import secrets
import sys
import time
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from core.brain.llm.latent_cortex import frontier_tasks as ft  # noqa: E402
from core.brain.llm.latent_cortex.objective_program_verifier import (  # noqa: E402
    solve_objective_program,
    validate_objective_program_solution,
    verify_objective_program,
)
from core.runtime.atomic_writer import atomic_write_text  # noqa: E402

GATE_SCHEMA = "aura.rlc.objective_program_producer_gate.v1"
DEFAULT_DIFFICULTIES = (1, 2, 3)
DEFAULT_SEED_COUNT = 2
EXPECTED_FAMILIES = frozenset(
    {
        "stable_nearest_traversal",
        "separated_subset_count",
        "stateful_python_trace",
        "interventional_chain_inference",
        "dependency_deadline_portfolio",
        "bayesian_frequency_update",
        "premise_audit_table",
    }
)


def _sha_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha_json(value: Any) -> str:
    return _sha_bytes(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    )


def _source_bindings() -> list[dict[str, Any]]:
    paths = (
        Path(__file__).resolve(),
        Path(inspect.getsourcefile(ft) or "").resolve(),
        Path(inspect.getsourcefile(solve_objective_program) or "").resolve(),
    )
    rows: list[dict[str, Any]] = []
    for path in paths:
        data = path.read_bytes()
        rows.append(
            {
                "path": str(path.relative_to(REPO_ROOT)),
                "sha256": _sha_bytes(data),
                "bytes": len(data),
            }
        )
    return rows


def run_gate(
    *,
    seeds: Sequence[int],
    difficulties: Sequence[int] = DEFAULT_DIFFICULTIES,
) -> dict[str, Any]:
    seed_values = tuple(int(seed) for seed in seeds)
    difficulty_values = tuple(int(value) for value in difficulties)
    if not seed_values or len(set(seed_values)) != len(seed_values):
        raise ValueError("producer gate seeds must be non-empty and unique")
    if not difficulty_values or any(value not in {1, 2, 3} for value in difficulty_values):
        raise ValueError("producer gate difficulties must be selected from 1, 2, 3")

    tasks = tuple(
        task
        for difficulty in difficulty_values
        for task in ft.generate_task_battery(
            seed_values,
            difficulty=difficulty,
            registry_version=ft.CONTAMINATION_SAFE_REGISTRY_VERSION,
        )
    )
    manifest = ft.build_task_manifest(tasks)
    commitment = ft.build_task_commitment(manifest)
    rows: list[dict[str, Any]] = []
    family_counts: Counter[str] = Counter()

    for task in tasks:
        # This is the trust boundary: only the public prompt crosses into the
        # producer. The private task remains in the scorer process scope.
        solved = solve_objective_program(task.public.prompt)
        if solved is None:
            rows.append(
                {
                    "task_id": task.task_id,
                    "domain": task.domain,
                    "difficulty": task.public.difficulty,
                    "solved": False,
                }
            )
            continue
        candidate, solution_receipt = solved
        validate_objective_program_solution(
            solution_receipt,
            objective=task.public.prompt,
            candidate=candidate,
        )
        public_verdict = verify_objective_program(
            candidate,
            objective=task.public.prompt,
        )
        malformed_verdict = verify_objective_program(
            "FINAL_ANSWER: {}",
            objective=task.public.prompt,
        )
        hidden_score = ft.score_task(task, candidate)
        family = str(solution_receipt["family"])
        family_counts[family] += 1
        rows.append(
            {
                "task_id": task.task_id,
                "domain": task.domain,
                "difficulty": task.public.difficulty,
                "family": family,
                "solved": True,
                "hidden_scorer_correct": hidden_score.correct,
                "hidden_scorer_receipt": hidden_score.to_dict(),
                "public_verifier_outcome": (
                    public_verdict.get("outcome") if public_verdict else "unavailable"
                ),
                "public_verifier_receipt_sha256": (
                    public_verdict.get("receipt_sha256") if public_verdict else ""
                ),
                "malformed_control_outcome": (
                    malformed_verdict.get("outcome") if malformed_verdict else "unavailable"
                ),
                "solution_receipt_sha256": solution_receipt["receipt_sha256"],
                "candidate_sha256": _sha_bytes(candidate.encode("utf-8")),
            }
        )

    solved_all = all(row.get("solved") is True for row in rows)
    hidden_scorer_all_correct = solved_all and all(
        row.get("hidden_scorer_correct") is True for row in rows
    )
    public_verifier_all_verified = solved_all and all(
        row.get("public_verifier_outcome") == "verified" for row in rows
    )
    malformed_controls_all_refuted = solved_all and all(
        row.get("malformed_control_outcome") == "refuted" for row in rows
    )
    family_coverage_complete = set(family_counts) == EXPECTED_FAMILIES
    admitted = all(
        (
            solved_all,
            hidden_scorer_all_correct,
            public_verifier_all_verified,
            malformed_controls_all_refuted,
            family_coverage_complete,
        )
    )
    body = {
        "schema": GATE_SCHEMA,
        "created_unix": time.time(),
        "registry_version": ft.CONTAMINATION_SAFE_REGISTRY_VERSION,
        "seeds": list(seed_values),
        "difficulties": list(difficulty_values),
        "task_count": len(tasks),
        "task_manifest_sha256": manifest.manifest_sha256,
        "task_commitment": commitment.to_dict(),
        "producer_interface": {
            "callable": "solve_objective_program",
            "signature": str(inspect.signature(solve_objective_program)),
            "candidate_visible_inputs": ["objective"],
            "private_answer_input": False,
        },
        "source_bindings": _source_bindings(),
        "family_counts": dict(sorted(family_counts.items())),
        "checks": {
            "all_tasks_solved": solved_all,
            "hidden_scorer_all_correct": hidden_scorer_all_correct,
            "public_verifier_all_verified": public_verifier_all_verified,
            "malformed_controls_all_refuted": malformed_controls_all_refuted,
            "family_coverage_complete": family_coverage_complete,
        },
        "admitted": admitted,
        "rows": rows,
    }
    return {**body, "receipt_sha256": _sha_json(body)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, action="append", default=[])
    parser.add_argument("--seed-count", type=int, default=DEFAULT_SEED_COUNT)
    parser.add_argument(
        "--difficulty",
        type=int,
        action="append",
        choices=DEFAULT_DIFFICULTIES,
        default=[],
    )
    args = parser.parse_args()
    if args.seed_count <= 0:
        parser.error("--seed-count must be positive")
    seeds = tuple(args.seed) or tuple(
        secrets.randbelow(2**31 - 1) + 1 for _ in range(args.seed_count)
    )
    if len(set(seeds)) != len(seeds):
        parser.error("seeds must be unique")
    report = run_gate(
        seeds=seeds,
        difficulties=tuple(args.difficulty) or DEFAULT_DIFFICULTIES,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(
        args.out,
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "admitted": report["admitted"],
                "task_count": report["task_count"],
                "family_counts": report["family_counts"],
                "receipt_sha256": report["receipt_sha256"],
                "artifact": str(args.out.resolve()),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0 if report["admitted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
