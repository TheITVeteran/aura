#!/usr/bin/env python3
"""Run the arms that can refute the commitment ratchet.

A mechanism whose only runnable arm is its own is not being tested. This is
the runner that makes the falsification executable rather than aspirational:

    python tools/run_commitment_ablation.py --tasks tasks.jsonl --draws 8

Each task is one JSON object per line:

    {"objective": "...", "answer": "42", "pool": ["...", "..."]}

``pool`` is optional; when present it is the candidate set constraints are
measured against, so the narrowing on every receipt is measured rather than
asserted.

The solver is pluggable. Without ``--solver`` it uses a deterministic offline
stub so the harness itself can be exercised in CI without a model — and the
report SAYS it was a stub, because a green run against a stub is evidence
about the harness, not about the mechanism.

Exit codes: 0 SUPPORTED, 1 REFUTED, 2 INCONCLUSIVE. A missing arm is always
INCONCLUSIVE; there is no combination of absent comparisons that returns
SUPPORTED.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.brain.llm.latent_cortex.commitment_ablations import (  # noqa: E402
    Arm,
    adjudicate,
    run_arm,
)
from core.brain.llm.latent_cortex.commitment_extraction import (  # noqa: E402
    propose_constraints,
)


def load_tasks(path: Path) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    for line_no, line in enumerate(path.read_text("utf-8").splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            row = json.loads(line)
        except ValueError as exc:
            raise SystemExit(f"{path}:{line_no}: {exc}") from exc
        if not isinstance(row, dict) or not row.get("objective"):
            raise SystemExit(f"{path}:{line_no}: task needs an 'objective'")
        pool = [str(item) for item in (row.get("pool") or [])]
        row["pool"] = pool
        row["constraints"] = propose_constraints(
            objective=str(row.get("objective") or ""),
            candidates=pool,
            refuted=[str(item) for item in (row.get("refuted") or [])],
        )
        tasks.append(row)
    return tasks


def _stub_solver(tasks: list[dict[str, Any]]):
    """Offline stand-in: honours exclusions, otherwise returns its pool in order.

    Deliberately simple and deliberately labelled. It exists so the harness
    is exercisable without a model; it is not a model, and a result produced
    with it says nothing about the mechanism.
    """
    by_objective = {str(task["objective"]): list(task.get("pool") or []) for task in tasks}
    cursor: dict[str, int] = {}

    def _solve(objective: str, conditioning: str) -> str:
        pool = by_objective.get(objective) or []
        if not pool:
            return ""
        excluded = {
            line.split("NOT '", 1)[1].split("'", 1)[0]
            for line in conditioning.splitlines()
            if "NOT '" in line
        }
        index = cursor.get(objective, 0)
        for candidate in pool[index:] + pool[:index]:
            if candidate not in excluded:
                cursor[objective] = (pool.index(candidate) + 1) % len(pool)
                return candidate
        return pool[0]

    return _solve


def _load_solver(spec: str):
    """Import ``module:function`` and use it as ``solve(objective, block)``."""
    module_name, _, attribute = spec.partition(":")
    if not module_name or not attribute:
        raise SystemExit("--solver must be module:function")
    import importlib

    module = importlib.import_module(module_name)
    solver = getattr(module, attribute, None)
    if not callable(solver):
        raise SystemExit(f"{spec} is not callable")
    return solver


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", required=True, type=Path)
    parser.add_argument("--draws", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--solver",
        default="",
        help="module:function taking (objective, conditioning_block) -> answer",
    )
    parser.add_argument(
        "--arms",
        default="vanilla,ratchet,depth_only,shuffle,random",
        help="comma-separated; vanilla, ratchet, depth_only and shuffle are "
        "required for any verdict other than INCONCLUSIVE",
    )
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    tasks = load_tasks(args.tasks)
    if not tasks:
        raise SystemExit("no tasks loaded")

    stubbed = not args.solver
    solve = _load_solver(args.solver) if args.solver else _stub_solver(tasks)

    results = {}
    for name in [part.strip() for part in args.arms.split(",") if part.strip()]:
        try:
            arm = Arm(name)
        except ValueError:
            raise SystemExit(f"unknown arm {name!r}") from None
        results[arm] = run_arm(arm, tasks, solve=solve, seed=args.seed)

    verdict = adjudicate(results)
    verdict["tasks"] = len(tasks)
    verdict["draws"] = args.draws
    verdict["solver"] = args.solver or "offline_stub"
    # Said loudly. A green run against the stub is evidence about this
    # harness and nothing else, and that must not be discoverable only by
    # reading the invocation.
    verdict["solver_is_stub"] = stubbed
    if stubbed:
        verdict["reason"] = f"{verdict['reason']} (OFFLINE STUB — not a model result)"

    payload = json.dumps(verdict, indent=2, sort_keys=True)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(payload + "\n", encoding="utf-8")
    print(payload)

    return {"SUPPORTED": 0, "REFUTED": 1}.get(verdict["verdict"], 2)


if __name__ == "__main__":
    raise SystemExit(main())
