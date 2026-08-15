#!/usr/bin/env python3
"""Audit whether Aura's public recurrent inputs identify each verified transition."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.brain.llm.latent_cortex.frontier_tasks import (  # noqa: E402
    CONTAMINATION_SAFE_REGISTRY_VERSION,
)
from core.learning.frontier_process_supervision import (  # noqa: E402
    frontier_process_task_battery,
)
from core.learning.transition_identifiability import (  # noqa: E402
    TRANSITION_IDENTIFIABILITY_SCHEMA,
    audit_public_transition_identifiability,
)
from core.runtime.atomic_writer import atomic_write_json  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--domains",
        default="mathematics,coding,calibration,misleading_premise",
    )
    parser.add_argument("--difficulties", default="1")
    parser.add_argument("--train-per-cell", type=int, default=128)
    parser.add_argument("--holdout-per-cell", type=int, default=6)
    parser.add_argument("--seed", type=int, default=2026081502)
    parser.add_argument("--holdout-seed-offset", type=int, default=9973)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    domains = tuple(value.strip() for value in args.domains.split(",") if value.strip())
    difficulties = tuple(
        int(value.strip()) for value in args.difficulties.split(",") if value.strip()
    )
    train = frontier_process_task_battery(
        domains,
        difficulties,
        args.train_per_cell,
        seed=args.seed,
        registry_version=CONTAMINATION_SAFE_REGISTRY_VERSION,
    )
    holdout = frontier_process_task_battery(
        domains,
        difficulties,
        args.holdout_per_cell,
        seed=args.seed + args.holdout_seed_offset,
        registry_version=CONTAMINATION_SAFE_REGISTRY_VERSION,
        excluded_prompts={task.prompt for task in train},
    )
    report = audit_public_transition_identifiability(train, holdout)
    report["generation"] = {
        "domains": list(domains),
        "difficulties": list(difficulties),
        "train_per_cell": args.train_per_cell,
        "holdout_per_cell": args.holdout_per_cell,
        "seed": args.seed,
        "holdout_seed_offset": args.holdout_seed_offset,
        "registry_version": CONTAMINATION_SAFE_REGISTRY_VERSION,
        "train_prompt_overlap": 0,
    }
    artifact_body = dict(report)
    artifact_body.pop("artifact_sha256", None)
    report["artifact_sha256"] = hashlib.sha256(
        json.dumps(
            artifact_body,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest()
    atomic_write_json(
        args.output,
        report,
        schema_version=1,
        schema_name=TRANSITION_IDENTIFIABILITY_SCHEMA,
    )
    summary = {
        "output": str(args.output.resolve()),
        "report_sha256": report["report_sha256"],
        "train_tasks": report["train_tasks"],
        "holdout_tasks": report["holdout_tasks"],
        "local_ambiguities": report["audit"]["overall"]["state_current_action"][
            "ambiguous_keys"
        ],
        "full_prefix_ambiguities": report["audit"]["overall"][
            "state_full_public_prefix"
        ]["ambiguous_keys"],
        "state_recurrent_transition_admitted": report["admission"][
            "state_recurrent_transition_admitted"
        ],
        "public_prefix_replay_admitted": report["admission"][
            "public_prefix_replay_admitted"
        ],
        "admitted": report["admission"]["admitted"],
    }
    print(json.dumps(summary, sort_keys=True))
    return 0 if report["admission"]["admitted"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
