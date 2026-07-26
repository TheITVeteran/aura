#!/usr/bin/env python
"""Independently verify and optionally project an RLC action calibration.

The verifier accepts no key material from the certificate itself.  A caller
must provide the externally trusted root PEM separately; that root authenticates
the four-role campaign policy, which in turn authenticates the issuer, runner,
contamination auditor, and evidence verifier attestations.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from core.brain.llm.latent_cortex.action_calibration import (  # noqa: E402
    ACTION_RESOURCE_DIMENSIONS,
    certified_evidence_snapshot,
    verify_action_calibration_certificate,
)
from core.brain.llm.latent_cortex.campaign_journal import (  # noqa: E402
    canonical_json_bytes,
)
from core.brain.llm.latent_cortex.campaign_trust import (  # noqa: E402
    validate_campaign_trust_policy,
)
from core.runtime.file_read_gateway import read_stable_bytes  # noqa: E402
from tools.independent_paired_campaign_scoring import (  # noqa: E402
    _effect_bounds,
)

_INDEPENDENT_ACTION_COUNT = 16
_INDEPENDENT_MIN_CERTIFIED_TASKS = 20
_INDEPENDENT_BOUND_FAMILY_COUNT = 34
_INDEPENDENT_ALPHA = 0.05


def _fraction(value: Mapping[str, Any]) -> float:
    if (
        set(value) != {"numerator", "denominator"}
        or type(value.get("numerator")) is not int
        or type(value.get("denominator")) is not int
        or value["denominator"] <= 0
    ):
        raise ValueError("independent rational field is invalid")
    return value["numerator"] / value["denominator"]


def _independent_cost_upper(costs: list[float]) -> float:
    radius = math.sqrt(
        math.log((2.0 * _INDEPENDENT_BOUND_FAMILY_COUNT) / _INDEPENDENT_ALPHA) / (2.0 * len(costs))
    )
    return min(
        1.0,
        math.ceil((sum(costs) / len(costs) + radius) * 10**12) / 10**12,
    )


def _independent_action_cost(
    resources: Any,
    caps: Mapping[str, int],
) -> float:
    if (
        not isinstance(resources, Mapping)
        or set(resources) != set(ACTION_RESOURCE_DIMENSIONS)
        or any(
            type(resources.get(name)) is not int or not 0 <= resources[name] <= caps[name]
            for name in ACTION_RESOURCE_DIMENSIONS
        )
    ):
        raise ValueError("independent action-resource vector is invalid")
    return max(resources[name] / caps[name] for name in ACTION_RESOURCE_DIMENSIONS)


def _independent_cells(candidate: Mapping[str, Any]) -> dict[str, Any]:
    observations = candidate.get("observations")
    raw_caps = candidate.get("action_resource_caps")
    if (
        not isinstance(observations, list)
        or not isinstance(raw_caps, Mapping)
        or set(raw_caps) != set(ACTION_RESOURCE_DIMENSIONS)
        or any(
            type(raw_caps.get(name)) is not int or raw_caps[name] <= 0
            for name in ACTION_RESOURCE_DIMENSIONS
        )
    ):
        raise ValueError("independent observation envelope is invalid")
    caps = {name: raw_caps[name] for name in ACTION_RESOURCE_DIMENSIONS}
    by_action: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    seen_tasks: set[str] = set()
    for row in observations:
        treatment_resources = (
            row.get("treatment_action_resources") if isinstance(row, Mapping) else None
        )
        control_resources = (
            row.get("control_action_resources") if isinstance(row, Mapping) else None
        )
        if (
            not isinstance(row, Mapping)
            or not isinstance(row.get("action"), str)
            or not isinstance(row.get("task_id"), str)
            or row["task_id"] in seen_tasks
            or type(row.get("treatment_success")) is not bool
            or type(row.get("control_success")) is not bool
            or type(row.get("treatment_action_estimated_flops")) is not int
            or not isinstance(treatment_resources, Mapping)
            or not isinstance(control_resources, Mapping)
            or row.get("treatment_action_estimated_flops")
            != treatment_resources.get("estimated_flops")
        ):
            raise ValueError("independent observation row is invalid")
        _independent_action_cost(
            treatment_resources,
            caps,
        )
        if (
            _independent_action_cost(
                control_resources,
                caps,
            )
            != 0.0
        ):
            raise ValueError("independent control action cost is nonzero")
        seen_tasks.add(row["task_id"])
        by_action[row["action"]].append(row)
    if len(by_action) != _INDEPENDENT_ACTION_COUNT:
        raise ValueError("independent action coverage is invalid")
    cells: dict[str, Any] = {}
    for action in sorted(by_action):
        rows = sorted(by_action[action], key=lambda row: row["task_id"])
        if len(rows) < 8:
            raise ValueError("independent action acquisition floor failed")
        wins = sum(int(row["treatment_success"]) > int(row["control_success"]) for row in rows)
        losses = sum(int(row["treatment_success"]) < int(row["control_success"]) for row in rows)
        ties = len(rows) - wins - losses
        bounds = _effect_bounds(
            wins,
            losses,
            ties,
            _INDEPENDENT_BOUND_FAMILY_COUNT,
        )
        costs = [
            _independent_action_cost(
                row["treatment_action_resources"],
                caps,
            )
            for row in rows
        ]
        cells[action] = {
            "n": len(rows),
            "unique_task_count": len({row["task_id"] for row in rows}),
            "measured": len(rows) >= _INDEPENDENT_MIN_CERTIFIED_TASKS,
            "gain_mean": round((wins - losses) / len(rows), 12),
            "gain_lcb": round(_fraction(bounds["lower"]), 12),
            "gain_ucb": round(_fraction(bounds["upper"]), 12),
            "cost_mean": round(sum(costs) / len(costs), 12),
            "cost_ucb": (
                _independent_cost_upper(costs)
                if len(rows) >= _INDEPENDENT_MIN_CERTIFIED_TASKS
                else 1.0
            ),
            "gain_bounds": {
                "method": ("simultaneous rational Clopper-Pearson contrast bounds"),
                "family_count": bounds["family_count"],
                "family_alpha": bounds["family_alpha"],
                "component_alpha": bounds["component_alpha"],
                "simultaneous_coverage_lower": bounds["simultaneous_coverage_lower"],
                "lower": bounds["lower"],
                "upper": bounds["upper"],
                "certified": bounds["certified"],
            },
            "cost_bounds": {
                "method": "simultaneous Hoeffding upper bound",
                "family_count": _INDEPENDENT_BOUND_FAMILY_COUNT,
                "family_alpha": {"numerator": 1, "denominator": 20},
                "bounded_interval": [0.0, 1.0],
                "normalization": ("max fraction of preregistered action-resource caps"),
                "dimensions": list(ACTION_RESOURCE_DIMENSIONS),
            },
        }
    return cells


def _json_file(path: Path, *, max_bytes: int) -> dict[str, Any]:
    value = json.loads(read_stable_bytes(path, max_bytes=max_bytes))
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} must contain one JSON object")
    return value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Verify an externally trusted RLC action calibration",
    )
    parser.add_argument("--certificate", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--trusted-root", type=Path, required=True)
    parser.add_argument(
        "--now-unix",
        type=int,
        default=None,
        help="Policy validation time; defaults to the current Unix time",
    )
    parser.add_argument(
        "--bucket",
        default=None,
        help="Also emit the worker-safe evidence snapshot for this exact bucket",
    )
    return parser


def verify_files(
    *,
    certificate_path: Path,
    policy_path: Path,
    trusted_root_path: Path,
    now_unix: int,
    bucket: str | None,
) -> dict[str, Any]:
    certificate = _json_file(
        certificate_path,
        max_bytes=64 * 1024 * 1024,
    )
    policy_document = _json_file(
        policy_path,
        max_bytes=2 * 1024 * 1024,
    )
    trust_root = read_stable_bytes(
        trusted_root_path,
        max_bytes=64 * 1024,
    )
    if not isinstance(certificate, Mapping):
        raise ValueError("calibration certificate must be an object")
    candidate_document = certificate.get("candidate")
    if not isinstance(candidate_document, Mapping):
        raise ValueError("calibration certificate candidate must be an object")
    campaign_name = candidate_document.get("campaign_name")
    policy = validate_campaign_trust_policy(
        policy_document,
        trusted_root_public_key_pem=trust_root,
        expected_campaign_name=campaign_name,
        now_unix=now_unix,
    )
    verified = verify_action_calibration_certificate(
        certificate,
        policy=policy,
    )
    candidate = verified["candidate"]
    independent_cells = _independent_cells(candidate)
    if canonical_json_bytes(independent_cells) != canonical_json_bytes(candidate["cells"]):
        raise ValueError("independent action statistics disagree with production")
    result = {
        "schema": "aura.rlc.action_calibration.independent_verdict.v1",
        "accepted": True,
        "certificate_sha256": verified["certificate_sha256"],
        "policy_sha256": policy.policy_sha256,
        "campaign_name": candidate["campaign_name"],
        "calibration_bucket": candidate["calibration_bucket"],
        "pair_count": candidate["pair_count"],
        "execution_count": candidate["execution_count"],
        "independent_statistics_agree": True,
        "measured_actions": sorted(
            action for action, cell in candidate["cells"].items() if cell["measured"]
        ),
        "frontier_claim_eligible": False,
        "limitations": list(candidate["limitations"]),
    }
    if bucket is not None:
        result["evidence_snapshot"] = certified_evidence_snapshot(
            verified,
            policy=policy,
            bucket=bucket,
        )
    return result


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        verdict = verify_files(
            certificate_path=args.certificate,
            policy_path=args.policy,
            trusted_root_path=args.trusted_root,
            now_unix=(int(time.time()) if args.now_unix is None else args.now_unix),
            bucket=args.bucket,
        )
    except (
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as exc:
        print(
            json.dumps(
                {
                    "schema": ("aura.rlc.action_calibration.independent_verdict.v1"),
                    "accepted": False,
                    "error": f"{type(exc).__name__}:{str(exc)[:400]}",
                },
                sort_keys=True,
                separators=(",", ":"),
            ),
            file=sys.stderr,
        )
        return 1
    sys.stdout.buffer.write(canonical_json_bytes(verdict) + b"\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
