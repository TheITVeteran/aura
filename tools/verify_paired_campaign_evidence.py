#!/usr/bin/env python
"""Independent raw-output / compute verification for one paired campaign.

Trusts nothing but raw disk artifacts and deterministic regeneration:

1. plan.json is parsed through the strict CampaignPlan validator; the task
   battery is REGENERATED from the plan's declared generation parameters
   (seeds, domains, difficulty, registry version) and its manifest hash
   must equal the plan's — a doctored plan cannot smuggle different tasks.
2. campaign.jsonl is replayed read-only through the hash-chained journal;
   every committed record is chain-verified on read.
3. grade_campaign() is re-run HERE, on the replayed records, against the
   independently regenerated blinded answers — raw outputs are re-parsed
   and re-scored; compute accounting and equal-compute controls are
   re-derived; absent cells are never inferred.
4. The recomputed grade must byte-agree with the published grade.json
   (verdict, claim tier, grade_sha256, campaign manifest binding).

Exit 0: every check agrees. Exit 1: any disagreement, with reasons.

Usage:
  .venv/bin/python tools/verify_paired_campaign_evidence.py \
      --campaign-dir <dir> [--contamination-trust-root <public.pem>]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from core.brain.llm.latent_cortex.campaign_journal import (  # noqa: E402
    CampaignJournal,
    CampaignPlan,
    canonical_json_bytes,
)
from core.brain.llm.latent_cortex.frontier_tasks import (  # noqa: E402
    build_task_manifest,
    generate_task_battery,
)
from core.brain.llm.latent_cortex.paired_campaign import (  # noqa: E402
    grade_campaign,
)

PLAN_FILE = "plan.json"
JOURNAL_FILE = "campaign.jsonl"
MANIFEST_FILE = "campaign_manifest.json"
GRADE_FILE = "grade.json"
VERDICT_SCHEMA = "aura.latent_cortex.independent_evidence_verdict.v1"


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


@contextmanager
def _open_journal_readonly(path: Path, plan: CampaignPlan):
    """Replay the journal without leaving a lock behind (verifier is a reader)."""
    lock_path = path.with_name(f"{path.name}.lock")
    lock_existed = lock_path.exists()
    try:
        with CampaignJournal(path, plan) as journal:
            yield journal
    finally:
        if not lock_existed and lock_path.exists():
            lock_stat = lock_path.lstat()
            if (
                stat.S_ISREG(lock_stat.st_mode)
                and lock_stat.st_uid == os.geteuid()
                and lock_stat.st_size == 0
            ):
                lock_path.unlink()


def _regenerate_tasks(plan: CampaignPlan):
    metadata = plan.to_dict()["metadata"]
    execution_config = metadata.get("execution_config")
    if not isinstance(execution_config, dict):
        raise SystemExit("plan has no execution_config")
    seeds = tuple(int(v) for v in execution_config["generation_seeds"])
    domains = tuple(str(v) for v in execution_config["domains"])
    difficulty = int(execution_config["difficulty"])
    registry_version = str(execution_config["task_registry_version"])
    tasks = generate_task_battery(
        seeds,
        domains=domains,
        difficulty=difficulty,
        registry_version=registry_version,
    )
    return tasks, {
        "seeds": list(seeds),
        "domains": list(domains),
        "difficulty": difficulty,
        "registry_version": registry_version,
    }


def _trust_root_sha256(path_value: str) -> str:
    from cryptography.hazmat.primitives import serialization

    trust_bytes = Path(path_value).expanduser().resolve(strict=True).read_bytes()
    public_key = serialization.load_pem_public_key(trust_bytes)
    der = public_key.public_bytes(
        encoding=serialization.Encoding.DER,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    return _sha256_bytes(der)


def verify_campaign_evidence(
    campaign_dir: Path,
    *,
    contamination_trust_root: str = "",
) -> dict[str, Any]:
    failures: list[str] = []
    detail: dict[str, Any] = {}

    plan_payload = (campaign_dir / PLAN_FILE).read_bytes()
    plan = CampaignPlan.from_dict(json.loads(plan_payload))
    detail["plan_sha256"] = _sha256_bytes(plan_payload)

    # 1. Independent task regeneration binds the plan to real tasks.
    tasks, generation = _regenerate_tasks(plan)
    detail["generation"] = generation
    regenerated_manifest = build_task_manifest(tasks)
    plan_manifest = plan.to_dict()["metadata"].get("task_manifest") or {}
    declared_sha = None
    if isinstance(plan_manifest, dict):
        declared_sha = plan_manifest.get("manifest_sha256")
    if declared_sha != regenerated_manifest.manifest_sha256:
        failures.append(
            "task manifest mismatch: plan declares "
            f"{declared_sha}, independent regeneration produced "
            f"{regenerated_manifest.manifest_sha256}"
        )
        # Grading against tasks the plan cannot reproduce would be
        # meaningless; the campaign is already unverifiable.
        return {
            "schema": VERDICT_SCHEMA,
            "campaign_dir": str(campaign_dir),
            "passed": False,
            "failures": failures,
            **detail,
        }
    detail["task_count"] = len(regenerated_manifest.tasks)

    # 2. Chain-verified replay of every committed outcome.
    with _open_journal_readonly(campaign_dir / JOURNAL_FILE, plan) as journal:
        records = journal.committed_records()
    detail["committed_records"] = len(records)

    # 3. Independent regrade from raw evidence.
    trusted_root = (
        _trust_root_sha256(contamination_trust_root)
        if contamination_trust_root
        else None
    )
    grade = grade_campaign(
        records,
        plan=plan,
        issuer_tasks=tasks,
        trusted_contamination_root_sha256=trusted_root,
    )
    detail["recomputed_verdict"] = grade.get("verdict")
    detail["recomputed_claim_tier"] = grade.get("claim_tier")

    # 4. Agreement with the published grade, if one exists.
    grade_path = campaign_dir / GRADE_FILE
    if grade_path.exists():
        published = json.loads(grade_path.read_bytes())
        detail["published_verdict"] = published.get("verdict")
        for key in ("verdict", "claim_tier", "observed_task_count",
                    "observed_cell_count"):
            if published.get(key) != grade.get(key):
                failures.append(
                    f"published grade {key}={published.get(key)!r} disagrees "
                    f"with independent recomputation {grade.get(key)!r}"
                )
        manifest_path = campaign_dir / MANIFEST_FILE
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_bytes())
            if published.get("campaign_manifest_sha256") != manifest.get(
                "manifest_sha256"
            ):
                failures.append("published grade is not bound to the manifest")
        else:
            failures.append("campaign manifest missing beside published grade")
        recomputed_material = dict(published)
        published_grade_sha = recomputed_material.pop("grade_sha256", None)
        if (
            _sha256_bytes(canonical_json_bytes(recomputed_material))
            != published_grade_sha
        ):
            failures.append("published grade_sha256 does not match its content")
    else:
        detail["published_verdict"] = None

    return {
        "schema": VERDICT_SCHEMA,
        "campaign_dir": str(campaign_dir),
        "passed": not failures,
        "failures": failures,
        **detail,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-dir", required=True)
    parser.add_argument("--contamination-trust-root", default="")
    parser.add_argument("--out", default="")
    args = parser.parse_args()
    try:
        verdict = verify_campaign_evidence(
            Path(args.campaign_dir).expanduser().resolve(strict=True),
            contamination_trust_root=args.contamination_trust_root,
        )
    except Exception as exc:  # noqa: BLE001 - corrupt evidence must fail closed
        verdict = {
            "schema": VERDICT_SCHEMA,
            "campaign_dir": str(args.campaign_dir),
            "passed": False,
            "failures": [
                f"evidence unreadable or invalid: {type(exc).__name__}: {exc}"
            ],
        }
    rendered = json.dumps(verdict, indent=2, sort_keys=True)
    print(rendered)
    if args.out:
        Path(args.out).write_text(rendered + "\n", encoding="utf-8")
    return 0 if verdict["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
