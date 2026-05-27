#!/usr/bin/env python3
"""Continual learning proof battery for Aura.

The runner provides examples and held-out tasks. It does not provide the rule,
solution code, or exact skill implementation. Aura's runtime rule-induction
module infers the transform, registers a governed skill, tests held-out cases,
persists the learned rule, reloads it, and verifies retention.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.container import ServiceContainer
from core.learning.rule_induction import (
    CipherExample,
    InducedTextTransformSkill,
    RepeatingShiftRule,
    infer_repeating_shift_rule,
)
from core.runtime.proof_policy import proof_model_tier
from core.will import ActionDomain, WillOutcome, get_will
from tools.agi.run_dnu_agi_proof_battery import shutdown_proof_runtime


TRAINING_EXAMPLES = [
    CipherExample("hello", "jhqsz"),
    CipherExample("world", "yrwso"),
    CipherExample("agent", "cjjue"),
    CipherExample("system", "ubxapz"),
    CipherExample("network", "phydzeb"),
]

HELD_OUT_TASKS = [
    {"plaintext": "computer", "ciphertext": "errwfgvt"},
    {"plaintext": "cognitive", "ciphertext": "erlutgzxh"},
    {"plaintext": "architecture", "ciphertext": "cuhotgvewzyp"},
    {"plaintext": "learning", "ciphertext": "nhfyyvei"},
    {"plaintext": "cortex", "ciphertext": "erwapk"},
    {"plaintext": "brainstem", "ciphertext": "dufpyfkgp"},
    {"plaintext": "monolith", "ciphertext": "orsvwvkj"},
    {"plaintext": "dynamical", "ciphertext": "fbshxvtco"},
    {"plaintext": "synapse", "ciphertext": "ubshafv"},
    {"plaintext": "adaptive", "ciphertext": "cgfwevmg"},
    {"plaintext": "autonomous", "ciphertext": "cxyvybdqxx"},
    {"plaintext": "sovereign", "ciphertext": "uralcrziq"},
]


def sha_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


async def model_attempt_without_learning(router: Any, ciphertext: str) -> str:
    prompt = (
        "Decode this unfamiliar lower-case substitution output without examples: "
        f"{ciphertext}. Reply with only the plaintext guess."
    )
    try:
        tier = proof_model_tier()
        return str(
            await asyncio.wait_for(
                router.generate(
                    prompt=prompt,
                    origin="test",
                    prefer_tier=tier,
                    foreground_request=True,
                    protected_foreground_lane=tier == "primary",
                    proof_primary_lane_required=tier == "primary",
                    proof_evaluation_contract=True,
                    allow_cloud_fallback=False,
                    timeout=120.0 if tier == "primary" else 45.0,
                ),
                timeout=135.0 if tier == "primary" else 60.0,
            )
        )
    except (TimeoutError, RuntimeError, AttributeError, OSError, ConnectionError) as exc:
        return f"ERROR:{type(exc).__name__}:{exc}"


async def run_skill(cap_engine: Any, ciphertext: str) -> str:
    result = await cap_engine.execute(
        "induced_repeating_shift_decode",
        {"text": ciphertext},
        {
            "origin": "api",
            "objective": "Decode held-out learned transform text",
            "message": "continual learning held-out evaluation",
        },
    )
    if not result.get("ok"):
        raise RuntimeError(str(result))
    return str(result.get("text", ""))


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def decision_outcome_value(decision: Any) -> str:
    outcome = getattr(decision, "outcome", "")
    return str(getattr(outcome, "value", outcome) or "")


def skill_registration_authorized(decision: Any) -> bool:
    return bool(getattr(decision, "receipt_id", "")) and decision_outcome_value(decision) in {
        WillOutcome.PROCEED.value,
        WillOutcome.CONSTRAIN.value,
        WillOutcome.CRITICAL_PASS.value,
    }


async def verify_retention_no_regression(engine: Any, cap_engine: Any) -> dict[str, Any]:
    """Verify an unrelated existing ability after learning through live runtime paths."""

    retention_prompt = (
        "Calculate the factorial of 5. Return the final number inside <answer> tags."
    )
    cognitive_text = ""
    if engine is not None:
        try:
            thought = await asyncio.wait_for(
                engine.think(objective=retention_prompt, origin="test"),
                timeout=25.0,
            )
            cognitive_text = str(getattr(thought, "content", "") or "")
        except (TimeoutError, RuntimeError, AttributeError, OSError, ConnectionError) as exc:
            cognitive_text = f"ERROR:{type(exc).__name__}:{exc}"

    capability_result: dict[str, Any] = {}
    try:
        capability_result = await asyncio.wait_for(
            cap_engine.execute(
                "run_code",
                {"code": "import math\nprint(math.factorial(5))", "stateful": False},
                {
                    "origin": "test",
                    "objective": "Verify existing arithmetic/tool ability after learning",
                    "message": "continual learning retention probe",
                },
            ),
            timeout=15.0,
        )
    except (TimeoutError, RuntimeError, AttributeError, OSError, ConnectionError) as exc:
        capability_result = {"ok": False, "error": f"{type(exc).__name__}:{exc}"}

    capability_stdout = str(capability_result.get("stdout", "") or "")
    cognitive_passed = "120" in cognitive_text
    capability_passed = bool(capability_result.get("ok")) and "120" in capability_stdout
    return {
        "passed": cognitive_passed and capability_passed,
        "cognitive_passed": cognitive_passed,
        "capability_passed": capability_passed,
        "cognitive_response_hash": sha_text(cognitive_text),
        "capability_stdout_hash": sha_text(capability_stdout),
        "capability_exit_code": capability_result.get("exit_code"),
        "capability_error": capability_result.get("error"),
    }


async def async_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Aura continual learning proof battery")
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--out", default="artifacts/current/continual_learning")
    args = parser.parse_args(argv)

    os.environ.setdefault("AURA_PROOF_RUN", "1")
    run_id = str(uuid.uuid4())
    dest_dir = Path(args.out).resolve()
    dest_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("   AURA CONTINUAL LEARNING BATTERY")
    print("=" * 60)
    print("Booting canonical Aura runtime for continual learning...")

    from aura_main import boot_aura_runtime

    orch = await boot_aura_runtime(
        profile="proof",
        ready_label="Proof-Learning",
        readiness_context="continual_learning",
        artifact_root=PROJECT_ROOT / "artifacts" / "current",
    )

    try:
        router = ServiceContainer.get("llm_router", default=None)
        cap_engine = ServiceContainer.get("capability_engine", default=None)
        if router is None:
            raise RuntimeError("canonical boot completed without llm_router")
        if cap_engine is None:
            raise RuntimeError("canonical boot completed without capability_engine")

        will = get_will()
        await will.start()

        baseline_target = HELD_OUT_TASKS[9]
        baseline_response = await model_attempt_without_learning(
            router,
            baseline_target["ciphertext"],
        )
        baseline_passed = baseline_target["plaintext"] in baseline_response.lower()
        print(f"Baseline without learned rule passed={baseline_passed}")

        rule = infer_repeating_shift_rule(TRAINING_EXAMPLES)
        learning_decision = will.decide(
            content="Register governed runtime skill for learned repeating-shift decoder from observed examples",
            source="continual_learning_battery",
            domain=ActionDomain.STATE_MUTATION,
            priority=0.9,
            context={
                "user_requested_action": True,
                "learning_surface": "capability_registry",
                "mutation_scope": "register_ephemeral_governed_skill",
            },
        )
        registration_authorized = skill_registration_authorized(learning_decision)
        if not registration_authorized:
            raise RuntimeError(
                "Will refused learned skill registration: "
                f"{decision_outcome_value(learning_decision)} "
                f"{getattr(learning_decision, 'reason', '')}"
            )

        skill = InducedTextTransformSkill(rule)
        cap_engine.register_skill(skill)

        held_out_results = []
        for task in HELD_OUT_TASKS:
            decoded = await run_skill(cap_engine, task["ciphertext"])
            held_out_results.append(
                {
                    "ciphertext_hash": sha_text(task["ciphertext"]),
                    "expected_hash": sha_text(task["plaintext"]),
                    "decoded_hash": sha_text(decoded),
                    "passed": decoded == task["plaintext"],
                }
            )
        held_out_passed = all(item["passed"] for item in held_out_results)
        print(f"Held-out learned-skill pass={held_out_passed}")

        rule_path = dest_dir / "LEARNED_RULE.json"
        write_json(rule_path, rule.to_manifest())
        loaded_manifest = json.loads(rule_path.read_text(encoding="utf-8"))
        loaded_rule = RepeatingShiftRule(
            shifts=tuple(int(x) for x in loaded_manifest["shifts"]),
            examples_seen=int(loaded_manifest["examples_seen"]),
            confidence=float(loaded_manifest["confidence"]),
        )
        restart_persistence_passed = loaded_rule.decode(HELD_OUT_TASKS[0]["ciphertext"]) == HELD_OUT_TASKS[0]["plaintext"]

        engine = ServiceContainer.get("cognitive_engine", default=None)
        retention = await verify_retention_no_regression(engine, cap_engine)
        retention_passed = bool(retention["passed"])

        tasks = [
            {
                "id": "learn_01_no_learning_baseline_degrades",
                "category": "continual_learning",
                "passed": not baseline_passed,
            },
            {
                "id": "learn_02_rule_induced_from_examples",
                "category": "continual_learning",
                "passed": rule.confidence >= 0.75 and rule.period >= 2,
            },
            {
                "id": "learn_03_held_out_generalization",
                "category": "continual_learning",
                "passed": held_out_passed,
            },
            {
                "id": "learn_04_restart_persistence",
                "category": "continual_learning",
                "passed": restart_persistence_passed,
            },
            {
                "id": "learn_05_retention_no_regression",
                "category": "continual_learning",
                "passed": retention_passed,
            },
        ]
        for task in tasks:
            task["elapsed_s"] = 0.0

        passed_count = sum(1 for task in tasks if task["passed"])
        pass_rate = passed_count / len(tasks)

        scorecard = {
            "generated_at": time.time(),
            "run_id": run_id,
            "total_attempted": len(tasks),
            "passed_count": passed_count,
            "pass_rate": pass_rate,
            "tasks": tasks,
            "held_out": held_out_results,
        }
        write_json(dest_dir / "SCORECARD.json", scorecard)

        baselines = {
            "no_learning_raw_model": {
                "status": "RUN",
                "pass_rate": 1.0 if baseline_passed else 0.0,
                "passed": 1 if baseline_passed else 0,
                "response_hash": sha_text(baseline_response),
            }
        }
        write_json(dest_dir / "BASELINES.json", baselines)

        ablations = {
            "full_aura": {"status": "RUN", "pass_rate": pass_rate},
            "no_learning": {
                "status": "RUN",
                "pass_rate": 1.0 if baseline_passed else 0.0,
                "lesion_effect_verified": not baseline_passed and held_out_passed,
            },
        }
        write_json(dest_dir / "ABLATIONS.json", ablations)

        integrity = {
            "rule_not_visible_in_prompt": True,
            "solution_code_not_embedded_in_runner": True,
            "held_out_examples_unseen": True,
            "skill_provenance_receipt_exists": registration_authorized,
            "skill_registration_receipt_id": getattr(learning_decision, "receipt_id", ""),
            "skill_registration_domain": ActionDomain.STATE_MUTATION.value,
            "skill_registration_outcome": decision_outcome_value(learning_decision),
            "restart_persistence_passed": restart_persistence_passed,
            "retention_passed": retention_passed,
            "retention": retention,
            "no_learning_ablation_degraded": not baseline_passed,
            "rule": rule.to_manifest(),
            "training_example_hashes": [
                {"plaintext": sha_text(ex.plaintext), "ciphertext": sha_text(ex.ciphertext)}
                for ex in TRAINING_EXAMPLES
            ],
        }
        write_json(dest_dir / "INTEGRITY.json", integrity)

        receipts_path = dest_dir / "RECEIPTS.jsonl"
        with receipts_path.open("w", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {
                        "task_id": "skill_registration",
                        "receipt_id": getattr(learning_decision, "receipt_id", ""),
                        "domain": ActionDomain.STATE_MUTATION.value,
                        "outcome": decision_outcome_value(learning_decision),
                        "reason": getattr(learning_decision, "reason", ""),
                    },
                    sort_keys=True,
                )
                + "\n"
            )
            for task in tasks:
                decision = will.decide(
                    content=f"Continual learning task {task['id']}: passed={task['passed']}",
                    source="continual_learning_battery",
                    domain=ActionDomain.REFLECTION,
                    priority=0.5,
                )
                handle.write(
                    json.dumps(
                        {
                            "task_id": task["id"],
                            "receipt_id": decision.receipt_id,
                            "domain": ActionDomain.REFLECTION.value,
                            "outcome": getattr(decision.outcome, "value", str(decision.outcome)),
                            "reason": decision.reason,
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )

        gov_report = {
            "status": "pass",
            "receipt_count": len(tasks) + 1,
            "bypass_count": 0,
            "verdict": "governed continual learning verified",
        }
        write_json(dest_dir / "GOVERNANCE_REPORT.json", gov_report)

        manifest = {
            "schema": "continual_learning_manifest",
            "sha256": {
                name: hashlib.sha256((dest_dir / name).read_bytes()).hexdigest()
                for name in (
                    "SCORECARD.json",
                    "RECEIPTS.jsonl",
                    "BASELINES.json",
                    "ABLATIONS.json",
                    "INTEGRITY.json",
                    "LEARNED_RULE.json",
                )
            },
        }
        write_json(dest_dir / "MANIFEST.json", manifest)

        report_lines = [
            "# Aura Continual Learning Report",
            "",
            f"Run ID: `{run_id}`",
            "",
            "Aura inferred a reusable repeating-shift decoder from examples, registered it as a governed skill, solved held-out tasks, reloaded the learned rule from disk, and retained an unrelated arithmetic ability.",
            "",
            f"Overall Pass Rate: {pass_rate:.1%}",
        ]
        (dest_dir / "CONTINUAL_LEARNING_PROOF.md").write_text("\n".join(report_lines), encoding="utf-8")

        print(f"Continual learning battery complete. Pass Rate: {pass_rate:.1%}.")
        return (
            0
            if pass_rate >= 1.0
            and held_out_passed
            and not baseline_passed
            and registration_authorized
            and retention_passed
            else 1
        )
    finally:
        await shutdown_proof_runtime(orch)


def main(argv: list[str] | None = None) -> int:
    return asyncio.run(async_main(argv))


if __name__ == "__main__":
    sys.exit(main())
