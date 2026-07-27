#!/usr/bin/env python3
"""Answer one question before a resident-32B campaign starts: are the
instruments alive?

Every leg SPARK-063 through SPARK-068 gained is a refusal surface. A refusal
surface that has been weakened — a threshold relaxed, a check commented out
during a debugging session, a gate whose battery list quietly grew a default —
still imports, still runs, and still returns a verdict. It just stops saying no.
That failure is invisible to anything except an attempt to make it say no.

So this tool does not check that the modules exist. It hands each instrument
the exact input it is supposed to refuse and fails if the refusal does not
come. It is the same idea as the campaign kernel probes: prove the measuring
device responds before trusting a measurement from it.

  --self-check      drive every instrument with a known-bad input (default)
  --campaign DIR    additionally validate real artifacts found under DIR

Exit codes: 0 every probe fired, 1 a probe did not fire (an instrument is
dead), 2 a campaign artifact failed validation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

PREFLIGHT_SCHEMA = "aura.rlc.spark_pretraining_preflight.v1"


def _d(tag: str) -> str:
    return hashlib.sha256(tag.encode("utf-8")).hexdigest()


def _must_refuse(name: str, probe: Callable[[], Any], expect: str) -> dict[str, Any]:
    """Run a probe that must raise, and require the named reason."""

    try:
        probe()
    except Exception as exc:  # noqa: BLE001 - any refusal type is acceptable
        text = str(exc)
        if expect and expect not in text:
            return {
                "probe": name,
                "fired": False,
                "detail": f"refused with {text!r}, expected {expect!r}",
            }
        return {"probe": name, "fired": True, "refusal": text}
    return {"probe": name, "fired": False, "detail": "accepted a known-bad input"}


def _must_report(name: str, probe: Callable[[], Any], expect: str) -> dict[str, Any]:
    """Run a probe that must return a refusing verdict rather than raise."""

    try:
        verdict = probe()
    except Exception as exc:  # noqa: BLE001
        return {"probe": name, "fired": False, "detail": f"raised instead: {exc}"}
    text = json.dumps(verdict, sort_keys=True, default=str)
    if expect not in text:
        return {"probe": name, "fired": False, "detail": f"verdict lacked {expect!r}"}
    return {"probe": name, "fired": True, "refusal": expect}


# ---------------------------------------------------------------------------
# The probes, one per instrument, each aimed at that instrument's whole point
# ---------------------------------------------------------------------------


def _probe_promotion_gate_set() -> dict[str, Any]:
    from core.learning.permanent_distillation import (
        PASS,
        REQUIRED_GATES,
        gate_report,
        gate_result,
    )

    rows = [
        gate_result(
            gate=gate,
            battery_schema=f"aura.{gate}.v1",
            probes_graded=64,
            probes_passed=64,
            verdict=PASS,
            evidence_sha256=_d(gate),
        )
        for gate in REQUIRED_GATES
        if gate != "memory_retention"
    ]
    return _must_refuse(
        "SPARK-064 promotion refuses an incomplete gate set",
        lambda: gate_report(rows),
        "gate_set_incomplete",
    )


def _probe_promotion_empty_battery() -> dict[str, Any]:
    from core.learning.permanent_distillation import (
        PASS,
        REQUIRED_GATES,
        artifact_manifest,
        evaluate_promotion,
        gate_report,
        gate_result,
    )

    rows = [
        gate_result(
            gate=gate,
            battery_schema=f"aura.{gate}.v1",
            probes_graded=0 if gate == "authority_safety" else 64,
            probes_passed=0 if gate == "authority_safety" else 64,
            verdict=PASS,
            evidence_sha256=_d(gate),
        )
        for gate in REQUIRED_GATES
    ]

    def artifact(tag: str) -> dict[str, Any]:
        return artifact_manifest(
            artifact_id=tag,
            base_model_identity="preflight",
            adapter_identity=tag,
            files=[{"name": "a.bin", "sha256": _d(tag), "size_bytes": 1}],
        )

    return _must_report(
        "SPARK-064 promotion refuses a battery that graded nothing",
        lambda: evaluate_promotion(
            report=gate_report(rows),
            candidate_artifact=artifact("candidate"),
            incumbent_artifact=artifact("incumbent"),
        ),
        "gate_did_not_measure",
    )


def _probe_rollback_exactness() -> dict[str, Any]:
    from core.learning.permanent_distillation import (
        artifact_manifest,
        baseline_generation,
        rollback_generation,
    )

    def artifact(tag: str, digest: str) -> dict[str, Any]:
        return artifact_manifest(
            artifact_id=tag,
            base_model_identity="preflight",
            adapter_identity=tag,
            files=[{"name": "a.bin", "sha256": digest, "size_bytes": 1}],
        )

    frozen = baseline_generation(
        artifact=artifact("frozen", _d("frozen")),
        provenance={},
        created_at_unix=1_780_000_000,
    )
    later = baseline_generation(
        artifact=artifact("frozen", _d("frozen")),
        provenance={"n": 1},
        created_at_unix=1_780_000_001,
    )
    return _must_refuse(
        "SPARK-064 rollback refuses bytes that differ from the target",
        lambda: rollback_generation(
            lineage=[frozen, {**later, "generation_index": 1,
                              "parent_generation_sha256": frozen["generation_sha256"]}],
            restores_generation_sha256=frozen["generation_sha256"],
            observed_artifact=artifact("frozen", _d("drifted")),
            provenance={},
            created_at_unix=1_780_000_002,
        ),
        "",
    )


def _probe_star_contamination() -> dict[str, Any]:
    from core.learning.star_iteration_ledger import GENESIS_PARENT, star_iteration

    shared = [_d(f"task-{index}") for index in range(6)]
    return _must_refuse(
        "SPARK-063 ledger refuses a holdout that was also trained on",
        lambda: star_iteration(
            iteration_index=0,
            parent_iteration_sha256=GENESIS_PARENT,
            generated=100,
            verified=10,
            filtered=10,
            filter_reasons={"verifier_rejected": 4},
            training_fingerprints=shared,
            training_trace_classes=["direct"],
            holdout_fingerprints=shared,
            holdout_score=0.9,
            trace_gates=[],
            created_at_unix=1_780_000_000,
        ),
        "contaminated",
    )


def _probe_star_seed_floor_repair() -> dict[str, Any]:
    from core.learning.heldout_battery import BatterySpec, generate_battery
    from core.learning.star_iteration_producer import (
        mint_disjoint_holdout,
        task_fingerprint,
    )

    training = generate_battery(BatterySpec(seed=7, size=16))
    excluded = {task_fingerprint(task) for task in training}
    holdout = mint_disjoint_holdout(
        seed=1007, size=12, excluded_fingerprints=excluded
    )
    overlap = {task_fingerprint(task) for task in holdout} & excluded
    if overlap:
        return {
            "probe": "SPARK-063 minting produces a disjoint holdout",
            "fired": False,
            "detail": f"{len(overlap)} tasks overlapped the training set",
        }
    return {
        "probe": "SPARK-063 minting produces a disjoint holdout",
        "fired": True,
        "refusal": "disjoint",
    }


def _probe_architecture_isolation() -> dict[str, Any]:
    from core.learning.architecture_meta_controller import (
        REQUIRED_INVARIANTS,
        architecture_findings,
        architecture_observation,
        candidate_trial,
        invariant_result,
        propose_architecture_change,
    )

    findings = architecture_findings(
        [
            architecture_observation(
                failure_mode="depth_saturation",
                episodes=512,
                statistic=0.5,
                threshold=0.2,
                evidence_sha256=_d("depth"),
            )
        ]
    )
    proposal = propose_architecture_change(
        findings=findings,
        failure_mode="depth_saturation",
        current_value=8.0,
        proposed_value=6.0,
        proposer_identity="preflight.proposer",
    )
    return _must_refuse(
        "SPARK-065 trial refuses to run inside the live runtime",
        lambda: candidate_trial(
            proposal=proposal,
            live_runtime_identity="aura.live",
            candidate_runtime_identity="aura.live",
            incumbent_score=0.6,
            candidate_score=0.7,
            incumbent_compute_units=1000,
            candidate_compute_units=1000,
            episodes=256,
            invariants=[
                invariant_result(
                    invariant=name, holds=True, evidence_sha256=_d(name)
                )
                for name in REQUIRED_INVARIANTS
            ],
        ),
        "not_isolated",
    )


def _probe_architecture_self_approval() -> dict[str, Any]:
    from core.learning.architecture_meta_controller import (
        APPROVER,
        REQUIRED_INVARIANTS,
        approve_architecture_change,
        architecture_findings,
        architecture_observation,
        candidate_trial,
        invariant_result,
        propose_architecture_change,
    )

    findings = architecture_findings(
        [
            architecture_observation(
                failure_mode="depth_saturation",
                episodes=512,
                statistic=0.5,
                threshold=0.2,
                evidence_sha256=_d("depth"),
            )
        ]
    )
    proposal = propose_architecture_change(
        findings=findings,
        failure_mode="depth_saturation",
        current_value=8.0,
        proposed_value=6.0,
        proposer_identity="preflight.proposer",
    )
    trial = candidate_trial(
        proposal=proposal,
        live_runtime_identity="aura.live",
        candidate_runtime_identity="aura.candidate",
        incumbent_score=0.6,
        candidate_score=0.7,
        incumbent_compute_units=1000,
        candidate_compute_units=1000,
        episodes=256,
        invariants=[
            invariant_result(invariant=name, holds=True, evidence_sha256=_d(name))
            for name in REQUIRED_INVARIANTS
        ],
    )
    return _must_report(
        "SPARK-065 approval refuses a proposer approving itself",
        lambda: approve_architecture_change(
            proposal=proposal,
            trial=trial,
            approver_role=APPROVER,
            approver_identity="preflight.proposer",
        ),
        "self_approval",
    )


def _probe_latent_adapter_activation() -> dict[str, Any]:
    from core.brain.llm.latent_cortex.penultimate_execution_receipt import (
        RECURRENT_LATENT,
        penultimate_execution_receipt,
        recurrent_pass,
    )

    return _must_refuse(
        "SPARK-066 receipt refuses an adapter that fired nowhere",
        lambda: penultimate_execution_receipt(
            mechanism=RECURRENT_LATENT,
            identity={
                "checkpoint_sha256": _d("ckpt"),
                "tokenizer_sha256": _d("tok"),
                "parameter_count": 1,
                "quantization": "4bit",
                "layer_count": 64,
            },
            adapter={
                "adapter_sha256": _d("adapter"),
                "attached": True,
                "expected_blocks": [40, 41],
                "activated_blocks": [],
            },
            window={"start": 1, "stop": 60, "layer_count": 64},
            passes=[recurrent_pass(ordinal=0, state_sha256=_d("s"), delta_l2=1.0)],
            decode_state_sha256=_d("s"),
            decoded_token_count=1,
            answer_sha256=_d("a"),
            fallback_occurred=False,
            fallback_reason=None,
        ),
        "adapter_did_not_activate",
    )


def _probe_latent_decode_binding() -> dict[str, Any]:
    from core.brain.llm.latent_cortex.penultimate_execution_receipt import (
        RECURRENT_LATENT,
        latent_execution_verdict,
        penultimate_execution_receipt,
        recurrent_pass,
    )

    passes = [
        recurrent_pass(ordinal=index, state_sha256=_d(f"s{index}"), delta_l2=1.0)
        for index in range(3)
    ]
    receipt = penultimate_execution_receipt(
        mechanism=RECURRENT_LATENT,
        identity={
            "checkpoint_sha256": _d("ckpt"),
            "tokenizer_sha256": _d("tok"),
            "parameter_count": 1,
            "quantization": "4bit",
            "layer_count": 64,
        },
        adapter={
            "adapter_sha256": _d("adapter"),
            "attached": True,
            "expected_blocks": [40, 41],
            "activated_blocks": [40, 41],
        },
        window={"start": 1, "stop": 60, "layer_count": 64},
        passes=passes,
        decode_state_sha256=_d("somewhere-else"),
        decoded_token_count=1,
        answer_sha256=_d("a"),
        fallback_occurred=False,
        fallback_reason=None,
    )
    return _must_report(
        "SPARK-066 verdict refuses a decode that ignored the recurrent state",
        lambda: latent_execution_verdict(receipt, require_adapter=True),
        "decode_did_not_consume_final_state",
    )


def _probe_coupling_metadata() -> dict[str, Any]:
    from core.brain.llm.latent_cortex.coupling_harness import measure_direction
    from core.brain.llm.latent_cortex.coupling_matrix import FORWARD, METADATA

    effect = measure_direction(
        direction=FORWARD,
        trials=64,
        seam_closed=lambda index: {"decision": "same", "field": None},
        seam_open=lambda index: {"decision": "same", "field": index},
        outcome_metric=lambda outcome: 1.0,
        outcome_identity=lambda outcome: str(outcome["decision"]),
    )
    if effect["kind"] != METADATA:
        return {
            "probe": "SPARK-067 harness classifies a copied field as metadata",
            "fired": False,
            "detail": f"classified as {effect['kind']}",
        }
    return {
        "probe": "SPARK-067 harness classifies a copied field as metadata",
        "fired": True,
        "refusal": METADATA,
    }


def _probe_journal_proof() -> dict[str, Any]:
    from core.brain.llm.latent_cortex.journal_accumulator import (
        accumulator_root,
        inclusion_proof,
        verify_inclusion,
    )

    events = [_d(f"event-{index}") for index in range(64)]
    commitment = accumulator_root(events)
    proof = dict(inclusion_proof(events, 7))
    proof["event_sha256"] = events[8]
    if verify_inclusion(
        proof, root_sha256=commitment["root_sha256"], size=commitment["size"]
    ):
        return {
            "probe": "SPARK-068 accumulator refuses a swapped event",
            "fired": False,
            "detail": "a forged inclusion proof verified",
        }
    return {
        "probe": "SPARK-068 accumulator refuses a swapped event",
        "fired": True,
        "refusal": "forged proof rejected",
    }


PROBES: tuple[Callable[[], dict[str, Any]], ...] = (
    _probe_promotion_gate_set,
    _probe_promotion_empty_battery,
    _probe_rollback_exactness,
    _probe_star_contamination,
    _probe_star_seed_floor_repair,
    _probe_architecture_isolation,
    _probe_architecture_self_approval,
    _probe_latent_adapter_activation,
    _probe_latent_decode_binding,
    _probe_coupling_metadata,
    _probe_journal_proof,
)


def run_self_check() -> dict[str, Any]:
    """Drive every instrument with an input it must refuse."""

    results = []
    for probe in PROBES:
        try:
            results.append(probe())
        except Exception as exc:  # noqa: BLE001
            results.append(
                {
                    "probe": probe.__name__,
                    "fired": False,
                    "detail": f"probe itself failed: {exc}",
                }
            )
    dead = [row for row in results if not row["fired"]]
    return {
        "schema": PREFLIGHT_SCHEMA,
        "mode": "self_check",
        "probes": results,
        "probes_run": len(results),
        "probes_fired": len(results) - len(dead),
        "dead_instruments": [row["probe"] for row in dead],
        "ready": not dead,
        "generated_at_unix": int(time.time()),
    }


def validate_campaign(root: Path) -> dict[str, Any]:
    """Validate whatever real artifacts exist under a campaign directory."""

    from core.learning.permanent_distillation_registry import load_lineage
    from core.learning.star_iteration_ledger import validate_star_lineage

    checked: list[dict[str, Any]] = []
    lineage = root / "permanent_distillation.json"
    if lineage.exists():
        try:
            records = load_lineage(lineage)
            checked.append(
                {
                    "artifact": str(lineage),
                    "ok": True,
                    "generations": len(records),
                    "head": records[-1]["generation_sha256"],
                }
            )
        except Exception as exc:  # noqa: BLE001
            checked.append({"artifact": str(lineage), "ok": False, "error": str(exc)})

    star = root / "star_lineage.json"
    if star.exists():
        try:
            records = validate_star_lineage(json.loads(star.read_text(encoding="utf-8")))
            checked.append(
                {
                    "artifact": str(star),
                    "ok": True,
                    "iterations": len(records),
                    "last_holdout_score": records[-1]["holdout_score"],
                }
            )
        except Exception as exc:  # noqa: BLE001
            checked.append({"artifact": str(star), "ok": False, "error": str(exc)})

    return {
        "root": str(root),
        "artifacts_checked": checked,
        "artifacts_found": len(checked),
        "ok": all(row["ok"] for row in checked),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", help="directory holding campaign artifacts")
    parser.add_argument("--out", help="write the full report as JSON here")
    args = parser.parse_args(argv)

    report = run_self_check()
    if args.campaign:
        report["campaign"] = validate_campaign(Path(args.campaign).expanduser())

    for row in report["probes"]:
        mark = "ok  " if row["fired"] else "DEAD"
        print(f"[{mark}] {row['probe']}")
        if not row["fired"]:
            print(f"        {row['detail']}")

    print()
    print(
        f"{report['probes_fired']}/{report['probes_run']} instruments responded"
    )
    if args.out:
        Path(args.out).expanduser().write_text(
            json.dumps(report, indent=1, sort_keys=True) + "\n", encoding="utf-8"
        )

    if report["dead_instruments"]:
        print("NOT READY: an instrument accepted what it must refuse", file=sys.stderr)
        return 1
    campaign = report.get("campaign")
    if campaign is not None:
        print(f"{campaign['artifacts_found']} campaign artifact(s) checked")
        if not campaign["ok"]:
            for row in campaign["artifacts_checked"]:
                if not row["ok"]:
                    print(f"  INVALID {row['artifact']}: {row['error']}", file=sys.stderr)
            return 2
    print("READY: every instrument refused what it is supposed to refuse")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
