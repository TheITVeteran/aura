#!/usr/bin/env python3
"""Operational frontier standards matrix for Aura closeout.

This is intentionally not a marketing checklist. Each target standard maps to
source evidence and executable validators. A standard is "established" only
when every required source path and every validator path exists. Live proof
artifacts remain separate because they are time- and machine-dependent.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class FrontierStandard:
    key: str
    label: str
    acceptance: str
    source_paths: tuple[str, ...]
    validator_paths: tuple[str, ...]
    live_artifacts: tuple[str, ...] = ()
    reference_models: tuple[str, ...] = ()


@dataclass(frozen=True)
class StandardStatus:
    key: str
    label: str
    status: str
    missing_sources: tuple[str, ...]
    missing_validators: tuple[str, ...]
    missing_live_artifacts: tuple[str, ...]
    acceptance: str
    reference_models: tuple[str, ...]


STANDARDS: tuple[FrontierStandard, ...] = (
    FrontierStandard(
        key="daily_runtime_reliability",
        label="Daily Runtime Reliability",
        acceptance=(
            "Launched desktop path boots, chats, perceives, acts, retries, and shuts down "
            "without fallback loops, duplicate kernels, RAM runaway, or false health."
        ),
        source_paths=(
            "core/runtime/desktop_boot_safety.py",
            "core/subsystem_audit.py",
            "interface/routes/chat.py",
            "tools/live_boot_proof.py",
        ),
        validator_paths=(
            "tests/test_desktop_boot_safety.py",
            "tests/test_full_desktop_runtime_contract.py",
            "tests/test_server_conversation_lane.py",
        ),
        live_artifacts=("artifacts/current/live_desktop_runtime",),
    ),
    FrontierStandard(
        key="humanlike_conversation",
        label="Conversational Human-Likeness",
        acceptance=(
            "Conversation remains contextual, self-consistent, non-canned, and full-mind "
            "routed over multi-turn desktop use."
        ),
        source_paths=(
            "core/brain/cognitive_engine.py",
            "core/voice/response_shaper.py",
            "core/conversation/response_reliability.py",
            "interface/routes/chat.py",
        ),
        validator_paths=(
            "tests/test_chat_human_level_contract.py",
            "tests/test_live_mind_generation_controls.py",
            "tests/test_server_conversation_lane.py",
        ),
        reference_models=("Samantha/SAM", "Data", "EDI", "Cortana"),
    ),
    FrontierStandard(
        key="sci_fi_ai_capability",
        label="Sci-Fi AI Capability Envelope",
        acceptance=(
            "Capabilities from fictional assistants map to governed real mechanisms: "
            "local/web actuation, memory, perception, speech, self-modeling, safety, "
            "planning, and sustained autonomy."
        ),
        source_paths=(
            "core/capability_engine.py",
            "core/runtime/desktop_action_gateway.py",
            "core/actuation/desktop_actuator.py",
            "core/perception/screen_perception.py",
            "core/social/social_imagination.py",
        ),
        validator_paths=(
            "tests/test_desktop_agency.py",
            "tests/test_desktop_planning_generality.py",
            "tests/test_capability_gateway_routing_runtime.py",
        ),
        reference_models=(
            "JARVIS",
            "Cortana",
            "EDI",
            "MIST",
            "Culture Minds",
            "Data",
            "Kokoro/Koroko",
            "Ava",
            "Caine",
        ),
    ),
    FrontierStandard(
        key="phenomenal_building_blocks",
        label="Phenomenal-State Building Blocks",
        acceptance=(
            "Do not claim subjectivity; prove functional correlates that are causal, "
            "ablatable, introspectable, and falsifiable against zombie/baseline controls."
        ),
        source_paths=(
            "core/phenomenal_substrate/experience_engine.py",
            "core/phenomenal_substrate/global_workspace.py",
            "core/consciousness/phenomenal_falsification.py",
            "core/affect/phenomenal_integration.py",
        ),
        validator_paths=(
            "tests/phenomenal/test_battery.py",
            "tests/test_phenomenal_falsification.py",
            "tests/test_phenomenal_causal_routing.py",
        ),
    ),
    FrontierStandard(
        key="frontier_reasoning_outside_model",
        label="Frontier Reasoning Outside The Model",
        acceptance=(
            "Native reasoning organs, exact solvers, verifier-derived preference data, "
            "and local reasoning-model lanes improve outputs without external servers."
        ),
        source_paths=(
            "core/reasoning/native_system2.py",
            "core/brain/reasoning_amplifier_v2.py",
            "core/brain/tool_augmented_reasoning.py",
            "core/learning/verifiable_preference_harness.py",
            "core/learning/model_merge.py",
        ),
        validator_paths=(
            "tests/test_tool_augmented_reasoning.py",
            "tests/test_reasoning_self_improvement.py",
            "tests/test_model_merge.py",
            "tests/test_verifiable_preference_harness.py",
        ),
    ),
    FrontierStandard(
        key="superintelligence_trajectory",
        label="Superintelligence Trajectory",
        acceptance=(
            "Show bounded recursive improvement ingredients: discovery, self-critique, "
            "safe mutation, validation, preference harvest, and local model upgrade path."
        ),
        source_paths=(
            "core/discovery/frontier_discovery_engine.py",
            "core/self_modification/mutation_safety.py",
            "core/self_modification/safe_modification.py",
            "core/learning/verifiable_preference_harness.py",
            "scripts/export_verifiable_preferences.py",
        ),
        validator_paths=(
            "tests/test_frontier_discovery_engine.py",
            "tests/test_rsi_expansion_components.py",
            "tests/test_verifiable_preference_harness.py",
        ),
    ),
    FrontierStandard(
        key="os_control_frontier",
        label="Frontier OS Control",
        acceptance=(
            "Computer use is perception-driven, governed, visible when requested, "
            "effect-verified, and recoverable from focus/popup/context failures."
        ),
        source_paths=(
            "core/body/desktop_motor.py",
            "core/actuation/desktop_actuator.py",
            "core/agency/desktop_planner.py",
            "core/runtime/desktop_task_contract.py",
            "core/security/native_desktop_bridge.py",
        ),
        validator_paths=(
            "tests/test_desktop_capabilities_runtime.py",
            "tests/test_desktop_planner.py",
            "tests/test_desktop_task_skill.py",
            "tests/test_native_desktop_bridge.py",
        ),
    ),
    FrontierStandard(
        key="nethack_general_environment",
        label="NetHack-Class General Environment Competence",
        acceptance=(
            "Use a general environment kernel with perception, memory horizon, reflexes, "
            "planning, and post-mortem learning rather than NetHack-only shortcuts."
        ),
        source_paths=(
            "core/environments/terminal_grid/nethack_adapter.py",
            "core/environments/terminal_grid/nethack_parser.py",
            "core/embodiment/games/nethack/state_compiler.py",
            "challenges/nethack_challenge.py",
        ),
        validator_paths=(
            "tests/nethack_crucible.py",
            "tests/environments/terminal_grid/test_nethack_adapter_preflight.py",
            "tests/environments/terminal_grid/test_nethack_audit_comprehensive.py",
            "tests/test_nethack_memory_horizon.py",
        ),
    ),
    FrontierStandard(
        key="generally_capable_ai",
        label="Generally Capable AI",
        acceptance=(
            "Demonstrate cross-domain planning, tool use, memory, learning, self-repair, "
            "reasoning, and transfer under hidden or sealed tests."
        ),
        source_paths=(
            "tools/agi/run_dnu_agi_proof_battery.py",
            "tools/agi/validate_dnu_final_bundle.py",
            "core/capability_engine.py",
            "core/brain/cognitive_engine.py",
        ),
        validator_paths=(
            "tests/agi/live/test_dnu_agi_proof_battery.py",
            "tests/test_capability_gateway_routing_runtime.py",
            "tests/test_autonomous_task_engine_runtime.py",
        ),
        live_artifacts=("artifacts/current/agi_live",),
    ),
)


def _missing(paths: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(path for path in paths if not (ROOT / path).exists())


def evaluate(*, require_live: bool = False) -> list[StandardStatus]:
    statuses: list[StandardStatus] = []
    for standard in STANDARDS:
        missing_sources = _missing(standard.source_paths)
        missing_validators = _missing(standard.validator_paths)
        missing_live = _missing(standard.live_artifacts) if require_live else ()
        if missing_sources or missing_validators or missing_live:
            status = "gap"
        else:
            status = "source_and_validator_mapped"
            if standard.live_artifacts:
                status = "source_validator_and_live_artifact_mapped" if require_live else status
        statuses.append(
            StandardStatus(
                key=standard.key,
                label=standard.label,
                status=status,
                missing_sources=missing_sources,
                missing_validators=missing_validators,
                missing_live_artifacts=missing_live,
                acceptance=standard.acceptance,
                reference_models=standard.reference_models,
            )
        )
    return statuses


def report(*, require_live: bool = False) -> dict[str, Any]:
    statuses = evaluate(require_live=require_live)
    complete = sum(1 for item in statuses if item.status != "gap")
    return {
        "standards": [asdict(item) for item in statuses],
        "summary": {
            "total": len(statuses),
            "mapped": complete,
            "gaps": len(statuses) - complete,
            "require_live": require_live,
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-live", action="store_true", help="require live proof artifacts")
    parser.add_argument("--strict", action="store_true", help="exit nonzero when any gap remains")
    parser.add_argument("--out", type=Path, help="optional JSON output path")
    args = parser.parse_args(argv)

    payload = report(require_live=args.require_live)
    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)

    if args.strict and payload["summary"]["gaps"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
