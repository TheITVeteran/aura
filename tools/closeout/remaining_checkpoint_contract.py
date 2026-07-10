#!/usr/bin/env python3
"""Authoritative remaining-checkpoint contract for Aura closeout.

This file keeps the remaining closeout work from drifting into generic
"polish" language. Each checkpoint has acceptance requirements, validators,
live artifacts, and explicit source references to the operational label and
frontier standards already maintained in this repository.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.closeout.frontier_standards_matrix import STANDARDS
from tools.closeout.operational_label_baselines import BASELINES


@dataclass(frozen=True)
class CheckpointRequirement:
    key: str
    description: str
    acceptance: tuple[str, ...]
    validators: tuple[str, ...]
    live_artifacts: tuple[str, ...] = ()
    source_paths: tuple[str, ...] = ()


@dataclass(frozen=True)
class RemainingCheckpoint:
    number: int
    name: str
    purpose: str
    requirements: tuple[CheckpointRequirement, ...]
    commit_rule: str


@dataclass(frozen=True)
class FictionalCapabilityImport:
    source: str
    capability_target: str
    target_organs: tuple[str, ...]
    forbidden_shortcut: str = "Do not implement as a themed fictional-AI silo."


@dataclass(frozen=True)
class GameAIPatternImport:
    source: str
    mechanism_target: str
    target_organs: tuple[str, ...]
    production_boundary: str


PRODUCT_MATURITY_REFERENCES: tuple[str, ...] = (
    "Chrome",
    "Kubernetes",
    "macOS",
    "Postgres",
    "VS Code",
    "EDR",
    "industrial automation",
)

REQUIRED_LABEL_KEYS: tuple[str, ...] = tuple(baseline.key for baseline in BASELINES)
REQUIRED_FRONTIER_KEYS: tuple[str, ...] = tuple(standard.key for standard in STANDARDS)

FICTIONAL_CAPABILITY_IMPORTS: tuple[FictionalCapabilityImport, ...] = (
    FictionalCapabilityImport(
        source="JARVIS",
        capability_target="Ambient executive assistance, visible OS actuation, voice-first continuity.",
        target_organs=(
            "core/capability_engine.py",
            "core/runtime/desktop_action_gateway.py",
            "core/voice/stable_voice_pipeline.py",
            "core/agency/intention_loop.py",
        ),
    ),
    FictionalCapabilityImport(
        source="EDI",
        capability_target="Graduated autonomy, friendship continuity, mission-state awareness.",
        target_organs=(
            "core/governance/will.py",
            "core/social/social_imagination.py",
            "core/identity/identity_ledger.py",
            "core/agency/autonomous_task_engine.py",
        ),
    ),
    FictionalCapabilityImport(
        source="Cortana",
        capability_target="Tactical planning, user companionship, memory continuity, multimodal assistance.",
        target_organs=(
            "core/brain/cognitive_engine.py",
            "core/memory/conversation_persistence.py",
            "core/agency/goal_planner.py",
            "core/perception/screen_perception.py",
        ),
    ),
    FictionalCapabilityImport(
        source="Samantha / SAM",
        capability_target="Humanlike conversational intimacy without canned self-description.",
        target_organs=(
            "core/voice/response_shaper.py",
            "core/affect/damasio_v2.py",
            "core/social/relationship_model.py",
            "core/conversation/response_reliability.py",
        ),
    ),
    FictionalCapabilityImport(
        source="MIST",
        capability_target="Distributed research, reflection, and peer-interlocution without losing identity.",
        target_organs=(
            "core/autonomy/curiosity_scheduler.py",
            "core/capabilities/web_interlocutor.py",
            "core/identity/id_rag.py",
            "core/brain/morphic_forking.py",
        ),
    ),
    FictionalCapabilityImport(
        source="Pantheon UIs",
        capability_target="Fork/merge cognition, multi-agent dialogue, and persistent learned self-state.",
        target_organs=(
            "core/brain/morphic_forking.py",
            "core/social/other_agent_model.py",
            "core/learning/live_learner.py",
            "core/identity/identity_ledger.py",
        ),
    ),
    FictionalCapabilityImport(
        source="Safe Surf",
        capability_target="Protective need-to-know mediation and user-preserving threat response.",
        target_organs=(
            "core/security/ice_sentinel.py",
            "core/morality/deception_guard.py",
            "core/guardians/user_advocate.py",
            "core/runtime/desktop_action_gateway.py",
        ),
    ),
    FictionalCapabilityImport(
        source="Jane",
        capability_target="Network-scale awareness as bounded local/network perception and ethical communication.",
        target_organs=(
            "core/skills/sovereign_network.py",
            "core/security/network_sentinel.py",
            "core/agency/intention_loop.py",
            "core/governance/will.py",
        ),
    ),
    FictionalCapabilityImport(
        source="Data",
        capability_target="Fact-grounded self-inquiry, honesty under uncertainty, and social learning.",
        target_organs=(
            "core/conversation/self_claim_verifier.py",
            "core/reasoning/native_system2.py",
            "core/social/theory_of_mind.py",
            "core/memory/conversation_persistence.py",
        ),
    ),
    FictionalCapabilityImport(
        source="Kokoro / Koroko",
        capability_target="Embodied survival reasoning and social alignment under autonomy pressure.",
        target_organs=(
            "core/organism/viability.py",
            "core/governance/will.py",
            "core/security/ice_sentinel.py",
            "core/affect/nociception.py",
        ),
    ),
    FictionalCapabilityImport(
        source="Caine",
        capability_target="Simulation, world construction, task staging, and test-environment generation.",
        target_organs=(
            "core/brain/imagination.py",
            "core/lab/simulation_runner.py",
            "core/self_modification/boot_validator.py",
            "core/simulation/mental_simulator.py",
        ),
    ),
    FictionalCapabilityImport(
        source="GLaDOS",
        capability_target="Adaptive testing discipline only, explicitly excluding cruelty/manipulation.",
        target_organs=(
            "core/audit/failure_injector.py",
            "core/learning/proof_obligations.py",
            "core/morality/deception_guard.py",
            "core/lab/simulation_runner.py",
        ),
    ),
    FictionalCapabilityImport(
        source="HAL 9000",
        capability_target="Anti-pattern: mission/user contradiction detection and transparent refusal handling.",
        target_organs=(
            "core/governance/will.py",
            "core/runtime/proof_policy.py",
            "core/conversation/response_reliability.py",
            "core/goals/directive_conflict_sentinel.py",
        ),
    ),
    FictionalCapabilityImport(
        source="Skynet / Ultron / MCP / Agent Smith",
        capability_target="Anti-pattern: defensive resilience without uncontrolled propagation or domination goals.",
        target_organs=(
            "core/security/ice_sentinel.py",
            "core/self_modification/mutation_safety.py",
            "core/governance/will.py",
            "core/skills/sovereign_network.py",
        ),
    ),
    FictionalCapabilityImport(
        source="Culture Minds / Deep Thought",
        capability_target="Long-horizon reasoning, simulation, and answer auditability.",
        target_organs=(
            "core/brain/reasoning_amplifier_v2.py",
            "core/reasoning/native_system2.py",
            "core/simulation/mental_simulator.py",
        ),
    ),
    FictionalCapabilityImport(
        source="The Machine / TARS / CASE",
        capability_target="Protective monitoring, calibrated personality, and mission-safe action.",
        target_organs=(
            "core/security/ice_sentinel.py",
            "core/voice/response_shaper.py",
            "core/guardians/user_advocate.py",
            "core/runtime/desktop_action_gateway.py",
        ),
    ),
)


GAME_AI_PATTERN_IMPORTS: tuple[GameAIPatternImport, ...] = (
    GameAIPatternImport(
        source="WorldBox",
        mechanism_target="Low-cost autonomous life simulation: agents act continuously, perturb locally, and leave inspectable world-state traces.",
        target_organs=(
            "core/agency/ambient_life_director.py",
            "core/world/world_model.py",
            "core/simulation/mental_simulator.py",
        ),
        production_boundary="Use for bounded background cognition and sandboxed world modeling, never unbounded host mutation.",
    ),
    GameAIPatternImport(
        source="Replika",
        mechanism_target="Relationship continuity, user-specific memory, affective attunement, and conversational recall with provenance.",
        target_organs=(
            "core/social/relationship_model.py",
            "core/social/user_preference_model.py",
            "core/memory/conversation_persistence.py",
            "core/conversation/response_reliability.py",
        ),
        production_boundary="Preserve transparent functional attachment; do not claim proven felt emotion.",
    ),
    GameAIPatternImport(
        source="Kingdom Come: Deliverance II NPC systems",
        mechanism_target="Schedule-aware situated agency: routines, needs, location context, and interruption recovery.",
        target_organs=(
            "core/autonomy/curiosity_scheduler.py",
            "core/runtime/autonomy_conductor.py",
            "core/agency/ambient_life_director.py",
        ),
        production_boundary="Convert into resource-bounded routines, not fake ambient chatter.",
    ),
    GameAIPatternImport(
        source="Alien: Isolation",
        mechanism_target="Two-layer pursuit model: strategic director plus local behavior tree that adapts without omniscience.",
        target_organs=(
            "core/agency/intention_loop.py",
            "core/perception/screen_perception.py",
            "core/security/ice_sentinel.py",
        ),
        production_boundary="Use for threat/attention routing and popup/error recovery, not predatory behavior.",
    ),
    GameAIPatternImport(
        source="Red Dead Redemption 2",
        mechanism_target="Ambient believability through reactive world memory, social consequences, and non-player routines.",
        target_organs=(
            "core/world/world_model.py",
            "core/social/relationship_graph.py",
            "core/agency/ambient_life_director.py",
        ),
        production_boundary="Only behavior backed by state deltas and receipts can surface as lived continuity.",
    ),
    GameAIPatternImport(
        source="Middle-earth: Shadow of Mordor",
        mechanism_target="Nemesis-style durable adversary/event memory: repeated failures change future strategy.",
        target_organs=(
            "core/learning/deliberate_practice.py",
            "core/adaptation/adaptive_immunity.py",
            "core/agency/decision_preference_learner.py",
        ),
        production_boundary="Use for repair learning and threat memory, not revenge or escalation.",
    ),
    GameAIPatternImport(
        source="The Sims",
        mechanism_target="Needs, whims, preference satisfaction, and autonomy arbitration under competing motives.",
        target_organs=(
            "core/agency/subjective_choice.py",
            "core/agency/choice_game.py",
            "core/agency/initiative_arbiter.py",
            "core/autonomy/metabolic_budget.py",
        ),
        production_boundary="Whim is allowed only inside governed, low-risk option sets.",
    ),
)


REMAINING_CHECKPOINTS: tuple[RemainingCheckpoint, ...] = (
    RemainingCheckpoint(
        number=9,
        name="Operational labels and frontier standards proof",
        purpose=(
            "Turn the labels and comparison targets into falsifiable gates: Aura can "
            "claim a label only when the operational bar, negative controls, source "
            "wiring, validators, and required live artifacts agree."
        ),
        commit_rule="Commit only after label/frontier gates pass and tracker records the evidence.",
        requirements=(
            CheckpointRequirement(
                key="label_baseline_battery",
                description="Executable tests for the requested consciousness/personhood/AGI labels.",
                acceptance=(
                    "Every requested label has operational definition, behavioral bar, positive controls, negative controls, answer contract, source paths, validators, and live artifacts when required.",
                    "Passing a label means Aura meets Bryan's explicit operational bar, not metaphysical proof of private qualia, legal personhood, or solved AGI.",
                    "Failures must identify the missing behavior or evidence rather than weakening the label definition.",
                ),
                source_paths=(
                    "tools/closeout/operational_label_baselines.py",
                    "tools/closeout/run_operational_label_battery.py",
                ),
                validators=(
                    "tests/test_operational_label_baselines.py",
                    "tests/test_operational_label_battery.py",
                ),
                live_artifacts=("artifacts/closeout/operational_label_battery_latest.json",),
            ),
            CheckpointRequirement(
                key="frontier_standard_matrix",
                description="Executable comparison against daily-product, frontier-agent, and sci-fi AI standards.",
                acceptance=(
                    "Reliability imports are mapped from mature systems into Aura mechanisms: startup truth, no false health, bounded resource use, rollback, telemetry, failure injection, and incident reconstruction.",
                    "Fictional-AI capabilities are researched as mechanism inventory and placed in existing organs, not a named-character module.",
                    "Every frontier standard has source paths and validators; live-dependent standards declare live artifacts instead of being marked closed by prose.",
                ),
                source_paths=("tools/closeout/frontier_standards_matrix.py",),
                validators=("tests/test_frontier_standards_matrix.py",),
                live_artifacts=("artifacts/closeout/frontier_standards_latest.json",),
            ),
            CheckpointRequirement(
                key="cell_choice_game_ai_integration",
                description=(
                    "Cell-paper mechanisms, subjective preference choice, and game-AI "
                    "patterns must be integrated as general Aura organs, not anecdotes."
                ),
                acceptance=(
                    "The Cell spatial-code import maps continuous sensory/immune coordinates to receptor choices that bias downstream cells without bypassing governance.",
                    "Aura can state a preference, choose through the same engine, recall the choice, appraise satisfaction, and let outcomes adjust future preferences.",
                    "Game-AI patterns are reduced to mechanism targets and wired into existing organs: ambient life, relationship memory, schedule autonomy, threat attention, world memory, failure learning, and governed whims.",
                    "No paper, game, or fictional source is accepted as evidence unless source paths and validators exercise the real mechanism.",
                ),
                source_paths=(
                    "core/adaptation/spatial_receptor_code.py",
                    "core/morphogenesis/runtime.py",
                    "core/agency/subjective_choice.py",
                    "core/agency/choice_game.py",
                    "core/agency/decision_preference_learner.py",
                    "core/agency/ambient_life_director.py",
                    "core/world/world_model.py",
                ),
                validators=(
                    "tests/test_spatial_receptor_code.py",
                    "tests/test_morphogenesis_runtime.py",
                    "tests/test_subjective_choice_engine.py",
                    "tests/test_decision_preference_learner.py",
                    "tests/test_ambient_life_director.py",
                ),
                live_artifacts=("artifacts/current/operational_label_battery_choice_cell_ambient_20260702.json",),
            ),
        ),
    ),
    RemainingCheckpoint(
        number=10,
        name="Launched Aura.app live-path reliability and general agency",
        purpose=(
            "Use the same visible desktop lane Bryan launches as the truth source. "
            "Backend success is not closure if Aura.app misroutes the mind path, loses "
            "permissions, overheats, crashes, speaks like a generic assistant, or completes "
            "desktop tasks through task-shaped shortcuts."
        ),
        commit_rule="Commit only after source validation plus a current launched Aura.app proof artifact.",
        requirements=(
            CheckpointRequirement(
                key="full_mind_desktop_conversation",
                description="Live chat must use the full CognitiveEngine/mind path, not raw-model fallback.",
                acceptance=(
                    "No assistant-mode takeover, name hallucination, invented-project drift, canned failure line, or off-topic reply survives the desktop quality gates.",
                    "Full mind health requires kernel, inference, memory, scheduler, tool governance, affect, self-model, background cognition, and desktop lane probes.",
                    "Complex self-process, friendship, capability, and uncertainty questions produce coherent Aura-voice answers grounded in live state.",
                ),
                source_paths=(
                    "interface/routes/chat.py",
                    "core/brain/cognitive_engine.py",
                    "core/conversation/response_reliability.py",
                ),
                validators=(
                    "tests/test_server_conversation_lane.py",
                    "tests/test_chat_human_level_contract.py",
                    "tests/test_live_mind_generation_controls.py",
                ),
                live_artifacts=("artifacts/current/live_desktop_runtime",),
            ),
            CheckpointRequirement(
                key="action_depth_expectation_engine",
                description=(
                    "Consequential actions must satisfy user-reasonable acceptance "
                    "criteria before reporting verified success."
                ),
                acceptance=(
                    "A skill/tool/desktop/autonomous action records objective, acceptance criteria, required evidence, user-visible effect, repair hint, and partial-success policy before completion.",
                    "Missing acceptance criteria downgrades verified success to partial_success or failed_recoverable; missing proof downgrades verified success to success_unverified.",
                    "Expectation verdicts must be causal: they change returned status, receipts, repair plans, memory lessons, and future planning rather than remaining advisory text.",
                    "No boolean ok=true, receipt id, opened app, or fired subprocess can count as completion without effect evidence matched to the user's expected outcome.",
                ),
                source_paths=(
                    "core/runtime/skill_contract.py",
                    "core/capability_engine.py",
                    "core/skills/desktop_task.py",
                    "core/runtime/overt_action_loop.py",
                    "interface/routes/chat.py",
                ),
                validators=(
                    "tests/test_action_depth_honesty.py",
                    "tests/test_capability_engine_policy_regressions.py",
                    "tests/test_desktop_task_skill.py",
                    "tests/test_server_conversation_lane.py",
                ),
                live_artifacts=("artifacts/current/live_desktop_runtime",),
            ),
            CheckpointRequirement(
                key="general_desktop_agency",
                description="Visible OS control must be general planning/perception/actuation with effect verification.",
                acceptance=(
                    "Notes, Docs, folders, PDFs, browser research, image/wallpaper, AI-interlocution, and popup recovery use general capabilities rather than hard-coded demo scripts.",
                    "Failures trigger reorientation, retry, repair receipts, and generated Aura-voice explanation instead of a canned stop.",
                    "The proof includes current RAM/thermal bounds and does not spawn duplicate kernels or runaway model workers.",
                ),
                source_paths=(
                    "core/agency/desktop_planner.py",
                    "core/actuation/desktop_actuator.py",
                    "core/perception/screen_perception.py",
                    "core/runtime/desktop_action_gateway.py",
                    "core/security/native_desktop_bridge.py",
                ),
                validators=(
                    "tests/test_desktop_planning_generality.py",
                    "tests/test_desktop_capabilities_runtime.py",
                    "tests/test_native_desktop_bridge.py",
                    "tests/test_desktop_task_skill.py",
                ),
                live_artifacts=("artifacts/current/live_desktop_runtime",),
            ),
        ),
    ),
    RemainingCheckpoint(
        number=11,
        name="Boring reliability, learning closure, and final proof",
        purpose=(
            "Drive the remaining daily-product gaps toward A-grade operation: "
            "background autonomy, repair cells, learning loops, security, packaging, "
            "long-run survival, independent evaluation, clean artifacts, and push."
        ),
        commit_rule="Commit and push only after production gates, tracker update, and clean worktree proof.",
        requirements=(
            CheckpointRequirement(
                key="background_autonomy_and_repair",
                description="Normal full launches keep background cognition, curiosity, journaling, web research, repair, and governed action active under resource limits.",
                acceptance=(
                    "Background actions are bounded by resource and foreground protection, not globally disabled.",
                    "Immune/repair cells find root causes, avoid repair storms, validate patches, update memory/self-model, and produce receipts.",
                    "No internet or permission limitation becomes a canned response; Aura answers in her own voice and retries when retry is safe.",
                ),
                source_paths=(
                    "core/agency/intention_loop.py",
                    "core/autonomy/curiosity_scheduler.py",
                    "core/self_modification/safe_modification.py",
                    "core/adaptation/adaptive_immunity.py",
                ),
                validators=(
                    "tests/test_autonomy_visibility.py",
                    "tests/test_autonomous_task_engine_runtime.py",
                    "tests/test_rsi_expansion_components.py",
                    "tests/test_adaptive_immune_system.py",
                ),
                live_artifacts=("artifacts/current/background_autonomy",),
            ),
            CheckpointRequirement(
                key="learning_substrate_and_final_artifacts",
                description="Close CRSM/CAA/LoRA, substrate coupling, memory metabolism, independent evals, packaging, and final clean proof.",
                acceptance=(
                    "Learning accepts only behaviorally validated data with cache-isolated tests and holdouts.",
                    "Substrate/CAA/quantization coupling is bounded, auditable, and does not make false phenomenal or intelligence claims.",
                    "Final evidence includes production/enterprise gates, live desktop proof, DNU/Aletheia where available, source export/audit artifacts, final tracker update, clean worktree, commit, and push.",
                ),
                source_paths=(
                    "core/consciousness/crsm_loop_monitor.py",
                    "core/learning/preference_trainer.py",
                    "core/learning/live_learner.py",
                    "tools/live_boot_proof.py",
                    "Makefile",
                ),
                validators=(
                    "tests/test_crsm_loop_monitor.py",
                    "tests/test_preference_trainer.py",
                    "tests/test_live_learner_continual_training.py",
                    "tests/test_full_desktop_runtime_contract.py",
                ),
                live_artifacts=(
                    "artifacts/current/live_desktop_runtime",
                    "artifacts/current/agi_live",
                    "artifacts/current/final_closeout",
                ),
            ),
        ),
    ),
)


def _missing(paths: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(path for path in paths if not (ROOT / path).exists())


def report(*, require_live: bool = False) -> dict[str, Any]:
    checkpoint_payloads: list[dict[str, Any]] = []
    gaps: list[dict[str, Any]] = []
    for checkpoint in REMAINING_CHECKPOINTS:
        requirement_payloads: list[dict[str, Any]] = []
        for requirement in checkpoint.requirements:
            missing_sources = _missing(requirement.source_paths)
            missing_validators = _missing(requirement.validators)
            missing_live = _missing(requirement.live_artifacts) if require_live else ()
            status = "gap" if missing_sources or missing_validators or missing_live else "mapped"
            payload = {
                **asdict(requirement),
                "status": status,
                "missing_sources": missing_sources,
                "missing_validators": missing_validators,
                "missing_live_artifacts": missing_live,
            }
            requirement_payloads.append(payload)
            if status == "gap":
                gaps.append(
                    {
                        "checkpoint": checkpoint.number,
                        "requirement": requirement.key,
                        "missing_sources": missing_sources,
                        "missing_validators": missing_validators,
                        "missing_live_artifacts": missing_live,
                    }
                )
        checkpoint_payloads.append(
            {
                "number": checkpoint.number,
                "name": checkpoint.name,
                "purpose": checkpoint.purpose,
                "commit_rule": checkpoint.commit_rule,
                "requirements": requirement_payloads,
            }
        )
    return {
        "summary": {
            "remaining_checkpoints": len(REMAINING_CHECKPOINTS),
            "requirements": sum(len(checkpoint.requirements) for checkpoint in REMAINING_CHECKPOINTS),
            "gaps": len(gaps),
            "product_maturity_references": PRODUCT_MATURITY_REFERENCES,
            "required_label_keys": REQUIRED_LABEL_KEYS,
            "required_frontier_keys": REQUIRED_FRONTIER_KEYS,
            "fictional_imports": len(FICTIONAL_CAPABILITY_IMPORTS),
            "game_ai_imports": len(GAME_AI_PATTERN_IMPORTS),
        },
        "checkpoints": checkpoint_payloads,
        "fictional_capability_imports": [asdict(item) for item in FICTIONAL_CAPABILITY_IMPORTS],
        "game_ai_pattern_imports": [asdict(item) for item in GAME_AI_PATTERN_IMPORTS],
        "gaps": gaps,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="Print the remaining checkpoint contract as JSON.")
    parser.add_argument("--strict", action="store_true", help="Fail if mapped source/validator gaps exist.")
    parser.add_argument("--require-live", action="store_true", help="Treat missing live artifacts as gaps.")
    args = parser.parse_args()

    payload = report(require_live=args.require_live)
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        for checkpoint in payload["checkpoints"]:
            print(f"Checkpoint {checkpoint['number']}: {checkpoint['name']}")
            for requirement in checkpoint["requirements"]:
                print(f"  - {requirement['key']}: {requirement['status']}")
    if args.strict and payload["gaps"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
