from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List

from core.autonomy.research_cycle import ResearchCycle
from core.ops.backup import BackupManager
from core.consciousness.subconscious_loop import SubconsciousLoop
from core.conversation.engine import CONTEXT_HISTORY_MAX_MESSAGES
from core.conversation import persistence as conversation_persistence
from core.conversation.unified_transcript import _MAX_HISTORY_DEFAULT as UNIFIED_TRANSCRIPT_MAX_HISTORY
from core.consciousness import authority_audit
from core.governance.will import UnifiedWill
from core.evolution import liquid_time_engine
from core.long_term_memory_engine import LongTermMemoryEngine
from core.memory import conversation_persistence as memory_conversation_persistence
from core.memory.episodic_memory import EpisodicMemory
from core.memory.retention_policy import (
    black_hole_retention_policy,
    hybrid_memory_retention_policy,
    training_buffer_retention_policy,
    working_history_retention_policy,
)
from core.memory.scar_formation import _MAX_SCARS
from core.resilience.incident_manager import IncidentManager
from core.phases import phi_consciousness
from core.resilience.lock_watchdog import get_lock_watchdog
from core.resilience.memory_governor import MemoryGovernor
from core.resilience.resource_governor import _LEDGER_PRUNE_DAYS
from core.resilience.stability_guardian import StabilityGuardian
from core.senses import circadian
from core.state.aura_state import MotivationState
from core.state.state_repository import StateRepository
from core.observability.unified_action_log import _MAX_ENTRIES as UNIFIED_ACTION_LOG_MAX_ENTRIES


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CRITICAL_SUPERVISION_FILES = {
    "belief_sync": PROJECT_ROOT / "core" / "collective" / "belief_sync.py",
    "proactive_perception": PROJECT_ROOT / "core" / "perception" / "proactive_perception_v2.py",
    "memory_subsystem": PROJECT_ROOT / "core" / "memory" / "memory_subsystem.py",
    "proactive_presence": PROJECT_ROOT / "core" / "autonomy" / "proactive_presence.py",
    "process_manager": PROJECT_ROOT / "core" / "ops" / "process_manager.py",
    "autonomic_core": PROJECT_ROOT / "core" / "autonomic" / "core_monitor.py",
    "concept_linker": PROJECT_ROOT / "core" / "knowledge" / "concept_linker.py",
    "intent_gate": PROJECT_ROOT / "core" / "intent" / "intent_gate.py",
    "narrative_thread": PROJECT_ROOT / "core" / "identity" / "narrative_thread.py",
    "memory_synthesizer": PROJECT_ROOT / "core" / "memory" / "memory_synthesizer.py",
    "server": PROJECT_ROOT / "interface" / "server.py",
    "boot_background": PROJECT_ROOT / "core" / "orchestrator" / "mixins" / "boot" / "boot_background.py",
    "boot_autonomy": PROJECT_ROOT / "core" / "orchestrator" / "mixins" / "boot" / "boot_autonomy.py",
}


@dataclass(frozen=True)
class KnownIssue:
    issue_id: str
    severity: str
    title: str
    status: str
    evidence: str
    recommendation: str


@dataclass(frozen=True)
class RepairCapability:
    subsystem: str
    detects: bool
    auto_recovers: bool
    notes: str


@dataclass
class RuntimeRegistry:
    hardware_model: str
    total_ram_gb: float
    total_storage_gb: float
    simulation_step_s: int
    liquid_time_clamp_s: float
    circadian_update_interval_s: float
    circadian_stale_threshold_s: float
    motivation_kernel_dt_cap_s: float
    motivation_budgets: Dict[str, Dict[str, float]]
    motivation_social_recovery_per_min: float
    research_cycle_interval_s: float
    research_idle_threshold_s: float
    research_goal_timeout_s: float
    subconscious_idle_threshold_s: float
    subconscious_dream_interval_s: float
    subconscious_sandbox_interval_s: float
    episodic_eval_interval_s: float
    episodic_max_episodes: int
    episodic_retention_basis: str
    black_hole_max_memories: int
    black_hole_prune_keep_fraction: float
    black_hole_retention_basis: str
    hybrid_memory_max_entries: int
    hybrid_memory_retention_basis: str
    ltm_consolidation_interval_s: float
    ltm_rehearsal_min_age_s: float
    ltm_max_memories: int
    ltm_retention_basis: str
    conversation_retention_days: int
    conversation_prune_interval_s: float
    conversation_prune_scheduled: bool
    unified_transcript_max_history: int
    conversation_context_max_messages: int
    conversation_cache_max_messages: int
    conversation_cache_max_sessions: int
    behavioral_scar_max_records: int
    executive_decision_history_max: int
    executive_decision_history_basis: str
    cognitive_thought_history_max: int
    cognitive_thought_history_basis: str
    learning_execution_history_max: int
    learning_execution_history_basis: str
    metacognitive_reasoning_history_max: int
    metacognitive_reasoning_history_basis: str
    omni_reflector_history_max: int
    omni_reflector_history_basis: str
    global_workspace_history_max: int
    global_workspace_history_basis: str
    live_learner_buffer_max_examples: int
    live_learner_buffer_basis: str
    unified_action_log_max_entries: int
    unified_will_audit_max_entries: int
    authority_audit_max_entries: int
    incident_history_max_entries: int
    pending_initiative_cap: int
    active_goal_cap: int
    vector_prune_interval_s: float
    vector_prune_threshold_days: int
    vector_soft_prune_threshold_days: int
    cognitive_ledger_prune_days: int
    memory_governor_check_interval_s: float
    memory_governor_threshold_prune_mb: float
    memory_governor_threshold_unload_mb: float
    memory_governor_threshold_critical_mb: float
    memory_governor_prune_cooldown_s: float
    memory_governor_unload_cooldown_s: float
    memory_governor_critical_cooldown_s: float
    memory_governor_prune_hysteresis_mb: float
    memory_governor_unload_hysteresis_mb: float
    backup_vacuum_interval_s: float
    backup_interval_s: float
    backup_max_backups: int
    backup_wired: bool
    database_coordinator_wired: bool
    lock_watchdog_threshold_s: float
    lock_watchdog_auto_repair: bool
    state_queue_repair_enabled: bool
    state_commit_queue_maxsize: int
    state_log_max_rows: int
    state_log_retention_basis: str
    stability_max_task_count: int
    stability_max_tick_lag_ms: float
    stability_memory_warning_pct: float
    stability_memory_critical_pct: float
    stability_repair_uses_task_cancellation: bool
    phi_dormant: float
    phi_reactive: float
    phi_deliberate: float
    phi_ignition: float
    dual_writer_guard_active: bool
    source_signature: Dict[str, Any] = field(default_factory=dict)
    known_issues: List[KnownIssue] = field(default_factory=list)
    repair_capabilities: List[RepairCapability] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["known_issues"] = [asdict(item) for item in self.known_issues]
        payload["repair_capabilities"] = [asdict(item) for item in self.repair_capabilities]
        return payload


def _read_source(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _extract_literal(path: Path, pattern: str, cast=float, default: Any = None) -> Any:
    match = re.search(pattern, _read_source(path), re.MULTILINE)
    if not match:
        return default
    return cast(match.group(1))


def _sha256_dict(payload: Dict[str, Any]) -> str:
    encoded = repr(sorted(payload.items())).encode("utf-8", errors="ignore")
    return hashlib.sha256(encoded).hexdigest()


def _critical_supervision_audit() -> Dict[str, Any]:
    audited: Dict[str, bool] = {}
    unresolved: List[str] = []
    for name, path in CRITICAL_SUPERVISION_FILES.items():
        source = _read_source(path)
        supervised = (
            "get_task_tracker" in source
            or "TaskTracker(" in source
            or "create_tracked_task(" in source  # task_ownership primitive
            or "tracker.create_task(" in source
            or "tracker.bounded_track(" in source
            or "_spawn_server_task(" in source
            or "_spawn_server_bounded_task(" in source
        )
        audited[name] = supervised
        if not supervised:
            unresolved.append(name)
    return {
        "audited": audited,
        "unresolved": unresolved,
        "all_resolved": not unresolved,
    }


def build_registry() -> RuntimeRegistry:
    motivation = MotivationState()
    memory_governor = MemoryGovernor(SimpleNamespace(memory_manager=None))
    black_hole_policy = black_hole_retention_policy(ram_gb=64.0)
    hybrid_memory_policy = hybrid_memory_retention_policy(ram_gb=64.0)
    executive_history_policy = working_history_retention_policy(
        "AURA_EXECUTIVE_DECISION_HISTORY_MAX",
        ram_gb=64.0,
    )
    cognitive_history_policy = working_history_retention_policy(
        "AURA_COGNITIVE_THOUGHT_HISTORY_MAX",
        ram_gb=64.0,
    )
    learning_execution_policy = working_history_retention_policy(
        "AURA_LEARNING_EXECUTION_HISTORY_MAX",
        ram_gb=64.0,
    )
    metacognitive_history_policy = working_history_retention_policy(
        "AURA_METACOGNITIVE_REASONING_HISTORY_MAX",
        ram_gb=64.0,
    )
    omni_reflector_policy = working_history_retention_policy(
        "AURA_OMNI_REFLECTOR_HISTORY_MAX",
        ram_gb=64.0,
    )
    global_workspace_policy = working_history_retention_policy(
        "AURA_GLOBAL_WORKSPACE_HISTORY_MAX",
        ram_gb=64.0,
    )
    live_learner_buffer_policy = training_buffer_retention_policy(ram_gb=64.0)
    backup_manager = BackupManager()
    ltm_engine = LongTermMemoryEngine()
    repo = StateRepository(db_path=":memory:")
    lock_watchdog = get_lock_watchdog()

    continuity_path = PROJECT_ROOT / "core" / "continuity.py"
    subconscious_path = PROJECT_ROOT / "core" / "consciousness" / "subconscious_loop.py"
    motivation_path = PROJECT_ROOT / "core" / "phases" / "motivation_update.py"
    episodic_path = PROJECT_ROOT / "core" / "memory" / "episodic_memory.py"
    will_engine_path = PROJECT_ROOT / "core" / "self" / "will_engine.py"
    lock_watchdog_path = PROJECT_ROOT / "core" / "resilience" / "lock_watchdog.py"
    core_baseline_path = PROJECT_ROOT / "core" / "orchestrator" / "initializers" / "core_baseline.py"
    boot_path = PROJECT_ROOT / "core" / "orchestrator" / "boot.py"
    orchestrator_main_path = PROJECT_ROOT / "core" / "orchestrator" / "main.py"
    state_repository_path = PROJECT_ROOT / "core" / "state" / "state_repository.py"
    stability_path = PROJECT_ROOT / "core" / "resilience" / "stability_guardian.py"
    conversation_path = PROJECT_ROOT / "core" / "conversation" / "persistence.py"

    motivation_source = _read_source(motivation_path)
    will_engine_source = _read_source(will_engine_path)
    lock_watchdog_source = _read_source(lock_watchdog_path)
    core_baseline_source = _read_source(core_baseline_path)
    boot_source = _read_source(boot_path)
    state_repo_source = _read_source(state_repository_path)
    stability_source = _read_source(stability_path)
    conversation_source = _read_source(conversation_path)

    gap_divisor = _extract_literal(continuity_path, r"gap_factor = _clamp01\(gap_seconds / ([0-9.]+)\)", float, 21600.0)
    reentry_gap_trigger = _extract_literal(continuity_path, r"gap_seconds\", 0\.0\) or 0\.0\) >= ([0-9.]+)", float, 900.0)
    subconscious_dream_interval = _extract_literal(subconscious_path, r"last_dream_cycle > ([0-9.]+)", float, 300.0)
    subconscious_sandbox_interval = _extract_literal(subconscious_path, r"last_sandbox_experiment > ([0-9.]+)", float, 900.0)
    motivation_dt_cap = _extract_literal(motivation_path, r"if dt > ([0-9.]+): dt = \1", float, 300.0)
    episodic_eval_interval = _extract_literal(episodic_path, r"next_eval = now \+ ([0-9]+)", float, 21600.0)
    vector_soft_prune_days = _extract_literal(orchestrator_main_path, r"threshold_days=([0-9]+)", int, 14)

    dual_writer_guard_active = "legacy_metabolism_active and name in {\"energy\", \"curiosity\"}" in motivation_source
    will_engine_still_mutates_curiosity = "motivation.budgets.setdefault(\"curiosity\"" in will_engine_source
    backup_wired = (
        "BackupManager" in core_baseline_source
        # Registration may go through the container or the registry seam.
        and (
            'register_instance("backup_manager"' in core_baseline_source
            or 'register_runtime_service("backup_manager"' in core_baseline_source
        )
        and "await orchestrator.backup_manager.on_start_async()" in core_baseline_source
    )
    database_coordinator_wired = 'register_instance("database_coordinator"' in boot_source
    conversation_prune_scheduled = (
        "periodic_conversation_prune" in conversation_source
        and "await scheduler.register" in conversation_source
    )
    lock_watchdog_auto_repair = "_attempt_recovery" in lock_watchdog_source and "on_stall" in lock_watchdog_source
    state_queue_repair_enabled = "async def repair_runtime" in state_repo_source and "coalesced_queue" in state_repo_source
    stability_repair_uses_task_cancellation = "anonymous unsupervised tasks" in stability_source
    supervision_audit = _critical_supervision_audit()

    source_signature = {
        "continuity_gap_divisor": gap_divisor,
        "continuity_reentry_gap_trigger": reentry_gap_trigger,
        "subconscious_dream_interval_s": subconscious_dream_interval,
        "subconscious_sandbox_interval_s": subconscious_sandbox_interval,
        "motivation_kernel_dt_cap_s": motivation_dt_cap,
        "episodic_eval_interval_s": episodic_eval_interval,
        "ltm_consolidation_interval_s": ltm_engine.consolidation_interval_s,
        "conversation_prune_interval_s": conversation_persistence.DEFAULT_CONVERSATION_PRUNE_INTERVAL_S,
        "vector_prune_interval_s": memory_governor.vector_prune_interval_s,
        "vector_soft_prune_threshold_days": vector_soft_prune_days,
        "cognitive_ledger_prune_days": _LEDGER_PRUNE_DAYS,
        "memory_governor_prune_cooldown_s": memory_governor.prune_cooldown_s,
        "memory_governor_unload_cooldown_s": memory_governor.unload_cooldown_s,
        "memory_governor_critical_cooldown_s": memory_governor.critical_cooldown_s,
        "memory_governor_prune_hysteresis_mb": memory_governor.prune_hysteresis_mb,
        "memory_governor_unload_hysteresis_mb": memory_governor.unload_hysteresis_mb,
        "backup_interval_s": backup_manager.backup_interval_s,
        "backup_vacuum_interval_s": backup_manager.vacuum_interval_s,
        "backup_wired": backup_wired,
        "database_coordinator_wired": database_coordinator_wired,
        "lock_watchdog_threshold_s": getattr(lock_watchdog, "_threshold", 180.0),
        "lock_watchdog_auto_repair": lock_watchdog_auto_repair,
        "state_commit_queue_maxsize": repo._mutation_queue_maxsize,
        "state_queue_repair_enabled": state_queue_repair_enabled,
        "critical_supervision_audit": supervision_audit,
    }

    known_issues: List[KnownIssue] = []
    if ltm_engine.consolidation_interval_s < 3600:
        known_issues.append(KnownIssue(
            issue_id="ltm_dev_cadence",
            severity="critical",
            title="Long-term memory consolidation still on development cadence",
            status="active",
            evidence=f"consolidation interval is {ltm_engine.consolidation_interval_s:.0f}s",
            recommendation="Raise to nightly or configure via environment/config.",
        ))
    else:
        known_issues.append(KnownIssue(
            issue_id="ltm_dev_cadence",
            severity="info",
            title="Long-term memory consolidation cadence normalized",
            status="resolved",
            evidence=f"consolidation interval is {ltm_engine.consolidation_interval_s:.0f}s",
            recommendation="Keep the interval environment/config-driven for dev overrides only.",
        ))

    if not dual_writer_guard_active or not will_engine_still_mutates_curiosity:
        known_issues.append(KnownIssue(
            issue_id="motivation_dual_writer",
            severity="warning",
            title="Motivation dual-writer hazard may still be active",
            status="active",
            evidence="kernel and legacy metabolic paths both touch motivation budgets",
            recommendation="Ensure only one path owns energy/curiosity budget mutation.",
        ))
    else:
        known_issues.append(KnownIssue(
            issue_id="motivation_dual_writer",
            severity="info",
            title="Motivation dual-writer hazard guarded",
            status="resolved",
            evidence="kernel phase now skips energy/curiosity budgets when WillEngine is active",
            recommendation="Keep the ownership split explicit in tests and registry output.",
        ))

    known_issues.append(KnownIssue(
        issue_id="backup_wiring",
        severity="warning" if not backup_wired else "info",
        title="Automatic backup scheduler wiring",
        status="active" if not backup_wired else "resolved",
        evidence="BackupManager boot registration was audited directly from the orchestrator baseline.",
        recommendation="Keep automatic backup registration in the boot path so 24h+ runtimes always get recoverable checkpoints.",
    ))

    known_issues.append(KnownIssue(
        issue_id="state_queue_repair",
        severity="warning" if not state_queue_repair_enabled else "info",
        title="State mutation queue self-repair coverage",
        status="active" if not state_queue_repair_enabled else "resolved",
        evidence="StateRepository was inspected for consumer restart and queue coalescing repair hooks.",
        recommendation="Keep the state-vault repair path active and check-covered so queue saturation does not silently accumulate.",
    ))

    known_issues.append(KnownIssue(
        issue_id="lock_watchdog_repair",
        severity="warning" if not lock_watchdog_auto_repair else "info",
        title="Lock watchdog recovery posture",
        status="active" if not lock_watchdog_auto_repair else "resolved",
        evidence="LockWatchdog source was checked for recovery callbacks and intervention path.",
        recommendation="Retain safe forced-release callbacks for watchdog-managed robust locks so deadlocks are repaired, not just reported.",
    ))

    if stability_repair_uses_task_cancellation and not supervision_audit["all_resolved"]:
        known_issues.append(KnownIssue(
            issue_id="task_explosion_root_fix_gap",
            severity="warning",
            title="Task explosion mitigation still partly cancellation-based",
            status="active",
            evidence=(
                "StabilityGuardian still has a fallback path that cancels anonymous unsupervised tasks under pressure; "
                f"critical supervision audit unresolved={supervision_audit['unresolved']}"
            ),
            recommendation="Continue replacing unsupervised spawn sites with bounded tracked loops so cancellation becomes a rare last resort.",
        ))
    else:
        known_issues.append(KnownIssue(
            issue_id="task_explosion_root_fix_gap",
            severity="info",
            title="Critical long-lived task supervision audit",
            status="resolved",
            evidence="The audited long-lived loop surfaces now route through supervised task ownership.",
            recommendation="Keep the critical supervision audit current as new perpetual loops are added.",
        ))

    repair_capabilities = [
        RepairCapability(
            subsystem="memory_governor",
            detects=True,
            auto_recovers=True,
            notes="Can unload models, prune memory, vacuum databases, and force a metabolic rest under pressure.",
        ),
        RepairCapability(
            subsystem="stability_guardian",
            detects=True,
            auto_recovers=True,
            notes="Can trigger GC, episodic eviction, state-vault repair, backup refresh, and anonymous-task shedding under pressure.",
        ),
        RepairCapability(
            subsystem="resource_governor",
            detects=True,
            auto_recovers=True,
            notes="Trims capped collections, compacts ledgers, and cleans completed background tasks.",
        ),
        RepairCapability(
            subsystem="state_repository",
            detects=True,
            auto_recovers=state_queue_repair_enabled,
            notes="Bounded queue now exposes runtime status and can restart the mutation consumer or coalesce pressure when needed.",
        ),
        RepairCapability(
            subsystem="lock_watchdog",
            detects=True,
            auto_recovers=lock_watchdog_auto_repair,
            notes="Tracks held-lock age and can now run a lock-specific repair callback when a robust lock stalls beyond threshold.",
        ),
        RepairCapability(
            subsystem="backup_manager",
            detects=True,
            auto_recovers=backup_wired,
            notes="Runs scheduled vacuum and backups, and can opportunistically create a fresh backup when health checks detect drift.",
        ),
    ]

    registry = RuntimeRegistry(
        hardware_model="Apple Silicon M5 MacBook Pro",
        total_ram_gb=64.0,
        total_storage_gb=2000.0,
        simulation_step_s=300,
        liquid_time_clamp_s=float(liquid_time_engine.MAX_SLEEP_WAKE_DT_S),
        circadian_update_interval_s=float(circadian.UPDATE_INTERVAL),
        circadian_stale_threshold_s=float(circadian.STALE_THRESHOLD),
        motivation_kernel_dt_cap_s=float(motivation_dt_cap),
        motivation_budgets={key: dict(value) for key, value in motivation.budgets.items()},
        motivation_social_recovery_per_min=0.5,
        research_cycle_interval_s=float(ResearchCycle.MIN_CYCLE_INTERVAL_S),
        research_idle_threshold_s=float(ResearchCycle.IDLE_THRESHOLD_S),
        research_goal_timeout_s=float(ResearchCycle.MAX_GOAL_DURATION_S),
        subconscious_idle_threshold_s=float(SubconsciousLoop(None).idle_threshold),
        subconscious_dream_interval_s=float(subconscious_dream_interval),
        subconscious_sandbox_interval_s=float(subconscious_sandbox_interval),
        episodic_eval_interval_s=float(episodic_eval_interval),
        episodic_max_episodes=int(EpisodicMemory.MAX_EPISODES),
        episodic_retention_basis=str(getattr(EpisodicMemory, "RETENTION_POLICY").basis),
        black_hole_max_memories=int(black_hole_policy.max_items),
        black_hole_prune_keep_fraction=float(black_hole_policy.prune_keep_fraction),
        black_hole_retention_basis=str(black_hole_policy.basis),
        hybrid_memory_max_entries=int(hybrid_memory_policy.max_items),
        hybrid_memory_retention_basis=str(hybrid_memory_policy.basis),
        ltm_consolidation_interval_s=float(ltm_engine.consolidation_interval_s),
        ltm_rehearsal_min_age_s=float(ltm_engine.rehearsal_min_age_s),
        ltm_max_memories=int(ltm_engine._max_memories),
        ltm_retention_basis=str(ltm_engine._policy().basis),
        conversation_retention_days=int(conversation_persistence.DEFAULT_CONVERSATION_RETENTION_DAYS),
        conversation_prune_interval_s=float(conversation_persistence.DEFAULT_CONVERSATION_PRUNE_INTERVAL_S),
        conversation_prune_scheduled=bool(conversation_prune_scheduled),
        unified_transcript_max_history=int(UNIFIED_TRANSCRIPT_MAX_HISTORY),
        conversation_context_max_messages=int(CONTEXT_HISTORY_MAX_MESSAGES),
        conversation_cache_max_messages=int(memory_conversation_persistence.MAX_HISTORY_IN_MEMORY),
        conversation_cache_max_sessions=int(memory_conversation_persistence.MAX_SESSIONS_ON_DISK),
        behavioral_scar_max_records=int(_MAX_SCARS),
        executive_decision_history_max=int(executive_history_policy.max_items),
        executive_decision_history_basis=str(executive_history_policy.basis),
        cognitive_thought_history_max=int(cognitive_history_policy.max_items),
        cognitive_thought_history_basis=str(cognitive_history_policy.basis),
        learning_execution_history_max=int(learning_execution_policy.max_items),
        learning_execution_history_basis=str(learning_execution_policy.basis),
        metacognitive_reasoning_history_max=int(metacognitive_history_policy.max_items),
        metacognitive_reasoning_history_basis=str(metacognitive_history_policy.basis),
        omni_reflector_history_max=int(omni_reflector_policy.max_items),
        omni_reflector_history_basis=str(omni_reflector_policy.basis),
        global_workspace_history_max=int(global_workspace_policy.max_items),
        global_workspace_history_basis=str(global_workspace_policy.basis),
        live_learner_buffer_max_examples=int(live_learner_buffer_policy.max_items),
        live_learner_buffer_basis=str(live_learner_buffer_policy.basis),
        unified_action_log_max_entries=int(UNIFIED_ACTION_LOG_MAX_ENTRIES),
        unified_will_audit_max_entries=int(UnifiedWill._MAX_AUDIT_TRAIL),
        authority_audit_max_entries=int(authority_audit._MAX_ENTRIES),
        incident_history_max_entries=int(IncidentManager.MAX_HISTORY),
        pending_initiative_cap=10,
        active_goal_cap=10,
        vector_prune_interval_s=float(memory_governor.vector_prune_interval_s),
        vector_prune_threshold_days=30,
        vector_soft_prune_threshold_days=int(vector_soft_prune_days),
        cognitive_ledger_prune_days=int(_LEDGER_PRUNE_DAYS),
        memory_governor_check_interval_s=float(memory_governor.check_interval),
        memory_governor_threshold_prune_mb=float(memory_governor.threshold_prune),
        memory_governor_threshold_unload_mb=float(memory_governor.threshold_unload),
        memory_governor_threshold_critical_mb=float(memory_governor.threshold_critical),
        memory_governor_prune_cooldown_s=float(memory_governor.prune_cooldown_s),
        memory_governor_unload_cooldown_s=float(memory_governor.unload_cooldown_s),
        memory_governor_critical_cooldown_s=float(memory_governor.critical_cooldown_s),
        memory_governor_prune_hysteresis_mb=float(memory_governor.prune_hysteresis_mb),
        memory_governor_unload_hysteresis_mb=float(memory_governor.unload_hysteresis_mb),
        backup_vacuum_interval_s=float(backup_manager.vacuum_interval_s),
        backup_interval_s=float(backup_manager.backup_interval_s),
        backup_max_backups=int(backup_manager.max_backups),
        backup_wired=bool(backup_wired),
        database_coordinator_wired=bool(database_coordinator_wired),
        lock_watchdog_threshold_s=float(getattr(lock_watchdog, "_threshold", 180.0)),
        lock_watchdog_auto_repair=bool(lock_watchdog_auto_repair),
        state_queue_repair_enabled=bool(state_queue_repair_enabled),
        state_commit_queue_maxsize=int(repo._mutation_queue_maxsize),
        state_log_max_rows=int(StateRepository.STATE_LOG_MAX_ROWS),
        state_log_retention_basis=str(StateRepository.STATE_LOG_RETENTION_BASIS),
        stability_max_task_count=int(StabilityGuardian.MAX_TASK_COUNT),
        stability_max_tick_lag_ms=float(StabilityGuardian.MAX_TICK_LAG_MS),
        stability_memory_warning_pct=float(StabilityGuardian.MEMORY_WARNING_PCT),
        stability_memory_critical_pct=float(StabilityGuardian.MEMORY_CRITICAL_PCT),
        stability_repair_uses_task_cancellation=bool(stability_repair_uses_task_cancellation),
        phi_dormant=float(phi_consciousness.PHI_DORMANT),
        phi_reactive=float(phi_consciousness.PHI_REACTIVE),
        phi_deliberate=float(phi_consciousness.PHI_DELIBERATE),
        phi_ignition=float(phi_consciousness.PHI_IGNITION),
        dual_writer_guard_active=bool(dual_writer_guard_active),
        source_signature={**source_signature, "hash": _sha256_dict(source_signature)},
        known_issues=known_issues,
        repair_capabilities=repair_capabilities,
    )
    return registry
