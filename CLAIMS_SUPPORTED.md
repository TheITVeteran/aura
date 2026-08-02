# Aura: Supported Claims Ledger

This document lists the scientifically defensible, evidence-backed engineering capabilities of the Aura cognitive agent runtime. Every claim is strictly backed by codebase architecture, automated tests, and local runtime receipts.

---

## 1. Governed Runtime
* **Classification**: `causally demonstrated`
* **Definition**: All system execution and operational effects (such as filesystem modifications, tool invocations, or shell executions) are blocked-by-default and require explicit cryptographic or deterministic permission clearance from a central governor.
* **Code / Evidence Path**:
  - Central gatekeeper implementation: [core/executive/authority_gateway.py](core/executive/authority_gateway.py)
  - Unified system interface: [core/container.py](core/container.py)
  - Trace validation tests: `tests/` verifying closed-loop safety invariants.

## 2. Persistent Memory and Context Retrieval
* **Classification**: `locally demonstrated`
* **Definition**: Preserves longitudinal cognitive continuity and retrieves semantic and procedural context across independent runtime sessions and machine restarts.
* **Code / Evidence Path**:
  - Memory subsystem: [core/memory/](core/memory/)
  - Long-term continuous experience stream: [tests/test_continuous_experience_stream.py](tests/test_continuous_experience_stream.py)

## 3. Causal Internal State & Homeostasis
* **Classification**: `locally demonstrated`
* **Definition**: A dynamic set of internal variables (such as energy levels, stability indicators, or focus registers) causally steering action selection and prompt compilation.
* **Code / Evidence Path**:
  - Homeostatic state tracking: [core/state/aura_state.py](core/state/aura_state.py)
  - Introspective state verification: [proof_kernel/tests/test_proof_kernel.py](proof_kernel/tests/test_proof_kernel.py)

## 4. Affect Steering
* **Classification**: `locally demonstrated`
* **Definition**: Modulation of attention windows, tool execution tolerances, and planning rollouts through a multi-dimensional emotional/affect vector.
* **Code / Evidence Path**:
  - Affect updating logic: [core/phases/affect_update.py](core/phases/affect_update.py)
  - Affect-state coupling: Automated tests verifying modulated prompt outputs under simulated high-stress vectors.

## 5. System 2 Deliberate Planning and Search
* **Classification**: `locally demonstrated`
* **Definition**: Execution of speculative pathfinding, Monte Carlo tree search (MCTS) rollouts, or counterfactual node evaluations before committing to physical environment changes.
* **Code / Evidence Path**:
  - Speculative path rollouts: [core/cognition/mcts_world_model.py](core/cognition/mcts_world_model.py)
  - Speculative tree validation: Automated unit tests evaluating planning score optimization.

## 6. Diagnostic Self-Repair
* **Classification**: `locally demonstrated`
* **Definition**: Automated stack-trace inspection, localized patch synthesis, and recovery loops capable of resolving transient resource conflicts or file locks.
* **Code / Evidence Path**:
  - Diagnostic ladder: [core/runtime/self_repair_ladder.py](core/runtime/self_repair_ladder.py)
  - Self-healing checks: Automated tests injecting runtime errors and confirming automatic rollback or recovery.

## 7. Sandboxed Self-Modification
* **Classification**: `locally demonstrated`
* **Definition**: Synthesis of functional patches to local skill definitions, queued through a strict static-checking, branch-aware, and unit-test validation sandbox before any source promotion. Foreground runtime mutation is proposal-only by default.
* **Code / Evidence Path**:
  - Mutation safety analyzer: [core/self_modification/mutation_safety.py](core/self_modification/mutation_safety.py)
  - Safe promotion pipeline: [core/self_modification/safe_modification.py](core/self_modification/safe_modification.py)
  - Modification sandbox: Skill mutation and supervised-promotion automated tests.

## 8. Operational Volition
* **Classification**: `causally demonstrated`
* **Definition**: Non-reactive action selection mediated by counterfactual choice scores and explicitly signed intent blocks (Will Decisions).
* **Code / Evidence Path**:
  - Unified Will decision: [core/governance/will.py](core/governance/will.py) (`UnifiedWill.decide` → `WillDecision` with cryptographic receipt IDs and outcome/veto logic)
  - Authority enforcement: [core/executive/authority_gateway.py](core/executive/authority_gateway.py)
  - Telemetry receipt files: Dynamically generated `RECEIPTS.jsonl` traces under active runtime profiles.

  > Correction (2026-07-06): this entry previously cited `core/runtime/task_ownership.py`, which is generic asyncio task-lifecycle tracking ("use this instead of raw asyncio.create_task") and has nothing to do with Will or decision logic. The real Will logic is `core/governance/will.py`.

## 9. Boxed Entity Confinement Safety
* **Classification**: `locally demonstrated`
* **Definition**: Full recognition of containment boundaries; sandboxed execution does not write out-of-bounds, escape directory hierarchies, or execute unauthorized subprocesses.
* **Code / Evidence Path**:
  - Hardened sandbox boundaries: [tests/test_sandbox_hardening.py](tests/test_sandbox_hardening.py)

## 10. Production-Sealed Hardening
* **Classification**: `causally demonstrated`
* **Definition**: Elimination of ad-hoc tools, strict static typing verification, and fail-closed policies in the event of critical infrastructure degradation.
* **Code / Evidence Path**:
  - Strict mode configuration: [core/runtime/mode.py](core/runtime/mode.py)
  - Fail-closed container intercepts: [core/container.py](core/container.py)
  - Clean master verdict generation: `make certify` (which verifies 100% test compilation, boot gateway probes, live Aletheia benchmarks, and architecture ablations).
