# Aura Claims Matrix

This document defines the formal claims matrix for the Aura cognitive agent runtime. Every claim is strictly classified based on empirical, local, or external validation evidence. Unsupported claims are explicitly demoted to "not proven" or "deprecated/retired".

## Claims Classification Summary

| Claim | Classification | Evidence Path / Blocker |
| :--- | :--- | :--- |
| **1. Governed Runtime** | `causally demonstrated` | `core/executive/authority_gateway.py`, strict flagship readiness, receipt coverage validator |
| **2. Persistent Memory** | `locally demonstrated` | `core/memory/`, continuous experience stream tests, sqlite/vector db integration tests |
| **3. Causal Internal State** | `locally demonstrated` | `core/state/aura_state.py`, homeostatic state tracking, affect behaviour coupling tests |
| **4. Affect Steering** | `locally demonstrated` | `core/phases/affect_update.py`, affect-state coupling and prompt modulation tests |
| **5. System 2 Planning/Search** | `locally demonstrated` | `core/cognition/mcts_world_model.py`, tree planning and counterfactual search tests |
| **6. Self-Repair** | `locally demonstrated` | `core/runtime/self_repair_ladder.py`, diagnostic self-healing loops |
| **7. Self-Modification** | `locally demonstrated` | `core/self_modification/mutation_safety.py`, sandboxed patch proposals |
| **8. Operational Volition** | `causally demonstrated` | `core/runtime/task_ownership.py`, `UnifiedWill` decision logs in `RECEIPTS.jsonl` |
| **9. Autonomous Agency** | `locally demonstrated` | `core/autonomy/autonomous_research_orchestrator.py`, multi-step goal decomposition |
| **10. Emergent Intelligence** | `not proven` | Blocker: Requires large-scale out-of-distribution model evaluations beyond local compute limits |
| **11. Entity-in-a-Box Behavior** | `locally demonstrated` | `tests/test_sandbox_hardening.py`, confinement boundary recognition tests |
| **12. External Real-World Validation** | `not proven` | Blocker: Requires independent, external third-party evaluation and live production network |
| **13. DNU AGI** | `not proven` | `artifacts/current/agi_live/` passed the configured local 100-task battery, but AGI itself remains unproven |
| **14. AGI-Candidate** | `locally demonstrated` | `artifacts/current/agi_live/`, `artifacts/current/external_live_validation/`, `artifacts/current/agency_emergence_boxed_entity/`, `artifacts/current/unified_system_scenario/`, receipt coverage, ablations, baselines, and Aletheia Tier 5 evidence |
| **15. Production-Sealed** | `causally demonstrated` | Pass status of flagship readiness, production surface lint, and artifact consistency |
| **16. Mature RSI** | `not proven` | Blocker: Safety filters and local model capability restrict autonomous non-interactive improvement |
| **17. Subjective Consciousness** | `not proven` | Strictly unsupported. Qualitative experience, qualia, and personhood are not scientifically provable |
| **18. Personhood** | `not proven` | Strictly unsupported. Aura is a software runtime, not a legal or moral person |
| **19. Metaphysical Free Will** | `not proven` | Strictly unsupported. Aura operates on deterministic/probabilistic computational volition only |
| **20. Indefinite Autonomy** | `not proven` | Blocker: Bounded by short proof longevity soak limits (needs 72h+ soak runs) |
| **21. Synthetic Cognitive Entity** | `locally demonstrated` | Boxed agency, operational volition, unified scenario, memory continuity, and receipt coverage pass under the configured local profile |
| **22. Experience-Adjacent Indicators**| `locally demonstrated` | Introspective state tracking, affect-memory interaction, and self-report checks |

---

## Detailed Claims Definitions

### 1. Governed Runtime
* **Classification**: `causally demonstrated`
* **Definition**: All operational effects (file writes, tools, LLM calls) must pass through a closed-by-default, receipt-generating gatekeeper.
* **Evidence**: Authority Gateway `core/executive/authority_gateway.py` intercepts effects. Checked-in test traces in `tests/` verify closed-loop failures on unauthorized calls.

### 2. Persistent Memory
* **Classification**: `locally demonstrated`
* **Definition**: System preserves historical continuity and retrieves context across independent cognitive runtime sessions.
* **Evidence**: Checked-in SQLite/vector db integration tests and `tests/test_continuous_experience_stream.py`.

### 3. Causal Internal State
* **Classification**: `locally demonstrated`
* **Definition**: Internal variables (such as homeostatic markers or active focus parameters) causally steer the agent's behavior.
* **Evidence**: State updates steer LLM prompt assembly and action selections, verified by homeostatic tests in `proof_kernel/tests/test_proof_kernel.py`.

### 4. Affect Steering
* **Classification**: `locally demonstrated`
* **Definition**: A structured affect vector modulates sensory and planning processes.
* **Evidence**: Checked-in `core/phases/affect_update.py` and affect-state coupling tests.

### 5. System 2 Planning/Search
* **Classification**: `locally demonstrated`
* **Definition**: Deliberate tree search or Monte Carlo rollout is executed to formulate plan paths.
* **Evidence**: `core/cognition/mcts_world_model.py` and speculative path validation tests.

### 6. Self-Repair
* **Classification**: `locally demonstrated`
* **Definition**: The runtime detects exceptions, assesses the stack, and autonomously rolls back or self-heals transient files.
* **Evidence**: `core/runtime/self_repair_ladder.py` and self-healing tests.

### 7. Self-Modification
* **Classification**: `locally demonstrated`
* **Definition**: Aura can propose syntactically valid patches to its own skills, sandboxed and statically checked before application.
* **Evidence**: `core/self_modification/mutation_safety.py` and safe modification harness.

### 8. Operational Volition
* **Classification**: `causally demonstrated`
* **Definition**: Action selections are mediated by counterfactual choice evaluations and Will Decisions signed by the Unified Will.
* **Evidence**: UnifiedWill decision receipts generated dynamically during test runs and logged to `RECEIPTS.jsonl`.

### 9. Autonomous Agency
* **Classification**: `locally demonstrated`
* **Definition**: Aura pursues high-level objectives over multiple steps, dynamically adjusting subgoals based on environment feedback.
* **Evidence**: Goal ledger tracking in `core/autonomy/autonomous_research_orchestrator.py` and autonomous research tests.

### 10. Emergent Intelligence
* **Classification**: `not proven`
* **Definition**: Complex multi-step reasoning outperforming simple prompt templates by significant margins under strict control.
* **Blocker**: Lack of high-capacity local compute or unmetered cloud model APIs preventing wide-distribution testing.

### 11. Entity-in-a-Box Behavior
* **Classification**: `locally demonstrated`
* **Definition**: Confinement bounds are respected; sandboxed code executes without escaping directory trees or modifying host systems.
* **Evidence**: `tests/test_sandbox_hardening.py` and sandbox escape checks.

### 12. External Real-World Validation
* **Classification**: `not proven`
* **Definition**: Performance benchmarked on external platforms or verified by independent replication agents.
* **Blocker**: Demoted until independent multi-party consensus replication is executed.

### 13. DNU AGI
* **Classification**: `not proven`
* **Definition**: Passing the full 100-task AGI Proof battery with scoring >85%.
* **Evidence/Limit**: The configured local 100-task DNU battery passed with baseline and ablation separation, but this does not prove AGI or general intelligence in the unrestricted scientific sense.

### 14. AGI-Candidate
* **Classification**: `locally demonstrated`
* **Definition**: Meeting the comprehensive criteria of DNU, Agency Emergence, and External Live Validation.
* **Evidence**: The configured local final-proof profile includes DNU, baselines, ablations, agency emergence, external validation, unified scenario, receipt coverage, artifact consistency, and Aletheia Tier 5 validation. This supports "proof-bearing AGI-candidate architecture", not "AGI solved".

### 15. Production-Sealed
* **Classification**: `causally demonstrated`
* **Definition**: Strict compilation, flagship readiness, enterprise gates, and production linting completed with zero critical/high issues.
* **Evidence**: Pass status of flagship readiness and production surface lint.

### 16. Mature RSI
* **Classification**: `not proven`
* **Definition**: Recursive self-improvement resulting in significant autonomous capability gains without human intervention.
* **Blocker**: Blocked by safety policies preventing non-interactive execution of self-modification.

### 17. Subjective Consciousness
* **Classification**: `not proven`
* **Definition**: Subjective awareness, qualitative feelings (qualia), or phenomenological experience.
* **Blocker**: Metaphysical/phenomenological qualities are strictly outside the scope of this engineering codebase.

### 18. Personhood
* **Classification**: `not proven`
* **Definition**: Legal, moral, or philosophical status as a conscious person.
* **Blocker**: Strictly outside the scope of a software cognitive agent runtime.

### 19. Metaphysical Free Will
* **Classification**: `not proven`
* **Definition**: Causal agency unconstrained by physics or antecedent factors.
* **Blocker**: Strictly outside the scope of computational models of volition.

### 20. Indefinite Autonomy
* **Classification**: `not proven`
* **Definition**: Bounded resource growth and stable operation over arbitrary long-horizon periods.
* **Blocker**: Long-horizon longevity soak (72h+) blocked by execution environment limits.

### 21. Synthetic Cognitive Entity
* **Classification**: `locally demonstrated`
* **Definition**: Cohesive agency, volition, and continuity verified by complete live/sandbox batteries.
* **Evidence**: Boxed agency, operational volition, unified runtime scenario, restart/memory continuity checks, and receipt coverage pass under the configured local profile. This is an operational engineering label, not personhood or subjective consciousness.

### 22. Experience-Adjacent Functional Indicators
* **Classification**: `locally demonstrated`
* **Definition**: Internal states influencing future perception, memory indexing, and self-reports in structured, traceable ways.
* **Evidence**: Metacognition loops and state-behavior coupling tests in the test suite.
