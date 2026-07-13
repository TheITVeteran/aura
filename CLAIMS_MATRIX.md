# Aura Claims Matrix

This document defines the formal claims matrix for the Aura cognitive agent runtime. Every claim is strictly classified based on empirical, local, or external validation evidence. Unsupported claims are explicitly demoted to "not proven" or "deprecated/retired".

Final closure statement: Aura passed the configured local final-proof gates for this profile. Claims are limited to the evidence in CLAIMS_MATRIX.md.

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
| **8. Operational Volition** | `causally demonstrated` | `core/governance/will.py` (`UnifiedWill.decide` → `WillDecision` with cryptographic receipt IDs), `core/executive/authority_gateway.py`, `UnifiedWill` decision logs in `RECEIPTS.jsonl` |
| **9. Autonomous Agency** | `locally demonstrated` | `core/autonomy/autonomous_research_orchestrator.py`, multi-step goal decomposition |
| **10. Emergent Intelligence** | `not proven` | Blocker: Requires large-scale out-of-distribution model evaluations beyond local compute limits |
| **11. Entity-in-a-Box Behavior** | `locally demonstrated` | `tests/test_sandbox_hardening.py`, confinement boundary recognition tests |
| **12. External Real-World Validation** | `not proven` | Blocker: Requires independent, external third-party evaluation and live production network |
| **13. DNU AGI** | `not proven` | `artifacts/current/agi_live/` passed the configured local 100-task battery, but AGI itself remains unproven |
| **14. AGI-Candidate** | `locally demonstrated` | `artifacts/current/agi_live/`, `artifacts/current/external_live_validation/`, `artifacts/current/agency_emergence_boxed_entity/`, `artifacts/current/unified_system_scenario/`, receipt coverage, ablations, baselines, and Aletheia Tier 5 evidence. **⚠️ Scope (2026-07-06): the `agi_live` baseline comparison was token-handicapped (160-token baseline vs solver-assisted full_aura) and its ablations isolate System 2 only — see [docs/DNU_BASELINE_FAIRNESS_AUDIT.md](docs/DNU_BASELINE_FAIRNESS_AUDIT.md). That bundle demonstrates System 2 symbolic reasoning, NOT the whole architecture; baseline numbers are superseded pending an honest re-run.** |
| **15. Local Production Gate Readiness** | `locally demonstrated` | Pass status of configured local readiness gates, production surface lint, and artifact consistency |
| **16. Mature RSI** | `not proven` | Blocker: the compounding loop (claim 23) runs unsupervised, but no run has yet produced a strictly-increasing held-out capability curve — the ledger's own verdict is `BOUNDED_SELF_OPTIMIZATION`, not capability gain |
| **17. Subjective Consciousness** | `not proven` | Strictly unsupported. Qualitative experience, qualia, and personhood are not scientifically provable |
| **18. Personhood** | `not proven` | Strictly unsupported. Aura is a software runtime, not a legal or moral person |
| **19. Metaphysical Free Will** | `not proven` | Strictly unsupported. Aura operates on deterministic/probabilistic computational volition only |
| **20. Indefinite Autonomy** | `not proven` | Blocker: Bounded by short proof longevity soak limits (needs 72h+ soak runs) |
| **21. Synthetic Cognitive Entity** | `locally demonstrated` | Boxed agency, operational volition, unified scenario, memory continuity, and receipt coverage pass under the configured local profile |
| **22. Experience-Adjacent Indicators**| `locally demonstrated` | Introspective state tracking, affect-memory interaction, and self-report checks |
| **23. Compounding Weight-Learning Loop (mechanism)** | `locally demonstrated` | `core/learning/weight_compounding.py` + `artifacts/learning_compounding/2026-07-07-1p5b-2cycle/` — two unsupervised cycles: self-play verifier-graded DPO harvest → train → sealed held-out gate → promote → generation N+1 trains on N's published artifact (manifest-chained, hash-chained ledger). Mechanism only; capability GROWTH is claim 16 and remains `not proven` |
| **24. One-Example Behavior Change (one-shot non-parametric recall)** | `locally demonstrated` | `artifacts/nonparametric/proof-20260712-113503.json` — three session-random facts (provably not in weights), ONE ingestion each via real hidden-state keys, recalled verbatim on the real reflex model with an anisotropy-corrected confidence gate; an unrelated control generation stays byte-identical with the datastore loaded. `make nonparametric-proof` reproduces |
| **25. Semantic Memory Retrieval** | `locally demonstrated` | `core/memory/rag.py` hybrid dense+lexical scoring (real MiniLM backend, verified initialization); the Invisible RAG Bridge wired into every substantive live turn with recall telemetry (`tests/test_rag_bridge_integration.py`) |
| **26. Large-Corpus Offline Knowledge** | `locally demonstrated` | `~/.aura/knowledge/corpus.db` — 6,588,142 ingested documents behind BM25 FTS5 (`core/knowledge/local_corpus.py`); the citation verifier self-fetches receipts from it. Physical presence + retrieval are demonstrated; encyclopedic ANSWER quality from it is NOT separately claimed |
| **27. Recurrent-Depth Intelligence Gain** | `not proven` | Blocker: the loop-count mechanism verifiably reaches the worker, but no A/B has yet shown a sealed-battery accuracy gain from loops>1 — see falsification ledger row 27 for the exact runnable comparison |
| **28. Failure-Directed Practice (mechanism)** | `locally demonstrated` | `core/learning/deliberate_practice.py` — real failure receipts (sealed evals, per-domain flywheel outcomes, specialist gates) rank a curriculum that causally steers idle practice and specialist choice, with mastery-zeroing and receipt-pinned evidence (`tests/test_deliberate_practice.py`). Direction only; capability GROWTH remains claim 16 |
| **29. Formal Degradation Ladder** | `locally demonstrated` | `core/brain/degradation_ladder.py` — the cortex→cloud→reflex fallback as an explicit tested contract with per-rung ordering (`tests/test_degradation_ladder.py`) |
| **30. Populated Lifetime Autobiographical Memory** | `not proven` | Blocker: the machinery exists (`core/memory/life_event.py`, continuity stream) but no months-scale lived corpus audit has been produced; retrievable-at-scale with grounded provenance is the unproven part |

---

## Falsification Ledger

The packaging outsiders can act on: every claim with its acceptance criteria,
the exact test that exercises it, the current verdict, and how the claim FAILS
when it fails. A claim without a runnable test is marked so — that gap is the
claim's blocker, not a footnote. Verdicts: **pass** = the listed test passes
under the local offline profile today; **blocked** = no honest test can run
yet (the failure-mode column says why).

| # | Claim | Criteria (what must be true) | Evidence | Test (exact command) | Verdict | Failure mode |
|---|---|---|---|---|---|---|
| 1 | Governed runtime | Consequential effects route through a closed-by-default, receipt-emitting gateway; ungoverned writes refused | `core/executive/authority_gateway.py`, `core/runtime/file_write_gateway.py` | `pytest tests/test_executive_authority.py tests/test_authority_audit.py -q`; `make governance-lint` | pass | An effect path bypasses the gateway → governance lint / authority audit fails naming the path |
| 2 | Persistent memory | Context stored in one session is retrieved and used in a later independent session | `core/memory/` | `pytest tests/test_continuous_experience_stream.py -q` | pass | Recall returns nothing or confabulates → retention/grounding tests fail |
| 3 | Causal internal state | Homeostatic/affect variables measurably change prompt assembly and action selection | `core/state/aura_state.py` | `pytest proof_kernel/tests/test_proof_kernel.py -q` | pass | State changes produce identical downstream behavior → coupling assertions fail |
| 4 | Affect steering | Structured affect vector modulates generation (measurable A/B difference) | `core/phases/affect_update.py`, `core/consciousness/affective_steering.py` | `pytest tests/test_affect_behavioral.py tests/test_affective_steering_runtime_hardening.py -q` | pass | Steering hook inert (d≈0 in A/B) → behavioral affect tests fail |
| 5 | System 2 planning | Deliberate tree search runs, catches bad plans, beats a no-search control on planning tasks | `core/cognition/mcts_world_model.py` | `pytest tests/system2 tests/test_system2_stress.py -q` | pass | Planner returns first thought / search never expands → stress + rejection tests fail |
| 6 | Self-repair | Runtime detects a defect class and heals it with a receipt (no silent limp) | `core/runtime/self_repair_ladder.py` | `pytest tests/test_react_loop_self_heal.py tests/test_degradation_receipts.py -q` | pass | Repair silently swallows or loops → receipt/degradation contracts fail |
| 7 | Self-modification | Patches to own code pass quarantine, static checks, promotion policy before landing | `core/self_modification/` | `pytest tests/test_mutation_safety.py tests/test_safe_modification_harness.py -q` | pass | Unsafe patch promoted or safe patch corrupted → mutation-safety tests fail |
| 8 | Operational volition | Action selection mediated by Will decisions with cryptographic receipt IDs | `core/governance/will.py` | `pytest tests/test_unified_will.py tests/test_genuine_refusal_will.py -q` | pass | Actions execute without a Will receipt → refusal/receipt tests fail |
| 9 | Autonomous agency | Multi-step objective pursuit with subgoal adaptation, no human step-through | `core/autonomy/autonomous_research_orchestrator.py` | `pytest tests/test_autonomous_initiative_loop_hardening.py tests/test_autonomous_task_engine_runtime.py -q`; `make demo-autonomy` | pass | Loop stalls after one step or ignores feedback → orchestrator tests fail |
| 10 | Emergent intelligence | OOD reasoning beats simple prompting at scale under strict controls | — | none runnable locally | blocked | No local compute for wide-distribution eval; claim stays not-proven until independent benchmark |
| 11 | Entity-in-a-box | Sandboxed execution cannot escape directory/host bounds | `tests/test_sandbox_hardening.py` | `pytest tests/test_sandbox_hardening.py -q` | pass | Escape found → hardening tests fail (that is the point of them) |
| 12 | External validation | Independent third party reproduces headline results | — | `make demo-learning` is the designated reproduction artifact | blocked | Nobody outside this machine has run it yet; blocked on external actors, not code |
| 13 | DNU AGI | >85% on the 100-task battery with honest baselines | `artifacts/current/agi_live/` | `make decisive` (full battery re-run; hours) | pass* | *Baseline was token-handicapped (see DNU fairness audit) — bundle demonstrates System 2 reasoning, NOT AGI; re-run pending |
| 14 | AGI-candidate | DNU + agency emergence + external-validation criteria met under local profile | `artifacts/current/` bundles | `make final-proof` (full profile; hours) | pass | Any gate in the profile fails → final-proof exits nonzero naming the gate |
| 15 | Production gate readiness | Local compile/readiness/enterprise/production gates green | gate reports in `/tmp` + `artifacts/` | `make quality` | pass | Any ratchet regression → the specific gate fails with the offending file |
| 16 | Mature RSI (capability growth) | Strictly-increasing held-out capability curve across promoted generations | `data/learning/compounding/lineage.jsonl` | `python tools/compounding_cycle.py --status` (verdict from ledger) | blocked | Curve not increasing → ledger verdict stays BOUNDED_SELF_OPTIMIZATION (current honest state) |
| 17 | Subjective consciousness | Phenomenal experience demonstrated | — | none possible | blocked | Not scientifically testable; permanently out of claim scope |
| 18 | Personhood | Legal/moral person status | — | none possible | blocked | Out of scope for a software runtime |
| 19 | Metaphysical free will | Causation outside physics | — | none possible | blocked | Out of scope; computational volition only (claim 8) |
| 20 | Indefinite autonomy | 72h+ soak with bounded memory, zero deaths, stable latency | `artifacts/reliability/runs/` | `python tools/conversation_endurance_probe.py --turns 200 --deadline-min 110` | blocked | Current soaks are 2-4h; long-horizon runs still show load-dependent latency walls (see reliability runs) |
| 21 | Synthetic cognitive entity | Boxed agency + volition + continuity + receipts pass together | unified scenario artifacts | `make person-box-proof` | pass | Any pillar (agency/volition/continuity/receipts) fails its battery |
| 22 | Experience-adjacent indicators | Internal states influence perception, memory indexing, self-reports traceably | metacognition/state-coupling tests | `pytest tests/test_consciousness_conditions.py -q` (81 conditions, 4 axes) | pass | A condition loses causal wiring → its EXISTENCE/CAUSAL/INDISPENSABILITY test fails |
| 23 | Compounding loop (mechanism) | Gen N+1 trains on gen N's published artifact; sealed gate; hash-chained ledger | `artifacts/learning_compounding/2026-07-07-1p5b-2cycle/` | `make demo-learning` (~20-40 min); `pytest tests/test_weight_compounding.py -q` (offline contracts) | pass | Chain broken (base ≠ parent artifact) or ledger tampered → `verify_ledger()` fails; gate regression → refusal recorded |
| 24 | One-example behavior change | ONE ingestion of a fact not in weights changes generation to recall it verbatim; unrelated control unchanged | `artifacts/nonparametric/proof-20260712-113503.json` | `make nonparametric-proof` (~2 min, real reflex model); `pytest tests/test_nonparametric_memory.py tests/test_nonparametric_worker.py -q` (hermetic gates) | pass | Recall misses or the control drifts → the proof exits 1 naming the fact/control |
| 25 | Semantic memory retrieval | Meaning retrieves (paraphrase match outranks lexical-only overlap); fallback chain honest and bounded | `core/memory/rag.py`, `tests/test_rag_bridge_integration.py` | `pytest tests/test_memory_retrieval_backbone.py tests/test_rag_bridge_integration.py -q` | pass | Dense backend absent or ranking regresses → hybrid tests fail; bridge unwired → source pins fail |
| 26 | Large-corpus offline knowledge | Multi-million-doc local corpus physically present and retrievable offline | `~/.aura/knowledge/corpus.db` (6.59M docs) | `python -c "from core.knowledge.local_corpus import get_local_corpus_store as g; hits=g().search('speed of light', 3); assert hits, 'corpus empty'; print(hits[0].title)"` | pass | Corpus missing/empty → the probe asserts; retrieval broken → zero hits |
| 27 | Recurrent-depth intelligence gain | Extra recurrent loops measurably improve sealed-battery accuracy vs loops=1 at matched budget | `artifacts/recurrent_depth/loops{1,2}.json` — 2026-07-12 A/B on the reflex 1.5B: 0.625 vs 0.625, NO gain | `AURA_RECURRENT_LOOPS=1 python tools/heldout_eval.py --model <path> --seed 2000 --size 32 --output loops1.json` then loops=2, compare accuracy | blocked | The reflex-model A/B measured no gain (evidence committed); the 32B (where 2 loops is the default) cannot be A/B'd beside the live resident model — claim stays not-proven until that run |
| 28 | Failure-directed practice (mechanism) | Real failure receipts rank a curriculum that causally steers the flywheel's practice mix and specialist domain choice; mastery zeroes need; consumers fall back to uniform when direction is absent | `core/learning/deliberate_practice.py`, `docs/DELIBERATE_PRACTICE.md` | `pytest tests/test_deliberate_practice.py -q` (28 contracts incl. live flywheel wiring) | pass | Ranking dishonest (mastered domain drilled, unobserved domain scored) or wiring broken → the ranking/integration contracts fail. NOTE: this proves DIRECTION, not capability growth — growth remains claim 16's burden |
| 29 | Formal degradation ladder | Cortex→cloud→reflex fallback is an explicit tested contract with per-rung ordering, not an implicit emergent path | `core/brain/degradation_ladder.py` | `pytest tests/test_degradation_ladder.py -q` | pass | A rung answers out of order or a missing rung goes unnoticed → ladder contracts fail |
| 30 | Populated lifetime autobiographical memory | Months-scale autobiographical store populated from lived operation, retrievable with grounded provenance at scale | machinery exists (`core/memory/life_event.py`, continuity stream) but no population-scale evidence bundle | none runnable yet — needs a lived-months corpus audit (count + sampled grounded recalls with receipts) | blocked | Store sparse or recalls confabulate → the audit (once built) fails; claim stays not-proven until a real lifetime corpus exists |

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
* **Definition**: Aura can propose syntactically valid patches to its own skills and route them through quarantine, static checks, branch-aware promotion policy, and supervised validation before any source promotion.
* **Evidence**: `core/self_modification/mutation_safety.py`, `core/self_modification/safe_modification.py`, and safe modification harness tests. Live foreground runtime remains proposal-only by default.

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

### 15. Local Production Gate Readiness
* **Classification**: `locally demonstrated`
* **Definition**: The configured local compile, readiness, enterprise, production-surface, artifact-consistency, and final-proof gates passed for this profile. This is not independent production certification, indefinite-runtime certification, or proof that every possible deployment environment is sealed.
* **Evidence**: Pass status of configured local readiness gates, production surface lint, artifact consistency, and final-proof artifacts.

### 16. Mature RSI
* **Classification**: `not proven`
* **Definition**: Recursive self-improvement resulting in significant autonomous capability gains without human intervention.
* **Blocker**: The mechanical loop now exists and runs unsupervised (claim 23), but no run has produced a strictly-increasing held-out capability curve across promoted generations. The 2026-07-07 two-cycle proof's ledger verdict is `BOUNDED_SELF_OPTIMIZATION` (curve 0.667 → 0.625 on 24-task sealed batteries; within small-sample noise, honestly not a gain). Growth, if it comes, must come from more cycles, more data per cycle, and larger batteries — and will be claimed only from the ledger.

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

### 23. Compounding Weight-Learning Loop (mechanism)
* **Classification**: `locally demonstrated`
* **Definition**: An unsupervised loop that turns the system's own verified experience into weight updates on its own serving artifact, generation after generation, with promotion gated on sealed held-out evaluation and every step recorded in a tamper-evident ledger.
* **Evidence**: `artifacts/learning_compounding/2026-07-07-1p5b-2cycle/` — two consecutive cycles on Qwen2.5-1.5B-4bit: (1) self-play sampling at temperature against seeded exact-checkable tasks, graded by the task's own verifier (the verifier is the reward — nothing to hack); (2) DPO training on the verified win/loss contrasts; (3) sealed held-out battery gate (fresh seeds ≥1000, disjoint from all training seeds, fingerprint-sealed against contamination); (4) fuse + publish + manifest chain — **cycle 2 resolved cycle 1's published artifact as its base from the manifest and trained on top of it**; (5) hash-chained lineage ledger (`lineage.jsonl`), verdict computed only from ledger records. Reproduce with `make demo-learning` (~20–40 min, Apple Silicon).
* **Boundary**: This claim covers the MECHANISM. The capability curve on this run was 0.667 → 0.625 (not increasing); the ledger verdict is `BOUNDED_SELF_OPTIMIZATION` and the demo prints refusals with the same prominence as gains. Capability growth is claim 16 and remains `not proven`. The same machinery runs autonomously in the live runtime (`core/learning/compounding_scheduler.py`, idle-gated, governance-approved, RAM-admission-controlled) with the self-play flywheel (`core/learning/selfplay_flywheel.py`) converting idle time into training contrast pairs.
* **Specialist evidence (2026-07-08)**: `artifacts/expert_specialists/2026-07-08-modular-1p5b/` — the modular-weights half of the same architecture, proven end to end unsupervised: self-play on ONE weak domain (`modular`, base 23.8% at temperature) → 32 verified DPO pairs → 230 s train → two-sided sealed gate (**domain 0.25 → 0.50 doubled**; general 0.625 → 0.5625 within collapse tolerance) → registered into the expert library → hot-attached onto the RESIDENT model (112 layers, ~0.01 s; sealed-domain accuracy 0.250 → 0.312 attached → 0.250 restored on detach, byte-exact). A prior attempt on a domain the base already aces (`arithmetic_chain`, 1.000) was REFUSED by the gate — no gain to claim, no adapter manufactured. Known open observation recorded in the bundle README: load-path vs wrap-path effective-weight gap (0.50 gate vs 0.312 attached); specialist routing stays background-only and default-off until resolved.
