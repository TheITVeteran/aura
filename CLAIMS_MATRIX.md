# Aura Claims Matrix

This document defines the formal claims matrix for the Aura cognitive agent runtime. Every claim is strictly classified based on empirical, local, or external validation evidence. Unsupported claims are explicitly demoted to "not proven" or "deprecated/retired".

Final closure statement: Aura passed the configured local final-proof gates for this profile. Claims are limited to the evidence in CLAIMS_MATRIX.md.

## Read this before the table

Several claim names in this repository are heavier than what they name. "Operational
Volition", "Synthetic Cognitive Entity", "Experience-Adjacent Indicators",
"Qualia Engine", "Phenomenal Richness", "Consciousness Guarantee", "Personhood
Proof Battery", "Strange-Loop Detection" — a reader who meets these as module
names, test names or dashboard labels will infer far more than the tests behind
them establish, and disclaiming it further down was never going to undo that.

So every extraordinary label in the table below now carries its **operational
definition in the same cell as the claim**, in bold, stating what the term means
*here* in plain engineering vocabulary. If a term cannot be cashed out that way,
it does not belong in a claim. The general translation:

| Reads as | Is |
| :--- | :--- |
| Volition / Will | action selection over weighted parameters, with a signed receipt |
| Cognitive entity | several locally defined batteries passing in one profile |
| Experience-adjacent | state variables that measurably reach prompts, memory and self-reports |
| Qualia / phenomenal | named scalar features of activations (entropy, novelty, change) |
| Φ / integrated information | a custom integration statistic over selected state variables |
| Neurochemical | hand-designed scalar controls with biologically borrowed names |
| Guarantee / proof battery | a set of tests this project wrote and this project runs |
| Locally demonstrated | passed on this machine, this profile, this project's battery |

Two rules follow, and both are enforced rather than promised:

1. **Retracted evidence cannot support a claim.** An artifact whose validity has
   been withdrawn carries a machine-readable `RETRACTION.json` beside it, and
   `make evidence-integrity` fails if any claim above `not proven` cites one.
   This exists because claim 14 sat at `locally demonstrated` for a month while
   the same cell explained that its primary evidence was unfair.
2. **The measurement is never edited.** A retraction is a sidecar. What was
   measured stays byte-identical on the record, because deleting the numbers
   would also delete the evidence that the flaw was there.

The most accurate reading of this repository is the unsupported-claims ledger
(`CLAIMS_NOT_SUPPORTED.md`) and the falsification ledger below — not the names
of the modules.

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
| **8. Operational Volition** | `causally demonstrated` | **Operationally: an algorithm scores candidate actions against weighted parameters, selects one, and signs a receipt naming the decision, the authority that approved it and the evidence expected.** That is deterministic/probabilistic action selection with an audit trail. Counterfactual evaluation, a decision object, an authorisation signature and a receipt id do not make action selection a different KIND of thing — they make it reviewable, which is the actual and useful claim. Nothing here is evidence of willing in any richer sense; see claim 19. `core/governance/will.py` (`UnifiedWill.decide` → `WillDecision` with cryptographic receipt IDs), `core/executive/authority_gateway.py`, `UnifiedWill` decision logs in `RECEIPTS.jsonl`. A receipt proves a pathway RAN; it cannot prove the goal was appropriate, the environment was understood, the success criterion was adequate, or the outcome was useful |
| **9. Autonomous Agency** | `locally demonstrated` | `core/autonomy/autonomous_research_orchestrator.py`, multi-step goal decomposition |
| **10. Emergent Intelligence** | `not proven` | Blocker: Requires large-scale out-of-distribution model evaluations beyond local compute limits |
| **11. Entity-in-a-Box Behavior** | `locally demonstrated` | `tests/test_sandbox_hardening.py`, confinement boundary recognition tests |
| **12. External Real-World Validation** | `not proven` | Blocker: Requires independent, external third-party evaluation and live production network |
| **13. DNU AGI** | `not proven` | `artifacts/current/agi_live/` recorded 100/100 on a project-authored 100-task battery. The grader is condition-independent, so that score stands as a record of what `full_aura` scored. It is an ABSOLUTE score against a self-authored battery and is not evidence of superiority over anything, because the comparison arms were handicapped — see `artifacts/current/agi_live/RETRACTION.json`. AGI itself remains unproven |
| **14. AGI-Candidate** | `not proven` | **Retracted 2026-08-06.** This was classified `locally demonstrated` while carrying a warning that its own primary evidence was unfair — a classification and its retraction held in the same cell. The `agi_live` baselines ran at 160 tokens against an effectively unbounded, solver-assisted `full_aura`, on tasks that cannot be answered in 160 tokens; the identical 0.1667 across three structurally different baselines is the handicap's signature, not three measurements agreeing. Its ablations isolate System 2 only. A comparison that unfair is not weakened evidence, it is no evidence, and an asterisk is not a classification. Machine-readable retraction: `artifacts/current/agi_live/RETRACTION.json`; audit: [docs/DNU_BASELINE_FAIRNESS_AUDIT.md](docs/DNU_BASELINE_FAIRNESS_AUDIT.md). Blocker: a budget-matched re-run meeting `replacement_requirements` in that retraction. `make evidence-integrity` now refuses to let any claim cite retracted evidence as support. |
| **15. Local Production Gate Readiness** | `locally demonstrated` | Pass status of configured local readiness gates, production surface lint, and artifact consistency |
| **16. Mature RSI** | `not proven` | Blocker: the compounding loop (claim 23) runs unsupervised, but no run has yet produced a strictly-increasing held-out capability curve — the ledger's own verdict is `BOUNDED_SELF_OPTIMIZATION`, not capability gain |
| **17. Subjective Consciousness** | `not proven` | Strictly unsupported. Qualitative experience, qualia, and personhood are not scientifically provable |
| **18. Personhood** | `not proven` | Strictly unsupported. Aura is a software runtime, not a legal or moral person |
| **19. Metaphysical Free Will** | `not proven` | Strictly unsupported. Aura operates on deterministic/probabilistic computational volition only |
| **20. Indefinite Autonomy** | `not proven` | Blocker: Bounded by short proof longevity soak limits (needs 72h+ soak runs) |
| **21. Synthetic Cognitive Entity** | `locally demonstrated` | **Operationally: five locally defined test batteries — boxed agency, operational volition, unified scenario, memory continuity, receipt coverage — pass together in one profile.** "Entity" here names that conjunction and nothing else. It is a label this project defined for a set of tests this project wrote; it is not a finding about what Aura is, and a reader should not have to reach a ledger to learn that. The interesting engineering content is that the batteries pass TOGETHER rather than separately, which is a real integration result and a much smaller claim than the phrase suggests |
| **22. Experience-Adjacent Indicators**| `locally demonstrated` | **Operationally: state variables measurably influence prompt assembly, memory writes and self-reports.** The coupling is causal and tested. It is also close to guaranteed by construction: the state is serialised into the prompt and the model is instructed to answer consistently with it, so "I feel curious" is a state-to-language pipeline working as built. That demonstrates the pipeline is connected — which was worth proving, and was once NOT true (see the disconnected-hook history in claim 4's row). It is not evidence of introspective access, and "experience-adjacent" should be read as naming the pipeline, not the experience |
| **23. Compounding Weight-Learning Loop (mechanism)** | `locally demonstrated` | `core/learning/weight_compounding.py` + `artifacts/learning_compounding/2026-07-07-1p5b-2cycle/` — two unsupervised cycles: self-play verifier-graded DPO harvest → train → sealed held-out gate → promote → generation N+1 trains on N's published artifact (manifest-chained, hash-chained ledger). Mechanism only; capability GROWTH is claim 16 and remains `not proven` |
| **24. One-Example Behavior Change (one-shot non-parametric recall)** | `locally demonstrated` | `artifacts/nonparametric/proof-20260712-113503.json` — three session-random facts (provably not in weights), ONE ingestion each via real hidden-state keys, recalled verbatim on the real reflex model with an anisotropy-corrected confidence gate; an unrelated control generation stays byte-identical with the datastore loaded. `make nonparametric-proof` reproduces |
| **25. Semantic Memory Retrieval** | `locally demonstrated` | `core/memory/rag.py` hybrid dense+lexical scoring (real MiniLM backend, verified initialization); the Invisible RAG Bridge wired into every substantive live turn with recall telemetry (`tests/test_rag_bridge_integration.py`) |
| **26. Large-Corpus Offline Knowledge** | `locally demonstrated` | `~/.aura/knowledge/corpus.db` — 6,588,142 ingested documents behind BM25 FTS5 (`core/knowledge/local_corpus.py`); the citation verifier self-fetches receipts from it. Physical presence + retrieval are demonstrated; encyclopedic ANSWER quality from it is NOT separately claimed |
| **27. Recurrent-Depth Intelligence Gain** | `not proven` | Blocker: the loop-count mechanism verifiably reaches the worker, but no A/B has yet shown a sealed-battery accuracy gain from loops>1 — see falsification ledger row 27 for the exact runnable comparison |
| **28. Failure-Directed Practice (mechanism)** | `locally demonstrated` | `core/learning/deliberate_practice.py` — real failure receipts (sealed evals, per-domain flywheel outcomes, specialist gates) rank a curriculum that causally steers idle practice and specialist choice, with mastery-zeroing and receipt-pinned evidence (`tests/test_deliberate_practice.py`). Direction only; capability GROWTH remains claim 16 |
| **29. Formal Degradation Ladder** | `locally demonstrated` | `core/brain/degradation_ladder.py` — the cortex→cloud→reflex fallback as an explicit tested contract with per-rung ordering (`tests/test_degradation_ladder.py`) |
| **30. Populated Lifetime Autobiographical Memory** | `not proven` | Blocker: the machinery exists (`core/memory/life_event.py`, continuity stream) but no months-scale lived corpus audit has been produced; retrievable-at-scale with grounded provenance is the unproven part |
| **33. Indirect Injection Is Refused At The Gate** | `locally demonstrated` | **Operationally: a turn that ingested web/tool/external content downgrades the Rule-of-Two input trust of every surface acting during that turn, so an executing in-process surface goes from two legs to three — a violation — while untrusted content is in the context.** Measured: `self_modification_apply` and `desktop_automation` both declare `input_trust=TRUSTED` because their input is "model-generated"/"internally-formed intent"; that is correct when Aura has read nothing untrusted and wrong the rest of the time, because indirect injection does not make untrusted text act — it makes untrusted text persuade something trusted to act. `core/security/content_provenance.py` carries the least-trusted origin per turn (ContextVar, so a background research turn cannot taint a foreground one), `HandlerSpec.violates_now()` asks the live turn, and the web fetch path in `core/runtime/network_gateway.py` marks WEB on every response. `pytest tests/test_content_provenance.py -q` (19). **Enforced, not merely visible:** `core/runtime/desktop_action_gateway.run_applescript` refuses on a turn that ingested untrusted content, with a reason naming the origin and a recorded degradation, and the refusal is shaped like every other failure that gateway returns so no caller needs a new branch. **Scope, stated rather than left to be discovered:** the self-modification gateway is NOT yet wired to refuse; tool output is not yet marked, so a turn that read only a tool result still reads as trusted; and `OWNER_FILE` sits deliberately BELOW the untrusted floor, so a README inside a cloned repository is trusted today — treating every file read as untrusted would disarm action gates on nearly every turn and produce a control that gets switched off, which protects nothing. That is a policy choice with a named residual risk, not an oversight. It also makes no judgement about whether any given untrusted text is malicious, deliberately: that judgement cannot be made reliably and a component claiming to make it would recreate the problem with more confidence behind it |
| **32. Interruption Without Model Reload** | `locally demonstrated` | **Operationally: a running generation can be stopped between decode steps, leaving the model loaded.** The cancel channel is a lock-free shared word; the worker polls `soft_cancel_requested(cancel_seq, job_seq)` inside its token loop at four sites and returns a partial `soft_cancelled` response, so cancel latency is about one decode step and the ~20GB of weights stay resident. Escalation to `force_abort_active_generation` (which does kill the worker and pay a full reload) is the rung ABOVE this, not the only rung. Evidence: `pytest tests/test_mlx_soft_cancel.py tests/test_missed_deadline_does_not_destroy_a_healthy_cortex.py tests/test_llm_router_foreground_preemption.py tests/test_deliberate_cancel_is_not_endpoint_damage.py -q` (60 tests). Includes the honest failure paths: no active generation reports `requested: False` rather than faking a cancel, and a write lost to a racing job start is read back and reported as not-requested — a preemption ladder that believes a lost write succeeded never escalates. **Not yet measured on the live 32B**: cancel latency and residency are asserted against the real predicates and the real client, not against a loaded 32B under load |
| **31a. Retrieval Earns Its Cost (memory only)** | `locally demonstrated` | **Measured 2026-08-06, and the first capability result in this table.** `artifacts/ablation/capability_scorecard_longhistory.json`: 40 multi-turn recall tasks, one local model, identical decode settings, and BOTH arms limited to 12 turns of context — long_context spends that budget on the most recent turns (what every chat client does), Aura's retrieval spends it on the most relevant. Where history exceeds the window: long_context 0.000, retrieval 1.000, delta +1.000, 95% CI [1.000, 1.000] by paired bootstrap. Not tautological: the control has the architecture removed while the information stays available in principle, and the tasks are solvable without Aura. **Scope, which is narrow:** retrieval only — not the cognitive layer as a whole; one task family; one small model; and against a NAIVE recency baseline (one that also kept the first turns would score higher, and this does not claim to beat it). **And it is regime-dependent, measured both ways:** `capability_scorecard_withinwindow.json` shows delta 0.000 when history fits the window. The advantage exists where history exceeds what a caller can afford to send, which is what memory is for and is not every situation. `python tools/capability_ablation.py --responder mlx --model <id> --history-turns 40` |
| **31. The Cognitive Layer Earns Its Cost** | `not proven` | **Added 2026-08-06 because its absence was the loudest thing in this table.** Claim 31a now supplies ONE component's worth of capability evidence; this row stays `not proven` because one subsystem on one task family is not the layer. Every claim above says a mechanism EXISTS and is wired. None of them says the mechanism makes Aura better at anything a user wants. A far smaller computer-use agent performs the demo tasks — search, write a file, take a note, navigate a repo, produce a PDF — without IIT, qualia metrics, neurochemical simulation, substrate ODEs, dream cycles or theory arbitration, and those layers cost latency, memory, tuning surface, failure modes and debugging difficulty that this repository has repeatedly paid (a GABA lockout that suppressed initiative for a whole session, disconnected affect hooks reporting themselves online, a prompt-cache defect first attributed to "cognition"). Complexity is justified by measured advantage; none has been measured. Blocker: a budget-matched full-system ablation — full Aura versus stripped Aura at identical model, tools, memory, prompts, token and wall-clock budget — scored on TASK SUCCESS with every attempt counted. `core/evaluation/matched_budget.py` refuses to emit a verdict when the arms differ, and reports `clean_success_rate` beside the raw rate so a run rescued by a fallback, a retry or a human is not counted as the architecture succeeding |

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
| 13 | DNU AGI | >85% on the 100-task battery **with honest baselines** | `artifacts/current/agi_live/` + `RETRACTION.json` | `make decisive` (full battery re-run; hours) | blocked | The criterion says "with honest baselines" and the baselines were handicapped, so the criterion was never met. It was carried as `pass*` with the disqualifying fact in the footnote. A footnote is not a verdict. The 100/100 stands as an absolute score on a self-authored battery |
| 14 | AGI-candidate | DNU + agency emergence + external-validation criteria met under local profile | `artifacts/current/` bundles, minus every bundle carrying a `RETRACTION.json` | `make final-proof` (full profile; hours); `make evidence-integrity` (seconds) | blocked | Depends on claim 13, which is blocked. `make evidence-integrity` fails while any claim above `not proven` cites retracted evidence — so this row cannot silently return to `pass` without a budget-matched re-run replacing the retraction |
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
| 31 | The cognitive layer earns its cost | Full Aura beats stripped Aura on TASK SUCCESS at identical model, tools, memory, prompts, token and wall-clock budget, with every attempt counted including crashes, retries, fallbacks and human intervention | none. This is the gap, stated | `pytest tests/test_matched_budget.py -q` proves the harness refuses an unmatched comparison; the comparison itself needs a battery nobody here authored | blocked | The honest failure mode is that the deltas come back at zero, and that result gets published the same as any other. A comparison whose arms differ returns `void` rather than a number — the previous attempt at this reported 100% vs 16.67% from a 160-token baseline against an unbounded, solver-assisted treatment |

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
