# Improvement Pass — July 1–2, 2026

> **Historical record — 2026-07-05.** A dated snapshot, kept as written for
> provenance. It is not a statement about the system today and is
> deliberately not updated. Current status: [DOC_STATUS.md](DOC_STATUS.md).

Whole-codebase fix-and-improve pass. Method: live-instance forensics first
(crash artifacts, logs, health pulses), then fix the proven defects and
ratchet each fix class so it cannot recur. Every change gated (lint, smoke,
full 6-chunk suite, enterprise static ratchet, security scan, governance
lint) and pushed to main.

## Live production defects found and fixed

| Defect (live evidence) | Fix | Commits |
|---|---|---|
| 20-minute crash-restart cycle: all 12 loop-wedge dumps showed capability-discovery fsync ON the event loop; liveness sentinel SIGKILLed the tree each time (14 reboots in one afternoon) | Off-loop probes; `durable=False` write lane for probe/cache content | `d925b61a` |
| External memory sentinel died 85s after arming; runtime sat contract-CRITICAL for 2h, nothing re-armed it | Daemon-thread supervisor (loop-independent by design), bounded respawn budget, armed/exit/SIGTERM evidence lines | `d925b61a` |
| `aura_json.log` 99.7% non-JSON; stdlib records bypassed redaction entirely | JsonLineFormatter wraps every record, universal redaction, `AURA_LOG_DIR` + hermetic test logging | `0c0e2348` |
| 1,568 identical memory-spike stack dumps (55MB) in one afternoon — one per routine 20GB MLX inference | Throttle (10 min) + lifetime cap (12), all spikes still counted | `1bb01e30` |
| Live governance violations every turn: `cognitive_trace.save`, learning-example appends refused as ungoverned | `local_internal_governed_scope` established at the write sites | `1bb01e30` |
| 30,298 `GET /api/health/boot 503` lines in one launch log (⅓ of the file) | uvicorn access-log filter for health/metrics polls | `76a28b0f` |
| 18,226 stall dumps (558MB) never drained — pruning only ran on NEW stalls | Reaper crash-artifact retention sweep (batch-bounded); reaper steps moved off-loop | `40bfe3ba` |
| Background phi job timeouts escalated to EMERGENCY incidents (fail-closed module), 19×/boot | Timeouts are backpressure: cache serves, only a fully starved cycle records | `84755e8c` |
| Conversation lane cold 75s behind a background 32B generation, then worker-kill + model reload | Foreground preemption ladder: grace → cooperative soft-cancel (worker yields between tokens, model stays warm) → old escalation | `cdd7fc01` (mechanism: `6b2b0a82`) |
| 13-minute boot with zero evidence of the slow phase | Boot flight recorder: per-phase marks, real-time slow-phase warnings, summary + JSON artifact at ready | `76a28b0f` |

## Structural guarantees added (ratchets — these only tighten)

- **Async write lane** (`async_atomic_*`, `FileWriteGateway.*_async`): the 5
  per-turn hot paths converted; `tests/test_async_write_lane_ratchet.py`
  freezes 54 legacy offenders — new on-loop writes in async code fail CI.
- **Injection provenance**: `training/caa_32b_validation.py` refuses any
  live A/B artifact without `sampling.temperature > 0` and
  `injection_count > 0`. The pre-rebuild artifact was prompt theater (the
  runner's "steered" condition never injected — instance `__call__`
  assignment is bypassed by Python; greedy decoding collapsed trial
  statistics; the baseline prompt was confounded). Rebuilt runner:
  `tests/run_32b_steering_ab_live.py` + `core/evaluation/steering_injection.py`.
  Steering readiness cannot reach PRODUCTION until a provenance-carrying
  artifact is regenerated (quiet window required).
- **Unmeasurable ≠ zero** (`core/consciousness/hierarchical_phi.py`): the φ
  estimator returned 0.0 ("perfect factorization") for subsets with zero
  qualifying evidence. Below the evidence floor a partition is `inf`
  (unmeasurable) and an unmeasurable subsystem returns `None`; the
  exclusion pass skips it instead of reporting a fake zero.
- **Phi process isolation**: MIP partition search (pure-Python, GIL-bound)
  runs in a 2-worker spawn pool with thread fallback and broken-pool
  recovery — the last staged ROADMAP isolation organ. Motor cortex stays
  in-process by design (token-gated asyncio reflex loop, no native crash
  surface).
- **Interface layer verified wedge-free**: AST scan for blocking calls in
  async handlers across all of `interface/` returns zero.

## Honest-scope decisions

- Frontend: seven surfaces audited (WS lifecycle, service worker, renderer
  escaping, auth, queue bounds, DOM pruning, contracts) — all solid; no
  changes made because none were warranted.
- `chat.py` / `routes/system.py`: targeted mechanical scans only; both are
  heavily hardened and blind refactoring would be churn.

## Operator actions — both CLOSED (July 2)

1. ~~Restart the live instance~~ — restarted July 2 15:27 PDT on current
   code. `artifacts/current/boot_profile.json`: core boot **12.9s** (was
   ~13 minutes), no slow phase.
2. ~~Regenerate the steering A/B artifact~~ — regenerated with the rebuilt
   runner: 12,642 injections, sampled (temp 0.7, paired seeds), 5 held-out
   tasks, `passes_adversarial_control: true`. Full chain validated:
   `artifacts/CAA_32B_RESULTS.json` → **passed: true** with all five
   behavioral checks green (steered-vs-baseline, steered-vs-rich,
   held-out generalization, quality delta, prompt hygiene).

## Phase 3 (in progress) — full-coverage sweep to closure

### Phase 3 closure (July 2, 17:15 PDT)

Final full-suite verification at the closing tree: **6/6 chunks, ~9,900
tests, zero real failures** (one statistical order-dependence in the STDP
external-validation test — fails under sibling RNG/state in-chunk, passes in
isolation; auto-registered by the runner). All gates green: lint, typecheck,
smoke, governance-lint, security (passed=true), enterprise ratchet (zero
debt), production readiness. Evidence chains closed: CRSM→LoRA (real weight
delta serving live), CAA extraction→geometry→behavioral A/B (passed=true,
provenance-enforced). Every subsystem swept or adjudicated per the table
below; consolidations and deferrals recorded.

What lands only on the next Bryan-initiated restart: deep-narrative
backpressure semantics, windowed integrity recovery, numeric-introspection
grounding (verify with tools/report_mechanism_consistency_probe.py),
abandonment soft-cancel, and the stable-identity signed app bundle.

### Phase 4 — raw-capability tier (test-time compute + agency)

Frontier-style capability on fixed local weights, all env-gated and
fail-open to the prior path:

| Capability | Mechanism | Verification |
|---|---|---|
| Speculative decoding | 1.5B draft proposes, steered 32B verifies every token — output distribution & steering semantics belong to the target by construction. Heavy lanes only; schema/logits-processor/prompt-cache jobs excluded (both generate + stream paths). | 9 tests; **live 32B measurement (July 3, fused model): 14.91 → 20.68 tok/s = 1.39x** — as predicted, the 32B's large target/draft ratio makes drafting pay far more than the 7B test (1.13x). artifacts: specdec_32b.json. |
| Batched best-of-N | N reasoning candidates decode in ONE batched worker pass (mlx_lm batch_generate); raw candidates by design, the truth-engine verifiers select. Amplifier tries batch first, serial fallback intact. | 7 tests; verifier-filter + self-consistency + DPO harvest all operate on the cheaper pool. |
| RFT flywheel | Verifier-clean derivations (from the amplifier's RLVR harvest) → SFT dataset → proven train/fuse pipeline. SFT-on-chosen (verifier = ground truth) over DPO (mlx_lm has no DPO mode). | 4 tests; gate fails closed on <64 rows OR failed preflight (live-instance/memory) — never a 32B train beside live Aura. |
| Self-repair backlog | Chunk runner writes a machine-readable defect register; ingestor turns it into approval-gated, shadow-planned repair goals for the (already mature) autonomous task engine. | 6 tests; requires_approval=True + is_shadow=True, idempotent, read-only ingestion. Aura's regression detector now feeds her own governed self-repair. |

The compounding loop is now closed end-to-end: **speculative + batched
decoding buy test-time compute → best-of-N reasoning → verifiers select →
verified derivations become weights (flywheel) → better drafts → repeat**,
and detected regressions route to governed self-repair.

### Phase 5 — expressive agency (does she KNOW to use her mind?)

Bryan's question: does Aura choose to show/demonstrate/ask/model from general
cognition, not scripts? What existed: real capabilities (FLUX imagination,
outcome simulator, subjective-choice engine with preference consistency,
vision client, autonomous task engine, capability_map proprioception) but no
GENERAL layer letting the mind decide among them mid-conversation — decisions
were trigger-routed, not reasoned.

- `core/cognition/expressive_affordances.py` + `affordance_realizers.py`: a
  decide-then-realize layer. The mind reads a menu of its own affordances
  (framed as self-knowledge, "the way a person knows their own hands are
  available"), emits an intent tag by its own judgment, and a governed
  realizer delegates to the existing subsystem. Five built-ins: show_sketch,
  demonstrate_artifact, request_media (know-to-ask), model_scenarios
  (sim + preference-consistent choice), deep_examine. Extensible by one
  register() call — no routing code. Live chat lane wired (env-gated menu
  injection + post-generation realization), 11 tests.

### Phase 5 regression caught (live-path reliability)

Re-verifying the conversation lane after the affordance wiring surfaced a
real defect introduced by an earlier commit (e7aa6efe's identity-continuity
grounding): `_is_identity_request` matched "what are you talking about?" (a
contextual-relevance CHALLENGE) via a bare "what are you" substring, so the
desktop quality gate rebound a hallucinatory "voices whispering in my ear"
reply to a canonical identity answer and returned 200 instead of failing
closed with 503. Fixed at the source — a relevance challenge is not an
identity request, and "what are you <gerund>" is topical, not identity.
Conversation lane green (228 tests). This is exactly the "no incoherent
output escapes the gate" reliability goal; caught only because the monolith
was re-verified after editing.

### Phase 5 additions — expressive agency, deepened

- **Deterministic artifact builder** (`core/actuators/artifact_builder.py`):
  build_table (CSV + styled HTML — opens in any spreadsheet OR any browser, no
  app required), build_doc (HTML + Markdown), build_program (runnable file),
  open_artifact (governed). The always-succeeds floor under
  demonstrate_artifact: verified live that when task-engine model
  decomposition was unavailable, it STILL produced a real openable table and
  opened it — Bryan's "recognize Excel might be missing, build one herself and
  export in a showable format." 9 tests.
- **Preference-consistent choice surfaced**: model_scenarios exposes the
  subjective-choice engine's preference_override — True exactly when her
  learned preferences overrode the drive-pick — and cites alignment scores.
- **Self-knowledge unified**: the always-on operational-self context now
  truthfully lists the expressive capabilities, so she knows (and can say) "I
  am not limited to words" even before the action-grammar menu is enabled.

**Honest gate**: the action-grammar menu injection stays env-gated
(AURA_EXPRESSIVE_AFFORDANCES=1). The mechanism is built, tested, fail-open,
and bounded (≤3 actions/turn, tags stripped). Enabling it changes every
turn's generation context, and the chat quality gates are sensitive (see the
identity-classifier regression this same phase) — so first live validation
belongs to a watched session, not a silent default. It is a one-variable
switch and the recommended next live experiment.

### GPU window (July 3) — live 32B validation, honest findings

Ran the pending 32B jobs with the instance up:
- **Speculative decoding, live on the fused 32B: 14.91 → 20.68 tok/s = 1.39x**
  (predicted; the large target/draft ratio pays). artifact: specdec_32b.json.
- **Affordance judgment probed live (2 runs): emission = 0.** The 32B does NOT
  adopt the ⟦affordance:⟧ grammar zero-shot even at max recency + a worked
  example. Honest conclusion: the mechanism is safe/wired but DORMANT; reliable
  use needs training (the flywheel), not prompting. **Reverted the menu to OFF
  by default** — enabling it was pure cost (prompt bloat, latency) for no
  benefit. Mechanism + desktop-objective gate stay.
- **Real daily-use bug found & fixed** (surfaced by the probe, independent of
  affordances): brief social turns ('thanks') drew a servile reply whose tail
  the quality gate correctly rejected → retries exhausted → ZERO tokens →
  parent inline-retry storm → sustained lag → a clean self-preservation
  shutdown (no crash, snapshot frozen, exit 0 — the safety system working).
  Fix: repair_generic_assistant_language salvages a brief clean reply for
  brief turns; the worker salvages on exhaustion instead of yielding empty.
  Live-verified: 'thanks' → "I'm glad it was helpful!" (clean, no storm).
- **Flywheel delta: not run** — 0 verified-preference rows accumulated yet
  (the reasoning amplifier needs live runs first); the gate correctly fails
  closed. Honest: ready, waiting on data.

Crash-watch discipline throughout (Bryan's ask): monitored neural stream +
loop-wedge/stall dumps live; 0 wedges, the one shutdown was graceful.

### Sweep ledger

| Area | Depth | Findings / actions |
|---|---|---|
| core/orchestrator + aura_main (tier 1) | AST hunts (broad/bare except, sync sleep/IO in async, untracked create_task): zero; 25 swallow sites inspected — all narrow-typed with intentional-no-op comments (optional telemetry seams, per-PID reaper continues, boot-time executor fallback); boot/shutdown paths already instrumented (flight recorder, drain bounds, boot-tail phases) | No defects; adjudications recorded. Contract surface green (48 tests). |
| core/memory (tier 1, 94 files) | AST hunt: 6 swallow sites, all inspected — queue-timeout worker continues, per-item skip of malformed records, best-effort optional stats/signals (correct patterns); WAL mode verified (episodic + vault), writer queue has flush_and_checkpoint + bounded drain batches; recall path previously line-read (gateway index rebuild) | No defects. Battery green (facade/index/sentinel 47 tests). |
| core/consciousness (tier 1, 140 files) | AST hunt: **zero** broad/bare excepts, sync-IO/sleep-in-async; deep-narrative backpressure fixed (streak/starvation semantics); report-vs-mechanism probe rerun live (still fast-path on pre-fix instance) → routing defect closed END-TO-END: numeric state requests now classify as internal-state reads and the grounded reply leads with parseable live substrate values | The phenomenal-measurement seam is now mechanically closeable: probe → grounded lane → substrate numbers → longitudinal consistency artifact. Verifiable on next live restart. |
| governance/runtime/resilience/security (tier 1, 221 files) | AST hunt: **zero** findings; Will + governance ablation + effect closure + auth/runtime hardening batteries green (427 tests); governance-lint clean; security scan → one finding (hardcoded WS handshake nonce in the live probe) fixed properly (random RFC 6455 nonce) → gate passed=true | Fail-closed adjudications already encoded in the fail-closed module list + ablation suite. |
| tier-2 areas (agency, autonomy, senses, voice, perception, skills, tools, executors, interface, agi, adaptation, learning, world_model — 362 files) | AST hunt (bare/broad except, sync sleep in async): **zero** findings; behavioral coverage delegated to the full-suite final verification | Static hygiene at flagship level across the entire tree — the ratchets did their job. |
| core/brain/llm (tier 1) | mlx_client hot paths line-read (init, spawn, listener, wait/SLA ladder, abandonment, reboot, warmup, owner/cancel); mlx_worker token loop + dequeue + emit line-read; AST hunt (bare/broad except, sync sleep/IO in async) over all 40+ brain modules: **zero hits**; substrate ODE NaN/rollback verified | Abandoned generations now soft-cancel the worker (no 32B reload on recoverable SLA breaches); reboot resets cancel channel + request seq; warmup fg/bg duplication adjudicated: keep (stability-annotated). |


Scope owned by the improvement pass (tasks tracked in-session):
evidence-chain closure ✅ · backpressure degradation audit (the
stream_of_being deep-narrative fix is the template) · tier-1 line sweeps
(brain/llm, orchestrator+boot, memory, consciousness, governance/runtime/
resilience/security) · tier-2 structured sweeps (agency, senses, skills,
interface) · duplicate-system consolidation with written adjudications ·
docs/claims truth pass · final full-suite + gates verification · the
evidence-grounded qualitative assessment.

---

# Phase 2 — July 2, 2026 (same pass, expanded standard)

## Ratchets drained to zero
- **Async write lane: allowlist EMPTY.** All 66 legacy on-loop write sites
  across 42 files converted to `*_async`; 18 test fake gateways grew async
  delegators; the scanner recognizes `to_thread(lambda: ...)`. Any future
  entry is a regression, not debt.
- **Enterprise debt baseline: ZERO in every category.** The last baselined
  findings were the anti-mock auditor flagged for naming its target; the
  detector now exempts audit/detector vocabulary and the baseline allows
  nothing.

## Foundation
- Generated subsystem census (`tools/subsystem_census.py`) with
  ISOLATED/LEAF/HUB/UNTESTED verdicts, dynamic-import visibility, and
  degradation-hygiene ranking. ISOLATED subsystems 4 → 0 (two archived
  with history, two rescued from false-dead by string-import scanning).
- Deliberate deferral, documented: the 177-file core-root reorganization
  (breaks thousands of imports for structural aesthetics).

## Live-runtime verification (the real benchmark)
- `tools/live_surface_probe.py`: read-only live verification (health,
  pulse, shell, assets, WebSocket, latency). Caught a real degradation on
  first use: conversation lane cold + mind_tick dead while transport
  stayed green.
- **Immune-system deadlock fixed**: the healer deferred mind_tick's repair
  66 times because failure-lockdown blocked the very repair that would
  clear the lockdown. Contract-subsystem repairs now proceed through
  lockdown (cooldown intact; luxury repairs still defer).
- Live relaunch on fixed code: probe full-PASS, 12.9s boot, authorized
  chat turn end-to-end in 13.1s with healthy contract.
- Boot flight recorder sub-marks: the orchestrator tail measures ~0s; the
  ready-path (model-lane warmup wait + manifests) now reports separately.

## High-end goal pieces (verified, receipted)
- **NetHack live proof: PASS** — 60/60 turns in the real game through
  Observation/CommandSpec with per-action receipts
  (`artifacts/environment/nethack_live_proof.json`).
- **Report-vs-mechanism consistency probe** built with longitudinal
  artifact; first honest verdicts: self-reports were NOT substrate-
  grounded — chat fast paths intercepted introspection. The reflex layer
  now yields to substrate-read requests
  (`is_substantive_introspection_request`); the deeper task-intake
  misclassifier (math questions answered with task tickets) is precisely
  reproduced and queued as its own fix.

## Ecosystem maturity
- One hashed lockfile, documented `--require-hashes` install path.
- Root debris removed; first chat.py decomposition slice landed
  (`chat_quality.py`) with alias-preserving re-exports.
- `make restore` refuses to overwrite a running instance's state
  (FORCE=1 override) — restoring under a live runtime was silent
  corruption.

## Reliability stack: audit → zero-debt → learning → causal (July 3)

Bryan built the aerospace/reliability hardening layer (fault taxonomy,
FMEA, TMR, contracts, SLO monitor, verified state machines, tracing,
chaos/canary/rollback, diagnostics endpoint, CI gate). Audited to the
standard it names, defects fixed wholesale, then extended:

### Audit findings fixed (commit 44805209 + prior fixes in 3ed7e668)
- **Two guaranteed self-deadlocks**: `FaultRegistry.status()` →
  `rpn_report()` and `ChaosFramework.status()` → `pass_rate()` re-acquired
  their own non-reentrant locks. Empirically pinned: the test suite hung
  at test 8 (`test_status`), and one GET of the diagnostics endpoint would
  have wedged the live event loop thread.
- **Severity fidelity**: `record_degradation`'s severity map was built and
  dropped — critical degradations recorded as MARGINAL faults. Explicit
  severity override added to `record_fault`.
- **Dead SLOs**: `error_rate_per_1k_requests`/`substrate_resets_per_hour`
  compared each event's value (1.0) to a count target — mathematically
  unable to violate. New `count_per_window` aggregation makes occurrence
  SLOs real (renamed `error_events_per_hour`).
- **Alert storms**: persistent SLO violation fired a CRITICAL alert per
  sample on hot paths (Will decisions target 5ms). 60s per-SLO cooldown.
- **Thread-safety**: SLO tracker deque mutated during status() iteration
  (RuntimeError under load) — per-tracker lock; VSM ran guards/actions
  under a non-reentrant lock (reentrant guard = deadlock) — RLock.
- **Mock rollback**: `rollback()` loaded state, logged "restored", applied
  nothing, returned True. Now: registered state applier or fail closed.
- **Broken traces**: per-span sampling re-rolls orphaned children; now
  head-based (root decides, descendants inherit, force_sample survives).
- **Traceability theater**: all 8 runbook references pointed at
  nonexistent files; 3 FMEA mitigation paths were wrong. Fixed + 3 new
  runbooks written (worker-crash, shutdown-hang, orphaned-tasks) + a test
  that fails if any catalog reference stops resolving.
- Full gate sweep to zero-debt: every broad except narrowed to the
  explicit guarded-callable envelope; tracing records in-flight
  exceptions via sys.exc_info in finally (catches CancelledError too,
  no false ERROR when a span opens inside an except block).

### FMEA that learns (68b1538e)
`core/resilience/fault_evidence.py`: cross-boot occurrence evidence
(governed atomic persistence, O(1) hot-path recording), MIL-STD-882E
band mapping from observed rates with a sufficiency gate, and a
probability-drift report with recalibrated RPNs —
`/api/diagnostics/reliability/drift`. The static hazard catalog is now
checked against measurement instead of staying a frozen guess.

### Causal wiring (857cc629)
- ShutdownCoordinator walks the verified shutdown lifecycle FSM; a
  re-entrant shutdown() (previously: warning + double-running every
  teardown handler) is now a recorded F17 and a refused duplicate.
- HTTP root spans on every /api request (middleware) with
  inference.generate child spans in mlx_client — slow turns read as one
  connected trace at /reliability/traces.

Remaining in this arc: recovery executor (faults actuate their
RecoveryStrategy) — deliberately deferred to integrate with the immune
system rather than duplicate it, blocked on the parallel agent's
in-flight immune_system.py edits.

## Knowledge substrate populated + search resilience (July 3, evening)

- **Local corpus LIVE**: full English Wikipedia ingested — 7,189,653
  pages processed, 6,588,092 articles indexed, 35.6GB FTS5 corpus, 96
  minutes, one command, resumable (survived multiple session restarts
  via the detached download supervisor). Warm queries ~90ms with real
  relevance (relativity → Hafele–Keating; Kalman → Schmidt–Kalman/EKF).
- **A live autonomy failure Bryan observed** (web_search FAILED during
  the 'Maintain and repair Aura' default goal) root-caused to the
  planner hardcoding web_search into every tool menu whether or not the
  skill existed. Three-layer fix (417ece86): local_reference_search
  first-class skill (offline, provenance-tagged, honest misses), planner
  tool-menu honesty + shortcut fallback, and runtime web→corpus
  degradation with offline_fallback provenance. Search intent can no
  longer dead-end in this system.
- **Order-dependence class extinct**: suite-wide verdict went from 22
  order-dependent failures to ZERO via one conftest guard
  (snapshot/restore of registry resolvers — tests were erasing the
  container-backed resolver during 'cleanup'). Real failures 22 → 4 → 0:
  the last four closed with real fixes, not markers (db2371bd) — a
  welfare-coverage contract that had been satisfied by a DEAD
  ActionExecutor import now wraps live search in an actual
  WelfareTransaction; a new skill's sync write inside async execute()
  moved to the async atomic lane; catalog ratchet updated 67→70 for
  three legitimate new skills; one more environment-dependent RSI test
  pinned.
- Remaining for the substrate arc: rebuild-and-swap refresh mode for new
  dumps, and retained-web-knowledge writeback (the continuous-growth
  path), then the flywheel distillation bridge.

## The live hello-turn cascade — root-caused end to end (July 4)

Bryan launched four times; three times "hello" got silence. Each launch
peeled one real layer (all fixed, all pinned):

1. **Proof-tool collision** — a proof-bundle generator left running
   from the previous night boots its OWN runtime; the port/registry
   collision SIGTERM'd the live instance mid-first-conversation.
   Operational rule recorded: no proof/decisive/certify tooling alive
   when Bryan may launch.
2. **VAD latch** (voice root cause): `is_speaking` never reset, so
   end-of-utterance never fired and user speech was discarded by the
   buffer wipe. One-line silence-timeout un-latch in voice_engine.
3. **Presence-check gate**: "can you hear me?" → "I hear you." was
   killed as a placeholder reply by the quality gate; presence-check
   acknowledgments exempted (ce3fcf0a).
4. **Coherence-lockdown cascade** (7841cadc): the binding engine
   blended a zero-evidence unity score into coherence → 0.00 →
   executive Rule 8 blocked EMIT_MESSAGE. Fixes: no-evidence unity is
   skipped (unmeasurable ≠ 0), Rule 8 degrades user-facing speech
   (bounded reply) instead of muting, watchdog got exc_info.
5. **Dead gate leases** (6856cbf1): force-aborted generations leaked
   their lease → every later turn queued forever (the overnight death).

Fifth launch: working multi-turn voice conversation, verified live.

## Grounded self-knowledge + capability honesty (July 4–5)

- **Self-forensics** (873c7229): asked about her own crashes she now
  answers from black boxes (shutdown grace flag, sentinel tail, crash/
  stall artifacts, live incidents) injected as evidence, with a gate
  (`ungrounded_self_cause_claim`) that rejects fluent causal stories
  lacking evidence markers — built the night she blamed her death on
  electromagnetic interference.
- **Capability map** (b51ea495): actionable requests get a lane
  decomposition (filesystem/scripting/GUI) with "never decline the
  whole task" — fixes the declined dinosaurs-note task.
- **Recovery bridge** (841a914e): fault records now actuate their
  cataloged RecoveryStrategy — AUTO lane through the immune system,
  OPERATOR lane as runbook-linked recommendations. Detection → action
  loop closed without duplicating the immune engine.
- **Sensorimotor grounding** (0a0eb234): consequential tool calls open
  an outcome-ledger expectation, execute, then verify reality with her
  own senses; a tool claiming success without the predicted effect is
  a recorded ACTION-CLAIM-MISMATCH (the confabulated-action class).
- **Belief reconciliation** (7bf369da): contested beliefs resolve
  (affirmed/retired, with evidence) and age out of the autonomy gate
  (6h freshness) instead of wedging epistemic_reconciliation forever.
- **Corpus continuous growth**: retained-web writeback (verified
  research accretes into the local corpus, deduped) + `--rebuild`
  atomic swap for new dumps. Ablation legibility shipped reviewer-
  runnable (f89ceac7) — the #1 external-review deduction.

## Interface presence (July 4–5)

Two CSS-only passes on the darker ground Bryan preferred (surface
ladder 0.012→0.055): affect-driven presence tokens on <body> (living
hue from curiosity, inner light from warmth, breath from fatigue),
values-lead stat strip, cluster seams, breathing presence card,
68ch measure, splash fade (59a90c0d). Shell contract re-pinned to the
current lane expression after the parallel agent's refactor (cfcc5c5e).

## Closeout

- Qualitative assessment: docs/WHAT_IS_AURA_2026_07.md — what she is,
  every claim receipted, every ceiling stated.
- Remaining, by design: #33 72-hour soak (excluded by Bryan for now);
  #37 in-repo proof bundles (generator boots its own runtime — needs a
  ~30-min quiet window with the live instance parked).
