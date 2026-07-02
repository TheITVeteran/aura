# Improvement Pass — July 1–2, 2026

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
