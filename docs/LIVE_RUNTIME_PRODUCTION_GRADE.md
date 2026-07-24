# Live Runtime Production-Grade Program

Status: ACTIVE (opened 2026-07-24, Bryan-directed)

The mandate, in Bryan's words: the backend is enterprise-grade; the live
desktop runtime — the thing users actually see — is prototype-grade and
is the #1 thing that breaks. Bring it to production grade in every
dimension: "fix all that is broken, improve all that needs to be
improved, build anything to fill in any gaps, standardize, make things
boringly reliable." Opening Aura should be a smooth, ordinary software
experience; if something doesn't work it should be because the feature
isn't mature, never because the runtime is broken.

## Operating standard (what "production grade" means here)

1. **Nothing blocks the event loop.** No sync filesystem walks, DB
   fetches, or fsyncs on the loop — ever. The async-write-lane ratchet
   covers writes; loop-affine READS get the same treatment as they are
   found (admission, health refresh, snapshot builders).
2. **Every registry is bounded.** Any dict/list that accrues per-event
   entries (procs, threads, records, caches) has an eviction policy and
   a test that proves it. Unbounded-by-design is a defect.
3. **Fail-closed paths are proven end to end.** A safety mechanism that
   only logs is not a mechanism. Every CRITICAL/refusal path needs a
   test that asserts the record/refusal actually lands (the
   runaway-recorder lesson: it had never fired once in production).
4. **Latency is architectural, not tuned.** Per-turn cost must not grow
   with conversation length (KV reuse), and watchdog timeouts must be
   derived from measured operation envelopes, not vibes.
5. **The error log is the backlog.** `~/.aura/logs/aura_json.log`
   error/critical clusters and `data/error_logs/` (stalls, crash,
   memory) are mined each session; every cluster either maps to a
   registry item below or gets one. Symptom suppression is forbidden.
6. **Bounded verification before "fixed".** Each repair gets a bounded
   app-down reproduction or a live-log confirmation criterion stated in
   advance; "should help" doesn't close an item.

## Defect registry

Frequencies are error/critical clusters from the last live sessions
(Jul 17-23 `aura_json.log`) unless noted.

| id | defect | evidence | status |
|---|---|---|---|
| LRP-001 | Conversation path can never reuse KV: 32B cache budget 0 under desktop guard AND clean_user_surface bypass+clear per turn → whole-history re-prefill each turn → the 15-turn latency wall | endurance-0715-clean staircase 11s→216s; cp receipts | **FIXED** 14a7a554 (budget 2 + 6144-token cap + scoped reuse). Live confirm on next restart; then check prefix stability of the rendered head in worker logs |
| LRP-002 | Model-load admission stat-walks the model dir ON the event loop under saturated IO → 5.5-8.6s loop stalls during reloads | stalls/stall_1784673149, _1784675621 | **FIXED** 14a7a554 (memoized + to_thread) |
| LRP-003 | RuntimeHygiene retained every Popen/Thread ref until shutdown → the Jul 7 soak "leak" (16k io.open + 21k TextIOWrapper live) | longevity_leakrepro tracemalloc | **FIXED** 25898416 (evict finished, bounded records) |
| LRP-004 | RUNAWAY fail-closed degradation never recorded (Severity.CRITICAL on a typing Literal → AttributeError swallowed) | "Could not record runaway degradation: CRITICAL" ×2 Jul 21 | **FIXED** this wave + end-to-end test |
| LRP-005 | Inference-gate timeout cascade: reflex ×47 + brainstem ×25 timeouts, "local paths exhausted" ×51, UnitaryResponsePhase breaker ×29, failed-closed replies ×37 | log clusters Jul 18-20 | OPEN-VERIFY: hypothesis = mostly downstream of LRP-001; after restart, re-mine the same clusters; whatever survives gets its own root-cause pass |
| LRP-006 | Event-loop/tick stalls beyond LRP-002: TICK STALL ×55 live; boot-phase 5s loop-stall storm reproduced in the Jul 24 verification soak | StabilityGuardian dumps; scratchpad soaklogs stalls | OPEN: attribute every loop-thread frame in the dumps; fix each on-loop offender per standard #1 |
| LRP-007 | External memory sentinel crash-loops (12 respawns/h, protection DOWN) and its log records carry 1970 timestamps (monotonic-as-epoch somewhere in the emitting path) | ×17, "1970-01-12" stamps | OPEN: root-cause sentinel deaths + fix the timestamper |
| LRP-008 | Governance refuses the adaptation pipeline's success-trace writes (`append_text:adaptation.finetune_pipe.dataset called outside governed context`) → learning data silently dropped | ×12 Jul 18 | OPEN: wrap the writer in `local_internal_governed_scope` (or route through the gateway's governed lane) |
| LRP-009 | RAM crisis machinery still firing: NEURAL PURGE ×19; `worker_survived_kill` ×8 (kill+join failed, caller state ambiguous) | log clusters | OPEN: expected to shrink with LRP-001 (fewer watchdog kills); then harden kill-escalation (process-group kill, reap proof) |
| LRP-010 | SLO budgets exhausted: boot_cold_p95 ×12, error_events_per_hour ×19 | SLO ALERT clusters | OPEN: re-baseline after waves 1-2 land; boot p95 needs its own profile pass |
| LRP-011 | Desktop reply lane degraded outputs: "compact desktop generation returned no usable text" ×8, zero-token clears | log clusters Jul 20 | OPEN: verify against LRP-001/005 first; then root-cause the residual |
| LRP-012 | 3 conversation-lane tests fail at origin/main (reply-stabilization wording drift) | pytest, verified pre-existing Jul 24 | OPEN: spawned as its own task chip |

## Cadence

Each session on this program: (1) pull the latest error/critical
clusters and update the registry; (2) fix the highest-evidence OPEN
item root-cause-first with tests; (3) bounded verification; (4) push the
checkpoint to origin main and update this doc's statuses. The march is
done when a full live session (boot → hours of conversation → shutdown)
produces zero error-level clusters that aren't feature-maturity gaps.
