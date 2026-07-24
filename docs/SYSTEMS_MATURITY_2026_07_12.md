# Systems Maturity Pass — 2026-07-12/13

Bryan's mandate: Chrome/Kubernetes/aerospace-grade attention to detail,
stability, and daily-user reliability — same pass until 100%, soaks last.
Starting point: the reliability roadmap (K1-K6, A1-A5, C1-C4) had landed
and `make triage` showed **0 hard deaths in 7 days**. Firefighting was
over; this pass raised the engineering floor.

## What landed

| Area | The defect it kills | Proof |
| --- | --- | --- |
| Launcher log + .env rings (facb808d) | desktop-launch.log grew unbounded to 145MB; .env had NO recovery path (destroyed once) | sandbox behavioral tests: rotation gate, gzip ring x5, env ring x10 collision-proof |
| `make preflight` (1425a29e) | every historical "won't boot" that wasn't code was environment (missing .env, disk, stale model path, port squatter) | stdlib-only by contract; first real run caught genuine lock drift (tiktoken) |
| `make audit-deps` (f79d58c9) | 5 known CVEs invisible in the venv | 4 fixed; torch CVE-2025-3000 waived with a SELF-REVOKING gate (greps for the unused vulnerable call) — docs/DEPENDENCY_AUDIT.md |
| Verified backup (7ea2c0c6) | `make backup` swallowed errors, raw-copied hot WAL stores, never verified, 6 weeks stale | real run: 1875 files, 28 consistent DB snapshots via sqlite backup API, sha256 manifest, restore-verify green |
| Integrity monitor rewrite (6254f858) | full page scans (~700MB) every 5 min over only 8 of 28 stores; identity/conversation stores unmonitored | 28/28 stores, quick_check on flag-controlled cadence, corruption = STATE carried across cycles |
| Chaos injector honesty (0b34926b) | docstring promised 10 faults, registry had 6 | parity pinned by test; apply→restore round-trips drillable (AURA_CHAOS_RESTORE_SECONDS) |
| mypy strict ratchet (13b6afbd) | strict typing covered 10 files with no growth mechanism | 31 files then, 71 now (parallel session adopted the ratchet); min-count pin, in-process enforcement, negative-proven |
| Live-ledger hermeticity (e3cc3b4a) | unit tests wrote fixture pins into the REAL ~/.aura/data — 445 phantom memories Aura could recall as genuine | 3 tests redirected; live ledger cleaned (745→300, Bryan's real pins untouched); tests/live_data_guard.py autouse write-hook, negative-proven |
| Observation-seam test ports (e3cc3b4a) | 12 tests injected legacy psutil doubles the census no longer reads | ported to SimulatedResourceObserver / host_observation markers |
| Immune loop-freeze fix (61c7c60d) | numpy + periodic PCA ON THE EVENT LOOP per immune event — froze the runtime 5.3s under degradation storms (caught live by the soak + narrator with receipts) | present_antigen via to_thread; 47 immune tests |
| Order-dependence roots (6d8011df) | module-global singletons latch test doubles past ServiceContainer.clear() — one disease, multiple organs | personality-engine + backpressure reset seams, conftest wiring, latch-mechanism contract test |
| Leak attribution surface (6d8011df) | totals said memory grows; nothing said WHERE | runtime_hygiene.allocation_growth() snapshot diffs + /api/system/memory/growth; proven on a deliberate 22MB leak |

## Verdicts

- **Full 6-chunk certification**: GREEN at e3cc3b4a for all of this pass's
  code (chunks 1-4 zero failures; remaining failures were the moving
  tip's unreconciled parallel-session debt, each fixed at its root in
  bb7d8822's lineage).
- **Memory leak (H1 vs H2): H1 CONFIRMED.** With no model loaded and zero
  conversation, the bare kernel grew **275MB/h at true idle**
  (867→1071MB / 44.6min). July 7's 242MB/h "under load" was the same
  leak. Attribution run: see leak_attribution addendum below.
- **200-turn endurance**: blocked twice by regressions the soak itself
  caught (the immune loop-freeze — fixed; then a candidate-readiness
  cluster in the actively-rewritten model-lane code, where every tier
  fails `candidate_worker_not_ready` — handed off with receipts, it
  predates and is orthogonal to this pass). The cleanest endurance
  evidence remains 2026-07-12's partial on a stable tip: p50 12.4s,
  zero deaths. A full 200-turn PASS on a stable tip is still owed.
- **Order-dependence register**: 6 of 7 root-fixed; 1 reclassified as a
  nondeterministic shutdown-census race (two occurrences, zero replays
  on identical sets) and chipped with reproduction guidance.

## Leak addendum (2026-07-13): H1 confirmed — a real idle leak

The H1-vs-H2 discriminator has its verdict. Two independent quiet-box
idle windows, computed from the soak drivers' RSS trend CSVs:

| run | idle window | idle slope | load slope |
|---|---|---|---|
| 2026-07-12 23:09 | 45 min, 90 samples | **+454 MB/h** | (n/a, 2 samples) |
| 2026-07-13 02:08 | 70 min, 140 samples | **+354 MB/h** | +379 MB/h |

The orchestrator process (model workers excluded — n_procs=1, sub-GB
RSS) grows at ~350–450 MB/h while doing NOTHING but its own idle
cognition, and the slope under load is statistically the same. That is
**H1: a real leak**, not proof-deferred reclamation under load (H2
falsified — reclamation pressure changes nothing because load isn't
the driver). July 7's "242 MB/h under load" was this same leak.

### Attribution + fix (2026-07-13, b5d69941)

A 66-minute tracemalloc idle window (worktree instance on :8001,
`AURA_RUNTIME_HYGIENE_TRACEMALLOC=1`, read through the new
`/api/system/memory/growth` allocation-diff surface) resolved the "where".
The verdict is instructive: the **Python heap was essentially flat**
(~18 MB/h, and every buffer involved is a bounded `deque(maxlen=…)` — the
Python side is already good engineering) while RSS climbed ~350 MB/h. That
gap IS the diagnosis — the leak is **native**, invisible to tracemalloc
because the Python objects are freed. Every top native allocation site
routed through `Popen`.

Root cause: the **desktop perception loop** shells out ~4 times per cycle
(osascript window-focus, pbpaste clipboard, browser-tab enumeration,
`ps -A`) at a 2 s cadence — ~7,000 subprocesses/hour **at idle**. On macOS
that fork/exec churn grows the parent's malloc arena and the OS never
returns it. The loop's own comment already called subprocess churn "the
dominant source of RSS churn" and made the *headless* path dormant — but
the *desktop* path Bryan runs daily kept storming.

Fix: activity-gated backoff (`AURA_PERCEPTION_IDLE_BACKOFF_AFTER_S`,
default 120 s, reusing the existing idle threshold). Full 2 s cadence
while the user is active; dormant 45 s cadence once idle — a ~35× cut in
idle subprocess spawns, worst-case ~45 s latency to notice the user's
return. The file-mutation scan window tracks the effective interval so
backoff never drops a change. 3 contract tests.

The general lesson for the codebase: **a flat tracemalloc heap next to a
climbing RSS is a native-churn signature** — look for subprocess/`Popen`,
mmap, or C-extension allocation, not a Python object leak. The
`/api/system/memory/growth` surface makes that diagnosis a one-call
operation on any future instance.

VERIFIED (2026-07-13/14): a clean 58-minute orchestrator-only idle window
on the fixed code ran RSS **1669→1698 MB, +30 MB/h — essentially flat**,
against +350 MB/h before the fix (~92% reduction). A partial earlier
window agreed (−332 MB/h). Sampler caveat: measure the ORCHESTRATOR
process, not the tree — the deferred 32B prewarm loads ~20 GB of MLX
worker RSS near window end, which is the model becoming resident (a
step), not the leak (a ramp); filter by RSS threshold or exclude the
worker children. A final multi-hour production-duration soak on the
sealed tip would confirm the fix holds under sustained load.

## CRSM→LoRA loop honesty (2026-07-13, 82f5c373)

The live health poll reported "proof integrity degraded: CRSM→LoRA loop
OPEN (33 captures untrained)" persistently. Running the closer exposed the
truth: all 33 captures are idle self-reflection moments (`<thought>`/
`<action>` tags, "will-approved self-reflection") that the training safety
gate rejects by design — you never train a model on its own control-plane
structure. The loop could never close, and the monitor was counting *raw
lines* as "untrained captures." Fixed: the monitor now gates OPEN on
*eligible* captures through the trainer's own gate (cached by dataset
sha256), reporting honest "idle" when nothing is trainable. Also built the
autonomous closer (`crsm_closure_scheduler.py`) for when real eligible
experience accumulates. The closer is enabled by default and can be
explicitly disabled with `AURA_CRSM_AUTOCLOSE=0`; execution still requires
the normal Will, authority, resource-admission, and model-lane receipts.

## Skill framework: three "technically-true" gaps (2026-07-13)

An audit of the 93 skills — prompted by "are any missing the attention to
detail a casual user or engineer would expect?" — found useful framework
controls (pydantic input validation, governance receipts, per-skill circuit
breakers, standardized error results, and a zero-stub gate). It also found
three cross-cutting gaps. These framework fixes establish a stronger baseline;
they do not replace per-skill causal, idempotency, and live-behavior
certification.

| Fix | The defect | Commit |
| --- | --- | --- |
| Retry double-fire | `safe_execute` retried transient failures 3× *unconditionally* — a skill that sent/acted then hit a transient error re-fired it (at-least-once where the user expects at-most-once). Retries are now explicit opt-in through `retry_safe` and are always disabled for approval-gated execution; each opt-in still requires an idempotency review. | 2e32b4ce + merge hardening |
| Dishonest success | `_infer_ok_flag` missed the `success` key (11 skills use it) — `local_reference` returning `{success: False, "corpus empty"}` was marked `ok=True`, so Aura believed she'd consulted her knowledge successfully. | c40f9f97 |
| CRSM loop honesty | (above) the monitor counted raw lines, not eligible captures. | 82f5c373 |

Each is the "actions fire but shallow/technically-true" pattern — caught
at the framework altitude so all 93 skills inherit safer behavior. The
remaining per-skill review and external live-effect certification stay open;
an internal integrity self-audit is evidence, not independent closure proof.

## Endurance soak (2026-07-24): endurance PASSES, retention finds real bugs

The 200-turn endurance run finally executed cleanly on the current tip —
the July `candidate_worker_not_ready` model-lane cluster that blocked it
was resolved by the parallel session's rework (verified: fresh boot to
conversation-ready in ~2 min, real turn served correctly).

**Endurance dimension: PASS.** 200 turns, **deaths=0, hijacks=0,
p50≈2–3.6s** (July's partial was p50 12.4s), thermal 0, and the
orchestrator RSS was flat-to-lower across the run (turn-100 < turn-10) —
the idle-leak fix holds under sustained load too. A re-run confirmed the
same shape (deaths=0, p50≈2s through 100 turns).

**Retention dimension: the soak did its job and surfaced real defects.**
The verdict was FAIL on `retention=0/3` — a fact planted early could not
be recalled later. Root-caused into two distinct defects:

| Defect | Root | Status |
| --- | --- | --- |
| Recall window | `_find_session_content_exchanges` searched only the latest **40** turns; a fact planted at turn 3 and probed at turn 111 was still in the 500-cap log but outside the window — so a long conversation silently "forgot" it. | **FIXED** (1b91eeabf): searches the full retained session; 2 tests, negative-proven. |
| Plant storage | The plant turns themselves didn't store: the first heavy turn hit a cold-32B-load timeout, and two others hit `canonical_chat_no_reply` — the quality gate rejected the drafts on a coupled reason set with no salvage. Because the facts never stored, the (now-fixed) recall can't find them. | **Diagnosed + chipped** — it lives in the parallel session's actively-developed response-reliability verifier, and the obvious "add to the deliverable set" fix would regress a legitimate hard-rejection of shallow technical non-answers. Fixing it right needs detector-precision work, coordinated with that session. |

The honest boundary: the **endurance claim** ("hundreds-of-turn
conversations, no degradation") is proven. The **retention feature** has
one real fix landed and one deeper defect precisely diagnosed and handed
to the session that owns that code — not blind-patched into a regression.

## The meta-lesson

Every blocking failure this pass found reduced to one of two shapes:
**process-global state without a reset seam** (singleton latches,
lease counters, shared live files) or **work on the event loop that
belongs off it** (PCA, page scans, fsync). The ratchets and reset seams
landed here make both shapes harder to reintroduce than to avoid. The
endurance soak added a third: **a bound sized for the common case that
silently drops the tail** (a 40-turn recall window in a 500-turn log) —
correctness that only a long run exposes.
