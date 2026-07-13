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

Still owed: ATTRIBUTION — the tracemalloc idle window
(`AURA_RUNTIME_HYGIENE_TRACEMALLOC=1`, read via
`/api/system/memory/growth`) names the subsystem; the 2026-07-13 05:50
attribution attempt aborted (`never_ready`, the candidate-readiness
cluster). It reruns with the next quiet idle window, before the fix.

## The meta-lesson

Every blocking failure this pass found reduced to one of two shapes:
**process-global state without a reset seam** (singleton latches,
lease counters, shared live files) or **work on the event loop that
belongs off it** (PCA, page scans, fsync). The ratchets and reset seams
landed here make both shapes harder to reintroduce than to avoid.
