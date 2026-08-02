# Final Soak Verdict — the grow-up mandate's last act (2026-07-14/15)

> **Historical record — 2026-07-16.** A dated snapshot, kept as written for
> provenance. It is not a statement about the system today and is
> deliberately not updated. Current status: [DOC_STATUS.md](DOC_STATUS.md).

Four runs, four findings, zero wasted deaths. Every run either hardened the
soak harness itself or caught a real defect — which is the entire purpose
of a soak.

## Run anatomy

| Run | Config | Lifetime | Death cause | Verdict |
|---|---|---|---|---|
| 1 | desktop, tracemalloc armed | 2 min post-ready | Hypervisor existential self-shutdown: tracemalloc (stale attribution config) amplified initiative-arbiter on-loop scoring into 5.8 s freezes → 21 emergency incidents → lag_threat 1.00 → orderly exit 0 | Protection chain works exactly as designed; harness config fault |
| 2 | desktop, no tracemalloc | 20 min, healthy | `desktop_signal:SIGTERM` — Claude-session teardown killed the process group (the documented historical soak-killer) | Runtime exonerated; harness needed true daemonization |
| 3 | desktop, daemonized (setsid) | 9 min | "GUI closed by user" — the desktop launcher opened a visible window + live microphone on the operator's screen; he reasonably closed it | Runtime exonerated; unattended soaks must be headless |
| 4 | **headless :8001, daemonized** | **3.5 h, zero deaths, zero self-shutdowns** | Probe SIGTERM at turn 40 after 152 min of timeout-walled turns | Stability PASS; **serving-path P0 caught** (below) |

## Run 4 results

**Idle stability: PASS.** 50-minute idle window completed (first run to do
so). Settled idle slope **+198 MB/h** on a ~1 GB orchestrator process vs the
pre-fix +350–450 MB/h baseline — improved, not yet flat; the remaining
churn is un-attributed (headless keeps the fixed desktop-perception loop
dormant, so this is a different, smaller source). Liveness guard: zero
false or missed firings.

**Endurance serving: FAIL — P0 caught.** All 40 completed turns pinned at
p50 ≈ 210–216 s (the timeout/salvage wall). Instance RSS ~1.8 GB
throughout: **the 32B cortex never loaded.** deaths=0, hijacks=0, memory
flat, thermal fine — the degradation ladder answered every turn, exactly
as designed, but never got its primary rung back.

## The P0: admission starvation of the primary cortex load

Evidence (worktree `scratchpad/launch_20260714_210631.log`):

- 35× `model_load_admission_denied … resource_timeout`
  (receipted, e.g. `resource_admission-159a6b3e…`)
- `Model-load admission lease expired before release lane=cortex`
- `Primary 32B cortex is dead. Triggering background respawn (Attempt 2/5)`
- Health pulses the whole night: RAM 40–43 % (moderate threshold is 85 % —
  pressure was NOT the blocker), probes PASS, conversation lane
  `recovering (warmup_deferred)` forever.

Mechanism (in `core/runtime/control_plane.py` admission controller):
`MODEL_LOAD` conflicts with any other `MODEL_LOAD` lease (`_work_conflicts`
:505). A cortex-load lease that dies without release (worker killed
mid-load, handshake failure) blocks every retry until its TTL lapses;
retries burn their own `timeout_s` inside that window and return
`resource_timeout`; the K1 reconciler dutifully retries into the same wall.
With continuous probe traffic the system never exits the cycle. The
`lease expired before release` line proves at least one lease outlived its
holder — lease lifetime and worker lifetime are not tied together.

**Fix direction** (filed as a follow-up task): tie lease release to worker
death (the mlx kill paths already report deaths to the K4 breaker — the
same seam should release the admission lease), and/or make
`_expire_leases_locked` reap leases whose owning process is gone rather
than waiting for TTL. A regression test should replay this anatomy:
lease-holder dies unreleased → next foreground cortex load admitted
within one poll interval, not one TTL.

## Cumulative soak-harness lessons (now encoded in the driver)

1. Stale instrumentation is load: tracemalloc turned a healthy runtime
   into a self-euthanizing one. Attribution tooling must be armed only
   for attribution runs.
2. `nohup` is not teardown-proof; double-fork + `setsid` is.
3. Unattended soaks must be headless — no window, no microphone.
4. A liveness probe belongs inside every idle loop; RSS sampling alone
   watched a dead instance for an hour.

## Mandate status

The reliability roadmap (K1–K6, A1–A5, C1–C4), the eight external-review
defects, claims-evidence alignment, and full-suite certification are all
landed and pushed. The soak — deliberately last — validated the stability
stack (zero unexplained deaths across 4 runs; every death traced and
named) and caught one new P0 in the newest subsystem, with a precise
evidence trail. That is a soak succeeding, not failing.
