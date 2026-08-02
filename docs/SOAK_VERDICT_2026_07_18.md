# Soak verdict — 2026-07-18 (headless :8001, tip c8d4ef5c + lease-P0 fix)

> **Historical record — 2026-07-18.** A dated snapshot, kept as written for
> provenance. It is not a statement about the system today and is
> deliberately not updated. Current status: [DOC_STATUS.md](DOC_STATUS.md).

The grow-up arc's final endurance run, on the tree carrying the
2026-07-15 P0 fix (admission leases now die with their holders). Headless,
daemonized (double-fork + setsid), tracemalloc off — every lesson from the
four July-14 runs baked into the driver.

## What the run measured

| Phase | Result |
|---|---|
| Boot → conversation-ready | **30 s** |
| Idle window (50 min, 100 samples) | **RSS slope −21 MB/h** (declining) |
| Lease P0 under real load | **FIXED, proven live** — 4 dead-holder reaps |
| Endurance (200-turn probe) | **deaths 0, hijacks 0** through every milestone |
| Endurance verdict | **FAIL on latency** — 65/200 in the 180-min deadline, p50 167 s |

## PASS: memory stability

The idle leak that opened this whole thread (2026-07-07's "+242 MB/h", the
+350–450 MB/h baseline of the Phase-13 runs) is **gone**. Fifty minutes of
genuine idle, 100 samples: the orchestrator RSS *declined* at −21 MB/h. The
perception-loop subprocess backoff (Phase 12) plus the headless dormancy of
the fixed perception loop hold. Load-phase slope was +295 MB/h across 182
min of continuous 32B thrash — churn under load, not a leak: it tracks the
worker cycling, and the process never approached pressure (RAM 43–72%,
thermal 0, zero deaths).

## PASS: the lease P0 is fixed, and this run proves it live

The 2026-07-15 P0 was: a cortex-load worker killed mid-load left its
MODEL_LOAD admission lease standing; because MODEL_LOAD conflicts with every
other MODEL_LOAD, every recovery load then burned its own timeout into
`resource_timeout` behind the orphan's TTL — all night, RAM at 40%, ladder
answering, cortex never loading.

This run's launch log shows the fix firing: **4× `🧹 Reaped N orphaned
model-load admission lease(s) for dead cortex worker`**. Each kill released
its lease and the next load re-admitted within a poll interval. The
all-night starvation lock is broken — today's anatomy is "reload and
continue," not "starve behind a dead lease."

## FAIL: serving latency — the honest, unchanged ceiling

The probe still walled: p50 167 s, p95 210 s, and it reached only turn 65 of
200 before the 180-min deadline. This is **the same reliability ceiling this
program has named since 2026-07-07**, and the P0 fix — correctly — did not
move it, because it was never the cause:

- Under 5-s-paced continuous load the resident 32B loses first-token
  bandwidth; the inference gate force-kills the stuck load; the K4 breaker
  then backs warmup off ("6 stuck-load kills in 300s — deferring 240s"), so
  the lane sits `recovering (foreground_warmup_deferred_memory_pressure)`
  while the reflex/brainstem fallback answers every turn at the ~210 s wall.
- `deaths=0` throughout: the cortex→cloud→reflex ladder always answered.
  What fails is *smoothness*, not liveness.
- `identical_reply_repeated_x32` and `math=0/7` are the reflex 1.5B carrying
  the whole conversation because the 32B never got a warm window — expected
  when the cortex can't hold under this load shape, not a new defect.

This is the architectural ceiling the roadmap named from the start:
**model-serving throughput under sustained fast-paced load on a 64 GB host.**
It is a warm-window/first-token-bandwidth problem (K3 admission + adaptive
first-token deadline territory), not a lease, leak, or crash problem — all
of which are now closed.

## Bottom line for the grow-up arc

- **Stability: PASS.** Zero deaths, zero self-shutdowns, memory flat-to-
  declining at idle across the run.
- **The two P0s this arc found are both fixed and proven:** the idle leak
  (Phase 12) and the admission-lease starvation (this session).
- **The remaining ceiling is honest and known:** serving p50 under
  sustained load. It is the next campaign's target (warm-window admission /
  adaptive first-token budget), not a regression and not a surprise.

Artifacts: `scratchpad/probe_20260718_113144.log`,
`scratchpad/rss_trend_20260718_113144.csv`,
`scratchpad/launch_20260718_113144.log` (search `Reaped`).
