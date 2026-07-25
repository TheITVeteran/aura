# Soak verdict — 2026-07-25 02:38 run

**Verdict: FAIL.** Not on stability. On whether she answered at all.

Run: headless :8001, booted 02:38:16, ready in 40s, 50-minute idle window,
then a 200-turn endurance probe. Code under test: `60065773c` — the K/A/C
reliability set, **before** the eleven fixes this document motivated.

## What passed

| Measure | Result |
| --- | --- |
| Turn deaths | **0** / 200 |
| Task hijacks | **0** / 200 |
| Server loss | none |
| Turns over 180s | none |
| p50 latency | **3.33s** (was 167s on the 2026-07-18 run) |
| p95 latency | 69.88s |
| RSS, idle hour | 1.14 → 1.22 GB over 50 min |
| Thermal events | 0 |

The stability work holds. Nothing died, nothing wedged, nothing ran away,
and the serving-latency ceiling that dominated the previous run is gone.

## What failed

| Failure | Value |
| --- | --- |
| `identical_reply_repeated_x173` | 173 of 200 |
| `retention` | 0 / 3 |
| `math_accuracy` | 0 / 37 |
| `critical_incident_active` | 5 active, 1 critical |

**173 of 200 turns returned the same sentence:** *"I couldn't put together
an answer I'd stand behind for that one…"*. At a 3.3s median. She was not
slow. She was not trying.

Answered-turn breakdown, by the probe's own turn kinds:

| Kind | Turns | Answered |
| --- | --- | --- |
| casual | 58 | 7 |
| knowledge | 49 | 2 |
| math | 37 | 4 |
| philosophy | 30 | 3 |
| introspection | 20 | 8 |
| plant (memory) | 3 | 0 |
| retention probe | 3 | 3 (all "contract not proven") |

`retention=0/3` is entirely downstream: all three memory plants went
unanswered, so there was never anything planted to recall. `math=0/37` is
mostly downstream too — only 4 math turns got any answer at all, and three
of those came from degraded lanes. Neither is independent evidence of a
memory or arithmetic defect; both are the silence, counted twice.

## Root cause

Two defects stacked, and both are the same category error this whole pass
has been about: **a deliberate deferral counted as damage.**

**1. The cortex could never finish loading.** `ensure_foreground_ready`
caps a *recovery* wait at 15s. That is a designed handoff — the warmup task
is shielded, the cortex keeps loading in the background, and the current
turn falls to the ready fallback. But the `TimeoutError` from that
deliberate cap called `_note_cortex_stuck_kill()`, the same counter a
force-killed stuck load feeds. At threshold the backoff deferred cortex
warmup by 240s. So the load never finished; so the lane was never ready; so
the next turn was also a "recovery" and timed out at 15s again. **62 load
attempts in one run, zero completions.** The mechanism that exists to
prevent GPU thrash was being read as GPU thrash.

**2. And the fallback was refused anyway.** With the cortex deferred, every
protected foreground turn hit a branch that returned `None` because "this
turn requires the primary lane" — while the same log recorded *"keeping
fallback workers resident (33.2GB free ≥ 24.0GB cortex + 8.0GB fallback)"*.
The Brainstem was loaded, ready, and never asked. That branch conflated
`strict_primary_proof_lane`, where the request's contract names the primary
model and a lower lane would misreport its own provenance, with
`protected_foreground_lane`, which only means a real person is waiting.
Protecting someone is not a reason to hand them nothing.

## The idle window

The quiet 50 minutes before the probe were their own finding — nothing was
failing, and she was still refusing her own work:

- `body_fatigue` pinned at **0.996** for the entire window, and
  `recovery_debt` saturated with it. Both decayed at a flat rate that the
  ordinary drip of idle-loop body costs cancels exactly, so neither ever
  left saturation. That held `welfare.recovery_drive` at 0.881, over the
  Will's 0.6 defer threshold, for the whole hour.
- ~110 `welfare_recovery_required_before_action` deferrals: 43 belief
  updates, 31 memory writes, 29 interaction commits, 22 initiatives.
- 71 knowledge writes blocked on `epistemic_reconciliation_required:2` —
  **two** contested claims gating every unrelated fact she learned.
- A completed 10.8-second web search discarded outright, logged as
  "all memory backends rejected the artifact". Nothing rejected it; the
  Will said "later" and nobody was holding it.
- 247 immune `reallocate_flow` actions, 192 of which failed silently.
- 4,212 `UnifiedField saturation rescue` firings — mean|F| still 0.906
  after each one.
- Nine identical web searches for *"What do I not know about something
  new?"*, returning Windows release notes.
- 3,377 of 4,650 stdout lines (73%) were sentence-transformers progress
  bars, burying all of the above.
- Two faults the runtime raised against itself every boot: a blacklist it
  wrote in one format and read in another, and a nightly-LoRA training
  write refused for want of a governed scope. With the welfare storm these
  drove cortisol into crisis and produced 15 `SubstrateAuthority BLOCKED`
  entries — real work refused because of noise the system made about itself.

## Status

Eleven fixes landed against this run (`db6afcba4` … `786a11968` and
earlier). This document records the measurement that motivated them; it is
**not** evidence that they worked. That requires a re-run on the new code —
see `docs/VERIFY_2026_07_25.md`.

An honest reading of the two numbers side by side: the 2026-07-18 run
failed on latency and this one did not, so the stability arc is real. But a
runtime that answers 13.5% of a conversation is not a reliable daily tool,
and no amount of zero-deaths changes that. The probe was right to fail it.
