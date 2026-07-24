# The ~15-turn resident endurance ceiling — root cause and repair (2026-07-24)

Fable lane (endurance forensics claim, ledger 8eb2b0eb). Status of the
ceiling before this work: "root unknown" (cortex-conversation-reliability
memory, Jul 15).

## Observed signature

`artifacts/reliability/runs/endurance-0715-clean-20260715-091002.jsonl`
(resident 32B, 5s-paced): latency climbs monotonically — 11s, 25s, 105s,
29s, 112s, 23s, 112s, then pins at the 216s turn ceiling from turn ~8-9
onward; the run dies by turn 20 of 200. Deaths = 0 — the model never
crashes, the turns just outgrow the timeout. That shape is an
accumulating per-turn cost, not a failure event.

## Root cause: the conversation path can never reuse KV — twice over

1. `_prompt_cache_entry_budget_for_model` (mlx_worker): 32B under
   `desktop_resource_guard_enabled()` → budget **0**, so the prompt-cache
   LRU is never constructed on the live desktop. (June 10 memory-ceiling
   era: `4e16a7a52 fix: enforce live mlx memory ceilings`.)
2. `_job_requires_prompt_cache_bypass`: every live user turn carries
   `clean_user_surface_contract=True` (set by the client whenever
   runtime_controls are bound), and that contract was in the bypass list —
   which not only skipped the cache but **cleared** it each turn.

Consequence: every turn re-prefills the ENTIRE conversation from token 0.
Per-turn cost grows linearly with history; total conversation cost grows
quadratically; under 5s pacing the staircase reaches any fixed timeout.
The measured knee (~turn 8-15) is where 32B prefill of the accumulated
history crosses the queueing threshold.

Two compounding amplifiers, both recorded in forensics:

* **JobWatchdog kills on 90s without a token.** Prefill emits no tokens,
  so once re-prefill alone crosses 90s the watchdog kills a HEALTHY
  worker mid-prefill; respawn + 20GB reload costs ~2 min and surfaces as
  216s/500s — the "kill-on-timeout IS the recovery" cluster from the
  mlx-worker-lifecycle memory.
* **Model-load admission stalls the event loop.** During the resulting
  reloads, `_declared_mlx_worker_footprint_gb → _path_size_gb` ran a
  synchronous `rglob`+`stat` walk of the model directory ON the event
  loop while 20GB of safetensors reads saturated the disk:
  `data/error_logs/stalls/stall_1784673149` (8.6s) and
  `stall_1784675621` (5.5s) both bottom out in exactly this frame. A
  stalled loop then degrades health/false-death detection — the known
  duplicate-runtime cascade trigger.

## Mechanism proof (bounded, 1.5B)

`prefill_ab_1p5b.json`: 12 growing turns, prefill-only, A/B —

* full re-prefill per turn: 0.043 → 0.253s, tracking context 58 → 614
  tokens (≈6x growth over 12 turns; total 1.47s)
* cached suffix prefill: flat ≈0.05s (total 0.51s)

The live 32B multiplies model cost ~20x and per-turn tokens ~10x; the
naive curve becomes tens of seconds growing past the watchdog and turn
timeouts — the observed staircase. (The cached arm's slow creep is the
honest residual: attention over a longer KV is linear, vs quadratic total
re-prefill.)

## Repairs (this checkpoint)

* 32B prompt-cache budget under the desktop guard: 0 → 2 entries, with a
  new per-entry cap of 6144 tokens (~256KB/token of KV at 32B geometry ⇒
  ≈1.5GB/entry, ≈3GB worst case — fixed and visible next to the ~20GB
  weights). Long conversations degrade gracefully to re-prefill instead
  of growing the cache unboundedly. 72B stays cacheless.
* `clean_user_surface_contract` no longer bypasses the cache; user turns
  get a partitioned scope (`user_surface` vs `default`) so internal
  lanes and conversation KV can never cross-contaminate. Probes and
  strict/proof contracts keep the full bypass.
* Bypass no longer implies CLEAR — health probes fire between user
  turns, and clear-on-probe would have silently evicted the conversation
  prefix each cycle. Only an explicit `clear_prompt_cache` clears; all
  existing OOM/zero-token/retry recovery clears are untouched.
* `_path_size_gb` memoized per (path, mtime) and the admission-path call
  moved off-loop via `asyncio.to_thread`.

Tests: `tests/test_mlx_memory_safety.py` (budget/cap/scope/bypass +
prefix-reuse accounting), existing mlx contract/resilience suites green.

## Honest boundaries

* Live end-to-end confirmation needs Bryan's restart; the live instance
  only picks up code at relaunch.
* Cache hits require the rendered prompt to be PREFIX-STABLE across
  turns. If the live head injects per-turn varying text (timestamps,
  advisories) before the history, reuse stops at the first divergent
  token and this fix under-delivers — check
  `Prompt cache budget ... entries` + hit behavior in the worker logs
  after restart; if hits are short, the follow-up is to move volatile
  content BELOW the stable history or into the suffix.
* The 15-turn wall should move substantially; the remaining linear
  attention growth and any prefix instability set the next ceiling.
