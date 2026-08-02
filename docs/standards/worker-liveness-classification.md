# Worker liveness classification

**Status:** live. `core/runtime/worker_liveness.py`, wired into the MLX lane's
stale-state recovery path.

## Upstream

| | |
|---|---|
| Project | llama.cpp / `llama-server` (and the same discipline in vLLM, TGI) |
| License | MIT |
| Adopted | Design idea only — the request-slot lifecycle concept |
| Code copied | **None.** Written from the published behaviour of server-grade inference schedulers. |

## The idea adopted

Mature inference servers treat **one request** as the unit of failure, not the
loaded model. A caller who stops waiting, a request that exceeds its budget, and
a worker whose decode loop has genuinely wedged are three different events with
three different correct responses. Collapsing them into "restart the worker"
makes every timeout cost a full model reload.

## Why Aura needed it

Aura's resident Cortex is ~20GB of wired memory. Killing it costs a cold reload,
and the reload has historically cascaded — a second worker stacking beside the
first, memory doubling, a death cluster. Killing is therefore among the most
expensive actions the runtime can take.

The evidence needed to avoid a wrongful kill already existed and was already
being published: the worker emits `active_job`, `job_age_s` and `loop_stalled`
in its heartbeat, and the client tracks `_last_heartbeat`,
`_last_token_progress_at` and per-request first-token budgets. What did not
exist was a single place where those signals decide whether killing is
warranted.

**Accuracy note.** The pre-existing code was not naive: `_reset_stale_lane_state`
already skipped the kill when any activity timestamp was newer than 30s. What it
lacked was *graded* classification — the decision was binary (recent activity or
not), so it could not distinguish a stalled request from a wedged process, and
it produced no auditable reason. This adoption adds the gradation and the
receipt; it did not fix a naked kill-the-healthy-worker bug, and the commit
history should not claim it did.

## The contract

| Verdict | Meaning | Kill justified |
|---|---|---|
| `GENERATING` | Produced output within the stall budget | No |
| `IDLE` | Fresh heartbeat, no job in flight | No |
| `STALLED` | Fresh heartbeat, job not progressing | No — cancel the *request* |
| `WEDGED` | Nothing has proven liveness in tolerance | Yes |
| `DEAD` | Process gone | Yes |
| `UNKNOWN` | Insufficient evidence | No |

Two rules carry most of the value:

1. **Proof of life outranks proof of bookkeeping staleness.** A worker that
   emitted a token one second ago is `GENERATING` even if its lane state has
   been stale for an hour and its own watchdog says `loop_stalled`.
2. **Absent evidence never licenses an irreversible action.** A worker that has
   not yet sent a first heartbeat is *starting*, not dying; `None` and "stale"
   are different claims and collapsing them would kill every worker during its
   initial load.

## Aura deviations

- Generic over any long-lived worker, not just inference lanes, so browser
  workers and tool runners can share the vocabulary.
- No batching or slot-sharing was adopted: MLX owns its own scheduling and
  Aura's lane is single-tenant by design.
- Thresholds are env-overridable but clamped, so an override cannot disable
  bounding.

## Conformance tests

`tests/test_worker_liveness.py` — 14 tests, including the two rules above and a
parametrised sweep asserting that no evidence-poor state is ever killable.

## Known unsupported

- Continuous batching, slot reuse, and KV-cache checkpointing are **not**
  adopted. They belong to a multi-tenant server; Aura's resident lane is not one.
- The classifier is advisory at the other kill sites in `mlx_client.py`; only the
  stale-lane path consults it so far.
