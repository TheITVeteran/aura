# Longevity Soak Standard

*Reviewed against the tree: 2026-08-01. See [documentation status map](DOC_STATUS.md) for how to read this file.*

How stability gets tested over hours rather than seconds.

Most failures in this runtime do not announce themselves on turn one. They
accumulate — memory that never comes back, latency that climbs a little each
turn, a worker that gets killed and reloaded until the reloads overlap. A
suite that passes in ten seconds tells you nothing about any of that, which
is what the soak is for.

## 1. Operational Parameters

Longevity soak testing verifies that Aura does not suffer from asymptotic failures, memory leaks, runaway thread creations, or event-loop degradation during sustained operation.

The standard defines four runtime profiles:

| Profile | Duration | Target Environment | Intended Claim Validation |
| :--- | :--- | :--- | :--- |
| **`proof_short`** | 5 - 10 iterations | local CI | Bounded pipeline validation, no memory growth leakage |
| **`local_4h`** | 4 Hours | dev workstation | Bounded state continuity, memory limit stabilization |
| **`local_24h`** | 24 Hours | staging server | Extended memory, queue, and thread safety |
| **`local_72h`** | 72 Hours | production cluster| Indefinite Autonomy, asymptotic loop stability |

## 2. Monitored Metrics & Failure Thresholds

During a longevity soak run, the system must monitor and log the following:
1. **Memory Allocation**: Resident Set Size (RSS) must not grow linearly. Memory consumption must stabilize within the first 10% of the run.
2. **Event-Loop Lag**: Average event-loop delay must remain below 50ms per tick.
3. **Queue Health**: Task queue lengths must remain bounded (no infinite queuing or thread leaks).
4. **Receipt Continuity**: 100% of tasks must emit trace logs and corresponding volition receipts.

## 3. Failure Conditions

The longevity soak gate must fail if:
- Any unhandled exception halts the core loops.
- Resident memory increases by more than 50% between the 20% mark and the end of the soak run.
- Queue length grows monotonically for more than 5 consecutive monitoring checkmarks.
