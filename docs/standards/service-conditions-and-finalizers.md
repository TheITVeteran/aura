# Service conditions and finalizers

**Status:** live. `core/runtime/service_conditions.py`, populated by
`RuntimeControlPlane._transition` on every real reconciliation.

## Upstream

| | |
|---|---|
| Project | Kubernetes API conventions (conditions, finalizers) |
| License | Apache-2.0 |
| Adopted | The condition vocabulary (type/status/reason/lastTransitionTime/observedGeneration) and the finalizer guarantee |
| Code copied | **None.** |

## What Aura already had

The desired/observed reconciliation half of this pattern was adopted long ago:
`control_plane.py` carries `DesiredServiceSpec` and `ServiceObservation` with
`desired_state`, `observed_state`, `generation`, `last_transition_at`, restart
budgets and backoff. **This adoption did not rebuild any of that.**

## What was missing

A single collapsed `observed_state` cannot express independent facts:

- loaded and usable, but **not accepting** new foreground work;
- running, but a **dependency is missing**;
- **degraded while recovering normally** — which is not the same as failing.

Squashing those into one enum forces every consumer to re-derive the
distinction from a prose `reason` string. Conditions make each claim
independently readable.

### Three semantics that carry most of the value

1. **`UNKNOWN` is not a failure.** It means the probe could not run. Collapsing
   it into `FALSE` turns "we could not check" into "it is broken", which
   triggers recovery for a healthy service.
2. **Transition times only move on a real status change.** A condition
   re-asserted with the same status keeps its original timestamp, so *"how long
   has this been true?"* stays answerable — the question that actually decides
   whether something is stuck.
3. **`observed_generation` stops a stale status reading as current.** A
   condition behind the object's generation describes a configuration that no
   longer exists, and says so.

## Finalizers

A subsystem is not "stopped" because something asked it to stop. It is stopped
when its cleanup completed: transactions closed, leases released, children
reaped. Aura has hit this exact bug — a lane declared cold while its worker
still held ~20GB.

Two deliberate behaviours:

- **One failing cleanup does not strand the others.** A released lease is worth
  having even if a temp file could not be removed.
- **A failed finalizer is RETAINED.** Dropping it would make the shutdown *look*
  clean, which is the failure mode the mechanism exists to prevent. It can be
  retried and will succeed later.

## Integration

Conditions are derived inside `_transition`, so they are populated by real
reconciliation rather than being a parallel surface someone must remember to
update. The derivation never raises: conditions are an observability surface,
and a reporting failure must not break the reconciliation that produced it.

## Conformance tests

`tests/test_service_conditions.py` — 13 tests, including ready-but-saturated,
degraded-while-recovering, UNKNOWN-is-not-failure, transition-time preservation,
generation staleness, partial finalizer failure, and the live control-plane
derivation.

## Known unsupported

- `ACCEPTING_WORK` and `DEPENDENCIES_SATISFIED` are defined and testable but not
  yet populated by the control plane — nothing there currently measures queue
  admission or dependency presence per service. They are set by callers who do.
- Finalizers are available to the control plane but not yet attached to the MLX
  lane's shutdown path, which is where they would have the most value.
