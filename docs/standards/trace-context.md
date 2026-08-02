# Trace context propagation

**Status:** live. `core/runtime/causal_trace.py`, injected at
`MLXLocalClient._authorize_job` — the single chokepoint every job passes through.

## Upstream

| | |
|---|---|
| Project | OpenTelemetry / W3C Trace Context |
| License | Apache-2.0 (OTel spec); W3C Trace Context is an open standard |
| Adopted | The `traceparent` wire format and the propagate-at-every-boundary discipline |
| Code copied | **None.** No OTel SDK dependency; Aura's tracer is its own. |

## What Aura already had

Both halves of the machinery, and this is the important part:

- `core/observability/tracing.py` — Span/Tracer, OTel-compatible export format;
- `core/runtime/causal_trace.py` — context carried across async tasks, with
  `inject_trace_carrier` / `extract_trace_carrier` already written.

## What was actually wrong

`inject_trace_carrier` and `extract_trace_carrier` had **zero call sites outside
their own module**. The propagation machinery was built and never wired, so a
turn's trace stopped dead at the IPC edge and worker-side events could not be
correlated back to the conversation that caused them.

This is the same failure shape as a check that is defined but never called: the
capability existing made the problem look solved. Grep is what disproved it.

## What this adoption adds

1. **Injection at the chokepoint.** Every job reaching the worker goes through
   `_authorize_job`, so wiring there covers all job types at once and no future
   job path can forget. Injection happens *after* any request digest the caller
   computed and under keys no digest covers, so propagation cannot invalidate a
   signed contract.
2. **W3C `traceparent`.** A carrier that leaves Aura is now readable by any
   conformant tool without translation. Aura's own ids remain the authority;
   the header is a rendering of them, normalised to the widths a parser
   requires — an unusual id is hashed rather than mangled, so it still yields a
   valid, stable header.

## Two rules

- **No active trace fabricates no correlation.** Inventing a trace id would be
  worse than having none: it would assert a causal link that does not exist.
- **Injection is total and never raises.** This is observability decorating a
  job about to do real work; a tracing fault must not be able to take down
  inference. A test asserts this, and it caught a real gap — the first
  implementation let `RuntimeError` escape.

## Conformance tests

`tests/test_trace_propagation.py` — 11 tests: W3C header validity including for
malformed ids, parent/child trace identity, round-trip across the boundary,
no-fabrication, chokepoint coverage, and the never-raises guarantee.

## Known unsupported

- No collector or exporter is wired. Everything stays local, which is the
  intent — the value here is one causal graph inside Aura, not shipping
  telemetry off the machine.
- The worker does not yet re-enter the extracted context for its own spans; it
  can rebuild it (tested) but does not scope its logging to it.
