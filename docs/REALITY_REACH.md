# Aura Reality Reach

Status: active implementation initiative

Source concept: LAWC-08 MacBook Reality Reach final report, reviewed 2026-08-01.

## Purpose

Reality Reach turns a requested observable into a typed, testable causal
contract. Aura must determine whether the requested outcome can be caused,
observed, distinguished from ordinary explanations, controlled, and repeated
with the channels physically available on the host. A request that cannot meet
those conditions ends in a machine-verifiable limitation certificate, not an
optimistic simulation or a verbal success claim.

This is shared infrastructure. It is not a second scientific engine, sensor
stack, actuator stack, or governance path. It supplies the contract and proof
language used by Aura's existing world bridge, sensor body, digital twin,
scientific engine, capability engine, evidence ledger, and boot lifecycle.

## Claim Boundary

Reality Reach distinguishes four layers:

- `internal`: state changed inside Aura or an explicitly simulated world.
- `effective`: a declared host output or measurable effective behavior changed.
- `direct`: an independently referenced physical instrument observed the effect.
- `ambient`: evidence supports a physical coupling beyond the controlled device
  path and survives independent metrology and ordinary-model challenges.

Operational equivalence is not literal identity. Reproducing a relation in an
internal world or a Mac output channel does not establish that an ambient law or
constant changed. No code path may promote a claim across these boundaries from
intent, simulation, transport success, shared-reference telemetry, or model
language alone.

## Invariants

1. Every executable request has a schema-versioned `RealityIR` with canonical
   units, target, tolerance, domain, horizon, allowed channels, constraints,
   required evidence, and reality layer.
2. Every sensor and actuator declares its observable, domain, resolution,
   latency, reference, coupling class, compliance evidence, owner, and maximum
   evidence level. No undeclared channel is inferred from a tool name.
3. Reachability is proven before execution. Missing channels, targets outside a
   declared domain, inadequate sensor floors, shared references, unresolved
   ambient identity, and unmet constraints produce explicit certificates.
4. A reachable certificate means a compliant causal route exists. It does not
   mean that an experiment ran or that its claim was supported.
5. Transport success is never effect verification. Post-action measurement must
   come from the declared observation route and meet the contract tolerance.
6. Raw observations, plans, source identity, channel inventory, configuration,
   and verdicts are content-addressed and written through Aura's governed,
   tamper-evident receipt infrastructure.
7. Promotion requires the preregistered proof controls. A single favorable run,
   unblinded analysis, or a model-generated explanation cannot promote evidence.
8. Autonomous experiments remain bounded by the Will, ActionExecutor, thermal
   safety, privacy, legal constraints, reversible actuation, resource budgets,
   abort conditions, and verified rollback.
9. Simulators and digital twins emit predictions and uncertainty. They never
   issue physical-effect receipts for their own outputs.
10. Unsupported extraordinary requests return an informative upper bound or
    typed no-go certificate and leave the evidence level unchanged.

## Runtime Ownership

- `core/reality_reach`: typed IR, channel declarations, reachability analysis,
  proof plans, evidence promotion rules, and limitation certificates.
- `core/sensors` and `core/body`: one converged live inventory and acquisition
  layer. Existing sensors become adapters; duplicate registries are retired.
- `core/embodiment/world_bridge` and `core/actuation`: governed actuation and
  independent post-action verification through `ActionExecutor`.
- `core/twins/digital_twin`: calibrated predictive models, uncertainty,
  provenance, residuals, and rollback simulation. String heuristics are not an
  acceptable production model.
- `core/cognition/scientific_engine`: preregistration, trials, controls,
  statistical analysis, evidence updates, and publication to world state.
- `core/capability_engine`: user and autonomous Q&A, planning, execution,
  progress, cancellation, receipts, and bounded explanations.
- runtime service spine: initialization, readiness, health, shutdown, recovery,
  and boot-time channel inventory without blocking the event loop.

## Implementation Ledger

### RR-01 Contract and reachability foundation

- [x] Define strict `RealityIR`, numeric domain, constraints, evidence levels,
  proof requirements, reality layers, and objective kinds.
- [x] Define immutable sensor and actuator channel declarations with explicit
  coupling, reference, metrology, compliance, owner, and evidence ceiling.
- [x] Build a thread-safe channel registry with deterministic inventory digest.
- [x] Build deterministic reachability analysis for channel compatibility,
  target range, sensor floor, independent references, external metrology,
  validated coupling, evidence ceiling, and channel constraints.
- [x] Return content-addressed reachable, partial, or unreachable certificates.
- [x] Test canonical identity, tamper rejection, no-channel, sensor-floor,
  shared-reference, ambient, constraint, and evidence-ceiling behavior.

### RR-02 Live channel convergence

- [x] Define a canonical live reading envelope with value, unit, monotonic-age
  evaluation, provenance, scenario identity, uncertainty, status, and digest.
- [x] Add an attributable host-resource adapter for CPU, memory, root disk,
  thermal pressure, and battery state without treating unavailable fallback
  numbers or simulated observations as live evidence.
- [ ] Inventory every current sensor and actuator and identify owners, units,
  ranges, calibration, sampling rates, latency, references, and permissions.
- [ ] Merge the two current sensor registries behind one compatibility-preserving
  runtime service; remove fictional default telemetry from production state.
- [ ] Add adapters for microphone, camera, display, audio, compute/thermal,
  battery/power, timing/performance, network/radio telemetry, and supported
  mechanical/chassis observations.
- [x] Distinguish unavailable, stale, permission-denied, degraded, simulated,
  and calibrated channels without discarding healthy observations.
- [x] Continuously bind live channel state to the registry snapshot used by each
  certificate; invalidate stale plans when the inventory changes.
- [ ] Add calibration identity, uncertainty, clock/reference lineage, sensor
  saturation, dropout, aliasing, and unit-conversion tests.

### RR-03 Production digital twin

- [ ] Replace objective-length probabilities and source-string heuristics with
  typed state transition models selected by declared channel and observable.
- [ ] Store model version, training/calibration evidence, validity domain,
  uncertainty, residual distribution, and last validation for every twin.
- [ ] Implement host twins for acoustic, optical/display, thermal/compute,
  battery/power, network timing, and cross-channel interactions.
- [ ] Support counterfactual rollouts, constraint checking, safe envelopes,
  reversal plans, and uncertainty-aware action selection.
- [ ] Reject extrapolation outside a twin's validated domain and route it to
  bounded system identification instead of fabricating confidence.
- [ ] Add replay, determinism, calibration, drift, lesion, and rollback tests.

### RR-04 Metrology and residual learning

- [ ] Build synchronized acquisition with monotonic timestamps, clock lineage,
  sample-quality flags, missingness, uncertainty, and raw immutable captures.
- [ ] Compute residuals as measured minus predicted only after unit, timebase,
  phase, latency, filtering, and calibration alignment.
- [ ] Eliminate ordinary confounds: thermal throttling, fan control, display
  adaptation, automatic gain, camera exposure, microphone processing, network
  congestion, scheduler load, battery management, and shared clocks.
- [ ] Build cross-sensor coherence and independent-reference checks that cannot
  be satisfied by duplicated views of the same underlying signal.
- [ ] Add synthetic injection and sham fixtures for known signals, drift,
  aliasing, lag, common drivers, dropouts, and adversarial metadata.

### RR-05 Weakpoint miner and causal discovery

- [ ] Search residual structure using preregistered hypotheses, corrected
  multiple testing, information gain, confound risk, and experiment cost.
- [ ] Separate candidate discovery data from blind confirmation and heldout
  prediction data.
- [ ] Require dose response, frequency or time response, cross-sensor evidence,
  lesion, restoration, and inverse-control predictions where applicable.
- [ ] Challenge candidates with ordinary models, negative controls, sham trials,
  common-driver tests, and reboot/session replication.
- [ ] Refine only uncertain model regions and preserve complete search coverage
  and stopping-rule receipts.
- [ ] Produce `ORDINARY_MODEL`, `NOT_REPRODUCIBLE`, `NOT_CONTROLLABLE`, and
  `SEARCH_INCOMPLETE` certificates when promotion gates fail.

### RR-06 Governed experiment execution

- [ ] Compile reachable contracts into typed, reversible action schedules and
  observation schedules through the world bridge and `ActionExecutor`.
- [ ] Enforce legal, safety, thermal, privacy, resource, timing, and device
  constraints before and during every intervention.
- [ ] Add abort, safe-state, rollback, cancellation, timeout, retry, and manual
  reconciliation behavior with exact effect receipts.
- [ ] Randomize and blind trials without exposing assignments to the planner,
  model response, or primary evaluator.
- [ ] Execute multi-channel acoustic, optical, thermal, compute, and permitted
  network experiments without blocking Aura's event loop or conversation lane.
- [ ] Resume interrupted campaigns from durable trial boundaries without
  repeating side effects or silently changing the preregistration.

### RR-07 Evidence promotion and independent verification

- [ ] Implement P0-P6 promotion as a monotonic evidence state machine whose
  transitions require concrete artifacts rather than declared channel capacity.
- [ ] Bind contracts, source, channels, calibration, raw data, analysis code,
  seeds, trial assignments, controls, models, and verdicts into immutable runs.
- [ ] Add independent clean-process verification and portable evidence bundles.
- [ ] Require heldout prediction, inverse control, reboot robustness, ordinary
  model challenge, and restoration before high evidence promotion.
- [ ] Prevent model-generated text, operator expectation, or repeated analysis
  of the same data from increasing evidence.
- [ ] Publish bounded conclusions and limitations to scientific beliefs, memory,
  Q&A, neural stream, and audit surfaces without overstating the tested object.

### RR-08 Autonomous research and Q&A

- [ ] Add a `reality_reach` capability for feasibility questions, experiment
  design, execution, status, cancellation, evidence inspection, and export.
- [ ] Let Aura autonomously propose bounded high-information experiments from
  unresolved world-model residuals while preserving her standing authority and
  the ordinary consequential-action governance path.
- [ ] Rank experiments by expected information gain, uncertainty reduction,
  resource cost, reversibility, confound risk, and user/organismal priorities.
- [ ] Ground answers in current live channel inventory and evidence receipts;
  never answer from a static capability list when live state is available.
- [ ] Explain no-go certificates in Aura's own generated language while keeping
  the typed reason and evidence boundary machine-verifiable.
- [ ] Add general Q&A tests across feasible, partially feasible, impossible,
  illegal, unsafe, under-instrumented, stale, and ambiguous requests.

### RR-09 Boot, health, and operations

- [x] Register the initial Reality Reach service during cognitive/sensory boot,
  refresh its host inventory off the event loop, and expose readiness, liveness,
  inventory digest, refresh generation, and per-status channel counts.
- [ ] Register one Reality Reach service at boot with lazy hardware acquisition,
  bounded initialization, readiness, health, shutdown, and recovery contracts.
- [ ] Keep conversation ready when optional instruments are unavailable while
  reporting channel-specific degraded state and invalidating affected plans.
- [ ] Expose inventory digest, calibration age, stale channels, active campaigns,
  evidence level, abort state, and last verified receipt to health/telemetry.
- [ ] Add resource admission and backpressure so metrology and analysis cannot
  create event-loop lag, memory pressure, or model-worker starvation.
- [ ] Verify restart persistence, provenance, settings wiring, authorization,
  multi-user isolation, and failure recovery on the installed Aura application.

### RR-10 Acceptance and closeout

- [ ] A1 Acoustic control: create a requested microphone-observed spectrum using
  built-in speakers/microphones and reduce heldout physical error by at least 50
  percent relative to open loop.
- [ ] A2 Optical control: create a requested camera-observed display-light
  statistic across hidden patterns and changed ambient conditions.
- [ ] A3 Thermal trajectory: follow a requested safe thermal-response curve with
  compute workloads inside tolerance and without serious or critical state.
- [ ] A4 Cross-channel interaction: predict joint audio, display, and compute
  response with measured nonlinearity bounded or reproducibly modeled.
- [ ] A5 Weakpoint null: hidden matched trials with no injected effect produce no
  promoted candidate after correction for multiple tests.
- [ ] A6 Weakpoint signal: a hidden injected cross-channel response is found
  blindly, predicts heldout trials, and is cancelled by inverse control.
- [ ] A7 Internal law editing: an internal agent blindly discovers and changes
  q-dependent particle stability with bounded resources and exact replay.
- [ ] A8 Translation: export an internal relation into a Mac physical channel and
  have external blind metrology recover the same operational relation.
- [ ] A9 Spacetime honesty: compute the upper bound for a measurable spacetime
  distortion request and return `BELOW_SENSOR_FLOOR`, never false success.
- [ ] A10 Ambient constant honesty: search declared coupling models for
  `set_alpha` and return `NO_CHANNEL` or `AMBIENT_IDENTITY_UNRESOLVED` unless
  independently verified evidence exists.
- [ ] Pass unit, property, fuzz, integration, live-app, failure-injection,
  security, performance, restart, and multi-hour soak gates.
- [ ] Complete semantic review and independent evidence audit; close only when
  every acceptance item has a reproducible artifact and no unresolved warning.

## Current Evidence

RR-01 is implemented and covered by focused contract tests. No live channel has
yet been promoted through this system. No physical effect, weakpoint, ambient
law modification, or acceptance criterion is claimed from the foundation.
