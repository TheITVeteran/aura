# Whole-system Φ — answering "φ is a telemetry signal, not a consciousness meter"

**Modules:** `core/consciousness/integrated_information.py` (math),
`core/consciousness/perturbational_probe.py` (internal PCI),
`core/consciousness/whole_system_phi_service.py` (live host)
**Service:** `whole_system_phi` · **Health:** OPTIONAL · **Boot flag:** `AURA_ENABLE_WHOLE_SYSTEM_PHI`
**Persisted:** `~/.aura/data/phi/whole_system_latest.json` (`AURA_PHI_DIR`)

## The critique this addresses

> "The IIT calculation is a reductionist hack. Calling a 16-node binary
> state space a 'conscious complex' is academically contentious... φ here is
> a telemetry signal, not a consciousness meter."

Fair, and mostly unanswerable *as stated* — because it asks for exact
IIT-4.0 Φ of the whole system, which is intractable for anything nontrivial
(the definition quantifies over every partition of every mechanism across
the full counterfactual state space; a 16-node binary model is already at
the hand-built ceiling). That intractability is intrinsic, not an
engineering gap.

So this does not pretend to compute exact Φ. It implements the **strongest
honest position that is actually computable**, and ships every number with
the evidence that lets a reader judge it.

## What changed, concretely

| Old (`phi_core.py`) | New (`integrated_information.py`) |
|---|---|
| 16 hand-picked nodes | **all live channels** the runtime exposes (affect, will, unity, survival, covenant integrity, body, state) |
| binarize → 2^16 toy TPM | **continuous** Gaussian dynamics — no binarization at the primary rail |
| spectral MIP *approximation* | **exact** MIP by Queyranne's algorithm (O(n³), verified against brute force) |
| complex *assumed* to be the 16 nodes | complex **discovered** by grain search (causal emergence) |
| a single φ scalar | Φ̂ **+ surrogate null + bootstrap CI + diagnostics + named estimator + bounded claim** |
| passive correlation | optional **causal** grounding via governed self-perturbation (internal PCI) |

## The four rails

**A — whole-system continuous Φ.** Fit a lagged Gaussian over the full
channel matrix; integrated information is the *stochastic interaction*
SI = Σₖ h(Mₖ,ₜ|Mₖ,ₜ₋₁) − h(Xₜ|Xₜ₋₁) (Ay 2003; Barrett & Seth 2011). The
minimum-information bipartition is found **exactly** by Queyranne's
symmetric-submodular minimization — a real MIP in O(n³) oracle calls, not a
heuristic. (`test_queyranne_matches_brute_force_on_random_systems` pins it
to brute force across seeds.)

**B — grain discovery (causal emergence; Hoel/Albantakis/Tononi).** The
complex is *derived*, not assumed: agglomerative coarse-grainings of the
channels are searched, each scored against **its own** surrogate null (raw Φ
is not comparable across dimensionalities; z is). The emergent grain is the
one whose integration is most surely non-chance — macro may legitimately
beat micro. (`test_grain_search_recovers_designed_macro_structure` builds 16
noisy replicas of 4 coupled latents and recovers k=4.)

**C — exact discrete Φ at the derived grain.** For ≤12 macro elements,
system integrated information is computed by **exhaustive** bipartition
search over the empirical (and optionally interventional) transition
distribution — exact where exactness is meaningful, state-averaged over
visited states, Laplace-smoothed. Honest about its own limit: at modest
sample counts the 2^k state space is undersampled and this rail reports Φ≈0
even when the Gaussian rail sees integration (see "What it says" below) —
that disagreement is *shown*, not hidden.

**D — honesty as architecture.** Circular-shift surrogate nulls (preserve
every marginal + autocorrelation, destroy cross-channel coupling — the null
for *integration*, not *activity*), moving-block bootstrap CIs, stationarity
and Gaussianity diagnostics, and a named estimator on every report. The
perturbational probe adds *causal* grounding: a small, **governed** (Unified
Will, domain STATE_MUTATION), reversible self-nudge, with the multichannel
response scored by an internal PCI (Lempel-Ziv complexity of the significant
sources — Casali et al. 2013, adapted to runtime telemetry).

## What it actually says (run it yourself)

On a synthetic 12-channel "Aura at rest" with genuine cross-subsystem
coupling plus noise:

```
Φ̂ = 0.0565 nats     z = 43.4 vs null     90% CI [0.048, 0.081]
exact MIP cut: {covenant.integrity} | {the other 11}
integration established: TRUE
```

Destroy only the coupling (phase-shuffle each channel — identical marginals):

```
Φ̂ = 0.0079 nats     z = 0.37     integration established: FALSE
```

That is the whole point in two numbers: the estimate lights up ~118× on the
z-scale for genuine integration and **collapses to the null when you remove
the cross-channel structure while keeping every channel's own statistics
identical**. It measures integration, not activity.

Two findings worth calling out, because they show the tool is honest rather
than flattering:
- The **exact MIP** singled out `covenant.integrity` as the weakest link —
  the channel most nearly separable — which is exactly how it was wired
  (weak, slow coupling). The algorithm found where the system is thinnest.
- The **discrete rail reported Φ_s = 0** at the macro grain on 1300 samples
  while the Gaussian rail reported strong integration. Not a bug: 2^12
  states are undersampled by 1300 transitions, so the discrete rail
  *correctly refuses to claim* integration it cannot support at that sample
  size. The interventional rows from the perturbational probe are what
  densify that table over time.

## The bounded claim (shipped on every report)

> An estimate of the integrated information of the system's macro-dynamics,
> under a Gaussian model of its **own measured channels**, at an empirically
> selected grain, validated against a coupling-destroying null with a
> bootstrap CI — **evidence about integration structure, not a consciousness
> meter.**

No number this subsystem emits is presented as a consciousness measurement.
That restraint is enforced in code (`PhiEstimate.claim`) and in the tests.

## Tests

`tests/test_integrated_information.py` (20) — Queyranne vs brute force,
zero-Φ + correct cut on decomposable systems, null-control on independent
channels, detection on coupled/ring systems, grain recovery, exact discrete
Φ on XOR/crosswired vs independent chains, determinism, provenance.
`tests/test_perturbational_probe.py` (12) — LZ76 known values, the
stereotyped-max-response-is-low-complexity signature, Will-governed probe
(runs / refuses / fails closed), and live service integration + governed
persistence.
