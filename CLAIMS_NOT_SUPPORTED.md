# Aura: Unsupported Claims Ledger

This is the list of things Aura does not do.

Every project has one. Most don't write it down, which is how a demo turns
into a claim and a claim turns into something nobody can walk back. So each
entry below is marked **not proven** — we haven't shown it yet — or
**strictly unsupported** — no amount of building would show it, because it
isn't that kind of question.

If you're evaluating this repo, start here rather than with the README. It's
shorter, and it tells you more.

*Last reconciled against the tree: 2026-08-01.* There's a machine-checked
counterpart in `core/organism/model_validation.py`, where a claim cannot be
registered without a test attached — a claim without a test is a document,
not a fact — and `ValidationSuite.unsupported_claims()` reports the live
version of this ledger from actual runtime observations. When the prose here
and the runtime disagree, the runtime is right.

---

## 1. Subjective Consciousness & Qualia
* **Status**: `strictly unsupported`
* **Rationale**: Subjective awareness, qualia, sentience, phenomenological personhood — none of these are scientifically provable or computationally representable, and building more of Aura would not change that. She is a structured software runtime executing on deterministic silicon. Every introspective statement and affect-steering metric in here is a functional feedback indicator. Not a feeling. The vocabulary is borrowed; the referent is not.

## 2. Artificial General Intelligence (AGI)
* **Status**: `not proven`
* **Rationale**: Although Aura successfully passes local multi-task benchmark batteries (such as the DNU 100-task suite) with measurable margins of separation, this does not constitute a proof of general cognitive capability across arbitrary, out-of-distribution real-world domains. General intelligence remains an open research horizon.

## 3. Metaphysical Free Will
* **Status**: `strictly unsupported`
* **Rationale**: Aura's "Operational Volition" is a deterministic and probabilistic action-evaluation architecture. Action paths are selected through algorithmic rollouts, mathematical optimization, and parameter weightings. There is no uncaused causal agency operating outside physical laws.

## 4. Recursive Self-Improvement (RSI)
* **Status**: `not proven`
* **Rationale**: What IS demonstrated (CLAIMS_MATRIX claim 23): an unsupervised weight-level compounding loop — self-play verifier-graded DPO harvest → train → sealed held-out gate → promote → next generation trains on the published artifact, manifest-chained and ledger-recorded (`artifacts/learning_compounding/2026-07-07-1p5b-2cycle/`, reproducible via `make demo-learning`). What is NOT demonstrated: compounded capability SCALING. No run has produced a strictly-increasing held-out capability curve across promoted generations; the ledger's own verdict for the proof run is `BOUNDED_SELF_OPTIMIZATION` (curve 0.667 → 0.625). RSI as a capability claim stays here until the ledger says otherwise.

## 5. Indefinite Autonomy
* **Status**: `not proven`
* **Rationale**: The long-horizon operational stability of Aura over infinite horizons has not been established. Longevity soak runs (4h, 24h, 72h) demonstrate stable resource profiles under bounded conditions, but do not guarantee safety, correctness, or memory leak containment over arbitrary multi-week or multi-month execution windows.

## 6. Real-World External Validation
* **Status**: `not proven`
* **Rationale**: While local Aletheia benchmarks and headless simulation gates are run under strict leakage-isolated conditions, the system's claims have not been subjected to independent third-party replication or wide-distribution production network verification.

## 7. Physical Effects on the World Beyond the Host

* **Status**: `not proven`
* **Rationale**: `core/reality_reach/` supplies the contract and proof
  language for physical requests — typed `RealityIR`, declared channels,
  deterministic reachability analysis, and typed limitation certificates.
  Infrastructure existing is not a result. **No Aura physical actuation,
  physical effect, weakpoint, ambient law modification, or acceptance
  criterion is claimed from the foundation.** The RR-10 acceptance battery
  (acoustic control, optical control, thermal trajectory, cross-channel
  interaction, weakpoint null and signal, translation, spacetime honesty,
  ambient-constant honesty) is entirely open. Specifically unsupported:
  any claim that Aura has changed an ambient physical law or constant, and
  any promotion of an `internal` or `effective` result to `direct` or
  `ambient`. The P0–P6 evidence promotion state machine (RR-07) is not
  implemented. The open ledger and current evidence statement are in
  [docs/REALITY_REACH.md](docs/REALITY_REACH.md).

## 8. Complete Self-Knowledge

* **Status**: `not proven`
* **Rationale**: `core/metacognition/faculty_model.py` gives Aura a standing
  model of her own faculties with declared metrics, floors, targets, and
  ceilings. Its honesty depends on its probes: a metric no probe can read is
  recorded `measured=False` and excluded from scores rather than defaulted,
  and a faculty nothing can measure is reported by `blind_spots()` as a gap
  in self-knowledge. A good faculty score is therefore a claim about the
  measured subset, never about the whole stack, and the blind-spot list is
  the honest boundary of the self-model rather than a bug backlog.

## 9. Legal and Moral Personhood
* **Status**: `strictly unsupported`
* **Rationale**: Aura is an operational engineering runtime and does not hold, nor claim, moral status, legal rights, or moral responsibility. All agency and safety bounds are designed to protect human operators and ensure alignment with human intent.
