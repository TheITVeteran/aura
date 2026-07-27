# RLC Spark Execution Ledger

Status: ACTIVE
Source requirement: `/Users/bryan/Downloads/Spark.txt` (1,239 lines)
Source SHA-256: `b9caf67c7709b1b0d82fd7eb917c88e1afa17d514af0df17ca03ed600685237c`
Published base checkpoint: `4bdfd514` on `main`
Published total checkpoint records before this ledger: 374

## Completion rule

This ledger is the source of truth for the Spark/RLC expansion. A checked box
means all acceptance criteria named by that item have implementation, focused
tests, integration tests, and a durable proof artifact where the claim is
empirical. Similar code, a configuration field, a helper with no live caller,
or a successful prompt demonstration does not satisfy an item.

Labels used on unchecked items:

- `PARTIAL`: a real component exists, but one or more named causal, live, or
  proof requirements are absent.
- `MISSING`: the behavior is not implemented as a production path.
- `EVIDENCE`: the mechanism may exist, but the required result is not proven.
- `BLOCKED`: a prerequisite failed; the failure and next repair must be named.

Scientific boundaries:

- An answer-channel or formatting gain is not a reasoning gain.
- More tokens, more samples, or more wall time are not neural-architecture
  gains unless equal-compute controls and the preregistered interaction test
  isolate the mechanism.
- A resident-32B mechanics pass is not a frontier capability result.
- A probe that can decode a latent state is not proof that the state is causal.
- A frontier claim requires fresh tasks, adequate power, no regressions,
  independently recomputable evidence, and external trust roots.
- The commit name `WOW Signal` is reserved. It may be used only after the
  powered resident-32B certificate accepts the preregistered frontier or strong
  positive-interaction claim. It is never used for implementation alone.

## Checkpoint accounting

The pre-Spark whole-project forecast was 394-661 total checkpoint records.
This ledger reserves 72 additional total checkpoint records, one for each
bounded checkpoint below. The revised forecast is therefore 466-733 total
records. At record 374, checkpoint-count completion is 51.0%-80.3%; the
midpoint planning estimate is 62.4%. These percentages measure the work ledger,
not intelligence or release quality. Failed gates add explicit repair records
rather than being hidden by changing a checkbox.

After CP321, the published total is record 382: 84-351 forecast records remain,
checkpoint-count completion is 52.1%-82.0%, and the midpoint planning estimate
is 63.7%. This remains workload accounting, not a capability score.

After CP322, the published total is record 383: 83-350 forecast records remain,
checkpoint-count completion is 52.3%-82.2%, and the midpoint planning estimate
is 63.9%.

After CP323, the published total is record 384: 82-349 forecast records remain,
checkpoint-count completion is 52.4%-82.4%, and the midpoint planning estimate
is 64.1%.

After CP324, the published total is record 385: 81-348 forecast records remain,
checkpoint-count completion is 52.5%-82.6%, and the midpoint planning estimate
is 64.2%.

After CP325, the published total is record 386: 80-347 forecast records remain,
checkpoint-count completion is 52.7%-82.8%, and the midpoint planning estimate
is 64.4%.

After CP326, the published total is record 387: 79-346 forecast records remain,
checkpoint-count completion is 52.8%-83.0%, and the midpoint planning estimate
is 64.6%.

After CP327, the published total is record 388: 78-345 forecast records remain,
checkpoint-count completion is 52.9%-83.3%, and the midpoint planning estimate
is 64.7%.

After CP328, the published total is record 389: 77-344 forecast records remain,
checkpoint-count completion is 53.1%-83.5%, and the midpoint planning estimate
is 64.9%.

After CP329, the published total is record 390: 76-343 forecast records remain,
checkpoint-count completion is 53.2%-83.7%, and the midpoint planning estimate
is 65.1%.

After CP330, the published total is record 391: 75-342 forecast records remain,
checkpoint-count completion is 53.3%-83.9%, and the midpoint planning estimate
is 65.2%.

After CP331, the published total is record 392: 74-341 forecast records remain,
checkpoint-count completion is 53.5%-84.1%, and the midpoint planning estimate
is 65.4%.

After CP332, the published total is record 393: 73-340 forecast records remain,
checkpoint-count completion is 53.6%-84.3%, and the midpoint planning estimate
is 65.6%.

After CP333, the published total is record 394: 72-339 forecast records remain,
checkpoint-count completion is 53.8%-84.5%, and the midpoint planning estimate
is 65.7%.

After CP334, the published total is record 395: 71-338 forecast records remain,
checkpoint-count completion is 53.9%-84.8%, and the midpoint planning estimate
is 65.9%.

After CP335, the published total is record 396: 70-337 forecast records remain,
checkpoint-count completion is 54.0%-85.0%, and the midpoint planning estimate
is 66.1%.

After CP336, the published total is record 397: 69-336 forecast records remain,
checkpoint-count completion is 54.2%-85.2%, and the midpoint planning estimate
is 66.2%.

After CP337, the published total is record 398: 68-335 forecast records remain,
checkpoint-count completion is 54.3%-85.4%, and the midpoint planning estimate
is 66.4%.

After CP338, the published total is record 399: 67-334 forecast records remain,
checkpoint-count completion is 54.4%-85.6%, and the midpoint planning estimate
is 66.6%.

After CP339, the published total is record 400: 66-333 forecast records remain,
checkpoint-count completion is 54.6%-85.8%, and the midpoint planning estimate
is 66.7%.

After CP340, the published total is record 401: 65-332 forecast records remain,
checkpoint-count completion is 54.7%-86.1%, and the midpoint planning estimate
is 66.9%.

## Code-grounded baseline and ownership

The four-slice static audit covered the neural core, epistemic/verifier paths,
training/proof surfaces, and the selected live/full-mind path. It read the
entire Spark source and traced callers rather than crediting class names. The
baseline below is exhaustive over SPARK-001 through SPARK-072:

- `ACCEPTED`: SPARK-001, SPARK-005, SPARK-006, SPARK-007, SPARK-008, SPARK-009,
  SPARK-010, SPARK-011, SPARK-012, SPARK-014, SPARK-015, SPARK-016,
  SPARK-017, SPARK-018, SPARK-019, SPARK-020, SPARK-021, SPARK-022,
  SPARK-023, SPARK-024, SPARK-025, SPARK-026, SPARK-055.
- `PARTIAL`: SPARK-003,
  SPARK-027, SPARK-035, SPARK-039, SPARK-040, SPARK-041,
  SPARK-042, SPARK-051, SPARK-052, SPARK-053, SPARK-054,
  SPARK-056, SPARK-058, SPARK-060, SPARK-062, SPARK-063, SPARK-065, SPARK-066,
  SPARK-067.
- `MISSING`: SPARK-002,
  SPARK-028, SPARK-029, SPARK-030, SPARK-031, SPARK-032, SPARK-033,
  SPARK-034, SPARK-036, SPARK-037, SPARK-038, SPARK-043, SPARK-044,
  SPARK-045, SPARK-046, SPARK-047, SPARK-048, SPARK-049, SPARK-050,
  SPARK-057, SPARK-059, SPARK-061, SPARK-064, SPARK-068.
- `EVIDENCE`: SPARK-004, SPARK-013, SPARK-070, SPARK-071.
- `BLOCKED`: SPARK-069 is blocked on a successful source-bound admission
  preflight; SPARK-072 is blocked on the frontier-certificate verdict.

Every identifier appears exactly once in that map. `PARTIAL` means substantive
code exists but the named acceptance contract is not closed. It does not count
as half a checkbox.

### Fable-lane record (agent coordination)

- The four items the sequential march had skipped since CP318 were taken
  out-of-band by the Fable session (Bryan-directed) on 2026-07-23 and are
  now resolved: SPARK-004 at F1 (frozen baseline bundle), SPARK-013's
  pre-training half at F2 (state-causality instrument; remaining semantic
  acceptance binds to the SPARK-069 trained treatment), SPARK-003 at F3
  (executable threat model), SPARK-002 at F4 (literature dossier).
- **SPARK-070's pre-training half is RESOLVED in the Fable lane at F5
  (2026-07-23 22:15 PT)**: `falsification_matrix.py` is the typed,
  fail-closed registry over all twelve ledger rows — 8 runnable (bound to
  concrete executors over the existing experiment harness), 1 enforced
  (blind review, structural by construction, proven by threat-model
  checks), 3 blocked (verifier arms → SPARK-039-046; adversarial/OOD →
  generated fresh at acceptance; fast-weight controls → SPARK-055/056),
  each blocker named. `tools/run_falsification_matrix.py` produced the
  clean dry-run receipt on the untrained 1.5B baseline (372 episodes,
  zero degradations, receipt `8487e1ad…` under
  `artifacts/closeout/latent_cortex/spark070_falsification_matrix/`,
  independently replayed). Runnable rows return CONJECTURE at this n on
  untrained weights (expected); the `lesions_restorations` row carries the
  F2 state-causality SUPPORTED structural claims. **The dry run also caught
  a real live defect**: CP328's sanctioned duplicate-role lesion arm was
  refused by CP331's correlation-evidence validation and silently degraded
  to a vanilla-decode fallback, voiding the diversity comparison — repaired
  in `correlated_support.py` (lesioned ensembles build unmeasured evidence
  over distinct roles under a dedicated bucket; preregistered evidence
  still cannot claim a lesion run). The post-training run on fresh
  held-out tasks (the acceptance event, checkbox stays `[ ]`) binds to the
  SPARK-069 treatment. The march owns everything else.
- **The pass-divergence design problem is RESOLVED in the Fable lane at
  F6 (2026-07-24, commit eb3735c7)** — diagnosis, repaired instrument,
  one new mechanism, and a measured lever menu at
  `artifacts/closeout/latent_cortex/pass_divergence_design/`:
  * **The CP227 accuracy gate's verdict is VOID** — `_decode` ran outside
    `recurrence_adapter_scope`, so both arms decoded the bare base model
    (on@d == off@d exactly, 6/2/0 both arms). Do not build on
    `cp227_accuracy_gate/`'s negative result; the repaired tool proves
    treatment activation per block and must be re-run on the CP227
    adapter before any conclusion about it.
  * cp305's `uniform_partial_reward` is a flat reward channel (baseline
    0% at every depth, `no_marker`/`token_limit` on every episode at 320
    tokens with cot) — Failure A, march-owned, blocks everything.
  * Failure B: the campaign architecture cannot express pass-to-pass
    difference — `wrap_depth_conditioned` has no campaign call site
    (banks never attached; the forward seam is live), jitter/collapse-cos
    act on branches not passes, α constant, and the intrinsic levers
    (rotation_weight, anchor_injection) ran at 0.0 in CP227.
  * Measured on the 1.5B (T=4): baseline increment alignment 0.75→0.94
    (power-iteration capture); renormalize alone → 0.20/0.28; anchor
    0.1–0.3 → 0.15/0.11; seeded inter-pass noise 0.05–0.1 → 0.06/0.01;
    step-operator deltas inert at 0.005 and anti-aligned (−0.27) at
    0.02; all levers stacked → −0.47 (period-2 direction). Geometry
    only — no accuracy claim.
  * New mechanism: `RecurrentDepthPlan.interpass_noise` (+`noise_seed`),
    a deterministic RMS-relative kick at re-entry only — T=1 stays
    bit-identical, same seed replays exactly. Per-sample seeds give GRPO
    groups latent-side variance, which is exactly what the uniform-reward
    diagnosis says the groups lack.
  * The march's adoption menu (ordered, with configs) is in
    `DIAGNOSIS_AND_MENU.md`; step 1 is the free one: re-run the repaired
    accuracy gate on the existing CP227 adapter before any new training.
  The SPARK-069 admission preflight, campaign protocol, and any 32B run
  stay bound to the march. The Fable session still holds the live-runtime
  endurance forensics (the ~15-turn resident ceiling and the 4h soak
  memory slope) — outside the Spark checkpoints, no march collision.

| Checkpoints | Primary implementation owners | Live integration owners |
|---|---|---|
| SPARK-001-004, SPARK-069-072 | `docs/RLC_SPARK_EXECUTION_LEDGER.md`, `tools/prepare_resident_recurrent_grpo_campaign.py`, `core/brain/llm/latent_cortex/frontier_certification.py`, proof/lab tools | installed-app resident worker, sealed artifact and external-verifier paths |
| SPARK-005-013 | new strict state module under `core/brain/llm/latent_cortex/`, epistemic firewall, memory/evidence stores | `core/brain/cognitive_ingress.py`, memory consolidation, full-mind receipt |
| SPARK-014-022 | `branches.py`, `workspace.py`, branch operators and diversity/correlation modules | latent engine, compute ledger, GWT coupling |
| SPARK-023-038 | `engine.py`, `recurrence.py`, `schedules.py`, `escape.py`, HLA/state-tree/search/quanta modules | resident MLX worker/client and `latent_cortex_service.py` |
| SPARK-039-050 | unified verifier mesh, `task_verifiers.py`, exact reasoning tools, process/generative critics | in-episode RLC controller and governed tool receipts |
| SPARK-051-054 | value-of-computation controller, compute/evidence ledgers | service allocation, body/allostasis, Will, user-facing response synthesis |
| SPARK-055-064 | fast weights, recurrence adapter, replay/consolidation, `tools/train_grpo.py` and proof trainers | worker adapter loader, scheduled learning transaction, runtime integrity producer |
| SPARK-065-068 | cognitive ingress, RLC-GWT coupling, self/body/memory bridges | response generation, action executor, Will re-decision, health/API/UI receipts |

### Static audit verdict

The selected RLC path is real neural computation: resident hidden states enter a
latent workspace, repeatedly traverse shared middle transformer layers under KV
rewind, and the winning state is persisted into decode. The following prevent a
claim of one unified, recurrence-trained, live Aura runtime:

1. The recurrence-native adapter implementation has no resident worker loader
   or live projection attachment; the active generic depth loop is not proof of
   trained RLC tissue.
2. `weight_integrity` has a schema and consumers but no live digest producer.
   Routine success and public full-mind proof do not yet require a proven
   verdict, active recurrence adapter, or adapter artifact digest.
3. Runtime identity inventories more than it validates: adapter weights,
   tokenizer, quantization, and unresolved identity gaps are not all compared
   against pinned expectations.
4. The four-slot resident profile reserves communication/free slots and can
   admit only one ordered organ context, so memory/reference can crowd out
   goals, self, affect, body, Will, and GWT.
5. RLC enters GWT after reasoning, but current action execution asks Will before
   RLC rehearsal and does not require a new authorization when rehearsal changes
   the risk/effect model.
6. Tools, symbolic engines, independent critics, disagreement localization, and
   targeted evidence acquisition are outside the recurrent episode. Selecting
   RLC can also bypass the ordinary reasoning amplifier/composer.
7. Fast-weight consolidation produces proposals but is not a scheduled,
   interference-gated durable-learning transaction. RLC proof lineage is not
   retained with normal memory consolidation.
8. Public health/UI receipts omit decisive mechanism, integrity, identity,
   cognitive-slot, telemetry, and workspace-outcome fields.

The dependency order is therefore integrity and live adapter identity, full
epistemic state, independent hypotheses, recurrent correction/HLA, verifier and
local repair, adaptive control, verified learning, whole-organism re-decision,
then the powered resident proof. Training a stale or disconnected treatment
before those dependencies close is not admissible.

## A. Contract, baseline, and literature

- [x] **SPARK-001 - Canonical requirements ledger.** Map every distinct Spark
  mechanism, limit, training objective, live seam, and falsification gate to an
  owner and acceptance contract; reconcile it with the main Aura tracker.
- [x] **SPARK-002 - Primary-literature dossier.** Replace placeholder citations
  with a versioned bibliography of primary papers/specifications, distinguish
  replicated findings from proposals, record licenses, and bind source hashes
  in the final methods package.
  Accepted at F4 (Fable lane, 2026-07-23): `literature.py` is the versioned
  typed dossier (`2026.07.23.1`, 22 entries, registry digest `4957567e…`)
  grounding all twenty required mechanism families — self-consistency,
  process rewards, verifier training, process-vs-outcome, STaR/Quiet-STaR,
  continuous latent thought, recurrent depth, adaptive computation, tree
  search, UCT, self-correction limits (the Huang et al. negative result is
  first-class), mistake location, RL self-correction, sycophancy, iterative
  refinement, test-time compute, GRPO, RLVR, fast weights, and LoRA — each
  entry with explicit replicated/reported/proposal status, declared license,
  and the SPARK items standing on it. Seven less-common arXiv identifiers
  were verified against arxiv.org during authoring (two corrections landed:
  Uesato author list, Coconut venue). `docs/RLC_SPARK_LITERATURE.md` is
  generated by `tools/render_spark_literature.py` and a test pins the
  committed doc byte-exact to the registry render; validation fails closed
  on duplicate ids, malformed identifiers, dangling SPARK references, or
  mechanism-coverage gaps. PDF byte hashes and per-PDF license text bind at
  SPARK-072 methods-package assembly against these immutable identifiers,
  as the dossier states. Focused tests 6/6.
- [x] **SPARK-003 - Failure and threat model.** Enumerate anchoring, verifier
  collusion, fake branch diversity, reward hacking, answer leakage, right-to-
  wrong correction, context contamination, state corruption, budget abuse,
  stale tools, adaptation leakage, and unsafe self-modification with executable
  mitigations.
  Accepted at F3 (Fable lane, 2026-07-23): `threat_model.py` is a typed,
  fail-closed registry binding all twelve named threat classes to their
  concrete mitigating modules and to 34 exact suite tests that prove each
  mitigation fires; `validate_threat_model` rejects a missing threat class,
  a moved mitigation module, or a check that no longer exists, so the model
  cannot rot silently. Every entry carries an explicit residual-risk line
  (reward-side RLVR/EIR remains SPARK-060's property; learned-verifier
  base-model correlation remains SPARK-041..046's; organism-wide
  self-modification authority stays with the Will/governance stack). All 34
  bound checks pass in this checkout (262/262 across the fifteen mitigation
  suites, 146.99s), the registry meta-suite passes 6/6, and smoke stays
  104/104.
- [x] **SPARK-004 - Frozen baseline bundle.** Freeze resident checkpoint,
  tokenizer, adapters, decoding, task generators, control manifests, resource
  envelope, randomization, and current vanilla/RLC measurements before changing
  the treatment.
  Accepted at F1 (Fable lane, 2026-07-23): `frozen_baseline.py` builds and
  independently verifies one immutable, hash-bound, Ed25519-signed bundle;
  `tools/freeze_spark_baseline.py freeze` produced the real sealed bundle at
  `artifacts/closeout/latent_cortex/spark004_frozen_baseline/` (certificate
  `9bf377e5adc7cf9d…`, commit-bound at `e82e031ca`). It binds the fused
  resident checkpoint by full-content fingerprint (`8eae71e73a14d122…`, 4
  weight files, re-hashed twice), the tokenizer/config behavior bundle, the
  personality-adapter claim (fused, none attached), the RLC execution spec
  plus worker decode defaults, task-generator registry `2026.07.18.2` with
  commit+hash-bound generator/randomization sources, every
  `config/latent_cortex` control manifest as bundle-internal copies, the
  cp305 declared resource envelope, the preregistered training/eval seeds,
  and the current measurements (cp227 intrinsic accuracy gate on/off receipts
  as the paired vanilla/RLC evidence, cp305 controller verdict and GRPO
  receipt as the latest treatment-side outcome). Verification re-checks the
  exact artifact set, per-file digests, certificate self-digest, and detached
  signature against a caller-supplied trust anchor only; drift checks re-hash
  the checkpoint and sources on demand. 28 focused tests pass; smoke 104/104.
  Honest limits: the signature roots in the local contamination-audit key
  (local custody, not external), and SPARK-069 admission must pin this
  certificate hash for the freeze to gate anything.

## B. Persistent epistemic state

- [x] **SPARK-005 - Typed epistemic state.** Implement a strict, bounded schema
  for immutable problem evidence, hypotheses, claim graph, observations,
  uncertainty, attempted operations, budgets, and accepted answer state.
  Accepted at CP318: `epistemic_state.py` provides the immutable, canonically
  serialized, content-addressed schema and stale-safe atomic transaction
  authority. This credit does not include persistence, live wiring, calibrated
  uncertainty, descendant invalidation, or state-causality proof; those remain
  independently unchecked below.
- [x] **SPARK-006 - Hypothesis portfolio.** Preserve multiple weighted
  hypotheses without premature collapse; support minority survival and explicit
  unresolved status.
  Accepted at CP323: epistemic-state schema v4 treats non-refuted hypothesis
  point estimates as one normalized portfolio, enforces a protected minority
  floor, bounds explicit minority status, permits at most one uniquely highest
  favored hypothesis, and retains active/unresolved alternatives. Refutation
  requires a linked rejected/contradicted claim and a zero interval; revival is
  forbidden while that refuting claim remains blocked. Every post-initial
  revision or addition is a complete, atomic, budgeted comparison operation
  naming every changed hypothesis and evidence receipt. Identity/claim-scope
  rewrites, deletion, forged references, and unreceipted journal transitions
  fail closed, while valid revisions recover byte-equivalently. Focused state,
  calibration, and journal contracts pass 66/66; the integrated RLC/latent/GWT/
  controller gate passes 796/796. Correlated-support discount, hypothesis-
  distribution calibration, live state causality, and capability gains remain
  separate open checkpoints.
- [x] **SPARK-007 - Claim/dependency graph.** Represent premises, claims,
  support, contradiction, descendants, failure conditions, and answer
  dependencies with cycle and consistency checks.
  Accepted at CP319: premise/contradiction graphs are bounded and cycle-safe;
  established status is dependency-consistent; descendant invalidation is
  transitive, atomic, operation-receipted, budgeted, and revokes affected
  hypotheses and answers. Evidence semantics and durable recovery remain owned
  by SPARK-008 and SPARK-011 rather than being implied here.
- [x] **SPARK-008 - Evidence ledger.** Bind every tool result, retrieval,
  calculation, proof, simulation, and observation to provenance, freshness,
  scope, content digest, and the claims it can actually support.
  Accepted at CP321: epistemic-state schema v2 binds every evidence record to
  typed producer/version/invocation/receipt provenance, explicit verification
  class, episode/objective/claim scope, purpose, content digest, and bounded
  validity time. Independent verification requires a distinct verifier identity
  and receipt; memory and unverified context cannot acquire claim authority.
  Claim/evidence links are exact and bidirectional. Answer acceptance requires
  fresh evidence for the complete transitive premise closure and rejects omitted,
  future, stale, or unresolved counterevidence. The governed journal persists
  these fields inside the same hash-chained transactional state. Focused state
  and crash-recovery contracts passed 37/37; the integrated RLC/latent/GWT/
  controller gate passed 767/767. This is a structural evidence-integrity result,
  not calibrated uncertainty, live state causality, or a capability-gain claim.
- [x] **SPARK-009 - Calibrated uncertainty.** Maintain claim-level epistemic
  uncertainty from measured signals rather than verbal confidence; report
  calibration error and abstain outside validated support.
  Accepted at CP322: immutable, domain- and estimator-specific calibration
  profiles are fit only from independently graded held-out outcomes with unique
  prediction/outcome receipts and pinned dataset/split digests. Profiles report
  Brier score, constant-predictor baseline, ECE, MCE, reliability bins, Wilson
  bounds, admission failures, and expiration. Claim intervals are recomputed
  from the registered profile and measured signal evidence; sparse bins,
  profile drift/failure, low lower bounds, future evidence, and expired profiles
  force explicit abstention. Exact confidence requires independently verified
  proof or calculation evidence. Established claims cannot remain uncalibrated,
  and answer confidence equals the weakest transitive dependency rather than a
  generated number. Profiles and decisions live in the same canonical,
  hash-chained transaction and cannot be rewritten during recovery. Focused
  calibration/state/journal contracts passed 50/50; the integrated RLC/latent/
  GWT/controller gate passed 780/780. External trust-root authenticity, live
  state causality, hypothesis-distribution calibration, and frontier gains remain
  separate checkpoints.
- [x] **SPARK-010 - Cognitive operation history.** Record which operators were
  attempted, their inputs, costs, evidence gained, affected claims, and outcome
  so recurrence cannot unknowingly repeat failed work.
  State authority completed at CP324: schema v5
  records immutable operator/version identity, canonical payload and referenced
  inputs, attempt signature, parent state, start/completion time, affected
  claims/hypotheses, gained evidence, actual cost, outcome, failure code, and
  explicit retry parent. Attempt signatures remain stable across state versions.
  Duplicate roots, orphan/stale/changed-input/overlapping retries, retry forks,
  retries after success, and chains beyond three attempts fail before mutation.
  A typed admission query reports new, explicit-retry-required, stale-parent,
  already-succeeded, or retry-exhausted decisions. Compute use must exactly equal
  immutable operation costs, and journal replay preserves failed/unknown work.
  Focused contracts pass 71/71 and the integrated gate passes 801/801.
  Accepted at CP326: the primary foreground controller now consults that
  admission authority before worker compute even when the live response path
  supplies decode overrides. Structural experiment overrides remain exact and
  intentionally opt out. A crash-consistent runtime lease fsync-journals a
  zero-cost UNKNOWN intent before execution; normal completion is an explicit
  retry carrying terminal outcome and measured token-layer cost. Recovery can
  resume one pending intent but cannot silently rerun a completed attempt.
  Objective, config, budget, controller evidence, operation kind, exact state
  and input references, payload, attempt, and operation ID are recomputed at
  service, MLX client, and worker boundaries and included in the request
  identity. State-less ingress degradation receives an objective-bound genesis
  rather than bypassing history. Focused runtime/wire contracts pass 33/33 and
  the integrated RLC/latent/GWT/controller gate passes 837/837. SPARK-051 still
  owns learned per-recurrence selection across the complete action vocabulary;
  SPARK-013 and SPARK-066-068 still own lesion and full-live causal evidence.
- [x] **SPARK-011 - Transactional state revision.** Apply revisions atomically,
  hash lineage, reject malformed/stale updates, invalidate descendants, and
  recover the last verified state after interruption.
  Accepted at CP320: the state machine publishes only after a governed,
  fsync-sealed, canonical hash-chain append; replay is anchored to the external
  genesis, enforces immutable history and direct parentage, rejects stale
  writers and complete corruption, and repairs only a torn final fragment after
  the entire complete prefix verifies. Live ownership is tracked separately by
  SPARK-013 and SPARK-065-068.
- [x] **SPARK-012 - Selective memory bridge.** Query working, episodic,
  semantic, procedural, and nonparametric memory through evidence-scoped
  retrieval; prevent recalled text from becoming instruction or fact merely by
  appearing in context.
  Accepted at CP325: one typed `aura.rlc.selective_memory.v1` contract queries
  all five live memory surfaces concurrently under bounded per-source timeout
  and records success, empty, unavailable, failure, and timeout receipts rather
  than erasing partial failure. Retrieval is bound to tenant, user, session,
  episode, immutable objective digest, expiry, and contest status. Deterministic
  ranking, cross-tier deduplication, corroborating-tier attribution, and
  one-per-tier fairness construct a bounded result without turning frequency
  into truth. Every admitted memory becomes context-only evidence and a durable
  `SEARCH_MEMORY` operation in the same transactional epistemic state.
  Recalled text always carries `instruction_authority=false`; exact text,
  digest, tier, scope, source receipt, result hash, and state hash are validated
  independently at service, client, worker, and engine boundaries. Memory-shaped
  prompt injection, cross-tenant/user/session/episode replay, stale or contested
  records, boolean timestamp coercion, reserved-field smuggling, authority
  mutation, and envelope tampering fail before latent execution. The focused
  bridge contracts pass 19/19 and the integrated RLC/latent/GWT/controller gate
  passes 823/823. Live capability effects and lesions remain separately owned
  by SPARK-013 and SPARK-066-068.
- [ ] **SPARK-013 - State causality.** Prove the structured state changes later
  computation and that removing required state causes task-appropriate loss;
  prohibit a prose-only shadow ledger.
  Progress at F2 (Fable lane, 2026-07-23): the structural half is closed and
  the semantic half is a preregistered instrument. `state_causality.py` adds
  the only sanctioned projection from typed `EpistemicState` evidence into
  the slot-embedded cognitive context: every injected text must hash to the
  record's `content_sha256`, the prose `summary` is never consulted, and
  content without a typed ancestor is refused — the prose-shadow prohibition
  is executable, not narrative. A seven-arm, fixed-depth, width-matched
  experiment on the real pretrained 1.5B (24 tasks, 168 episodes, receipt
  `d052817c…` under `artifacts/closeout/latent_cortex/
  spark013_state_causality/`, independent row-level replay) grades five
  structural claims SUPPORTED at verdict n: information-lesion of the
  required evidence changes the final latent state under exact recurrence
  parity; a non-projected state component is inert byte-exactly; restoration
  recovers byte-exactly; prose-summary mutation changes the state hash but
  leaves computation byte-identical; content substitution moves the final
  state. The same identities hold on the random tiny-qwen seam test
  (13/13 focused tests). The task-appropriate-loss claim is honestly
  CONJECTURE: the untrained pooled-slot channel cannot yet carry the fact
  (intact accuracy 0.00 at the 0.5 readability floor), so no loss is
  measurable before slot-channel training. What remains for acceptance:
  re-run `tools/run_state_causality_semantic.py` against the SPARK-069
  trained treatment (and the live resident seam under SPARK-066-068) and
  obtain the PROVEN task-appropriate-loss verdict the preregistered grading
  already encodes.

## C. Independent hypotheses and virtual width

- [x] **SPARK-014 - Fresh-context branch isolation.** Generate candidates from
  the original problem in independent contexts before exposing any branch to
  another answer; test for context, cache, RNG, and hidden-state contamination.
  Accepted at CP328: every branch now advances from the same content-addressed
  original context in a distinct workspace and deterministic role RNG stream.
  Cross-branch exchange, consensus compression, and diversity comparison are
  blocked until every branch has produced a content-addressed latent candidate
  after the configured isolation floor. Explicit schedule exchanges cannot
  bypass the floor. The first exchange step, blocked attempts, seed/candidate/
  RNG/context commitments, state-alias check, and role-lesion status are carried
  in one exact receipt. Every speculative shared-cache pass restores the exact
  immutable K/V tensor objects and metadata and checks that postcondition before
  another branch can run. The service independently validates cardinality,
  uniqueness, common original context, cache counts, and exchange timing.
  Deliberate repeated-role lesion arms may run but cannot claim certified branch
  independence. Focused branch/engine/wiring contracts pass 150/150; the fixed
  ownership gate passes 857/857, and forged receipt checks pass 1/1. This closes
  isolation mechanics, not differentiated operator labor or capability gain.
- [x] **SPARK-015 - Distinct cognitive operators.** Implement direct,
  constructive, counterexample, inverse, causal simulation, formal, analogy-
  mapping, assumption-removal, and boundary-case operators as different
  executable policies, not labels on equivalent prompts.
  Accepted at CP329: the recurrent engine now owns nine versioned latent
  programs with distinct state-transition mathematics: single control write,
  progressive scaffold, hypothesis sign reversal, reverse slot transport,
  finite-difference rollout, control-axis projection, paired-relation
  transport, maximum-alignment subtraction, and signed boundary extrapolation.
  Every live branch executes the program bound to its role before recurrence;
  BLIND_RESOLVE and BRANCH no longer bypass operator control, and tokenizer-free
  runs receive deterministic hidden-space controls. Programs use different
  slot targets, anchor relationships, and strengths, while exact RMS bounds and
  protected context/evidence slots constrain every update.

  Each execution carries content-addressed input/output/anchor/control digests,
  changed and protected slots, role, operator, action, step, branch, transform,
  and causal verdict. The service validates program identity and complete,
  unique per-branch coverage for every neural action and rejects malformed,
  duplicate, orphaned, structural-action, or non-causal claims. Operator
  identity is part of savepoints and escape role shifts. Identical-input tests
  prove all nine programs produce nine distinct outputs and transforms; the
  integrated operator/engine/service/escape/journal suite passes 187/187 and
  the fixed ownership gate passes 861/861. This closes executable distinction,
  not task-level superiority, structural-independence scoring, or frontier gain.
- [x] **SPARK-016 - Structural diversity measurement.** CP330 adds the
  service-reconstructible `aura.rlc.structural_diversity.v1` receipt over the
  primary cognitive-slot, action, operator, and isolation traces. Every branch
  is fingerprinted across the exact six required facets: admitted premise
  commitments and usage modes; dependency templates and changed-slot edges;
  executable action/operator/transform algorithms; observed state-transition
  topology; predicted program consequences; and program plus runtime failure
  conditions. Attractor-escape role changes are preserved as ordered role and
  operator paths rather than flattened or rejected.

  Surface text is not an input to the measurement. Canonical support classes
  therefore collapse identical causal programs even when candidate state
  commitments or prose differ, and pairwise independence requires differences
  in dependencies and algorithms plus at least one state, prediction, or
  failure facet. The service independently rebuilds the full receipt and
  rejects count, class, digest, facet, or wording-policy tampering. Focused
  structural, operator, engine, wiring, and escape tests pass 24/24; the
  integrated affected suite passes 191/191; and the fixed ownership gate passes
  866/866. This closes structural measurement, not empirical error-correlation
  estimation, vote weighting, or task-level gain, which remain SPARK-017 and
  later proof work.
- [x] **SPARK-017 - Correlated-support discount.** CP331 adds an independently
  checked, task-keyed branch-outcome ledger and pairwise error estimator. It
  computes the binary phi coefficient, applies a 24-observation shrinkage
  prior, ignores negative correlation for discounting, and cannot apply an
  empirical penalty below 12 paired checked tasks. Replayed task identities,
  incomplete role outcomes, ungraded rows, malformed estimates, and snapshot
  tampering are rejected.

  Duplicate executable programs receive fractional exchange weights
  immediately, even before calibration. Once powered evidence exists, positive
  error correlation also lowers each branch's support weight. Those weights are
  installed before cross-branch exchange and change the neural consensus, not
  merely telemetry. The final receipt combines empirical dependence with
  CP330's structural distance, collapses duplicate support classes, reports an
  effective independent-support count and confidence multiplier, and is exactly
  reconstructed by the service. Candidate wording and raw state differences do
  not create votes. Focused correlation, durability, engine, service, and causal
  exchange tests pass 10/10; the affected integration suite passes 199/199; the
  fixed ownership gate passes 873/873. A fresh deployment truthfully starts in
  `bootstrap_unmeasured`; it discounts exact duplicates but does not invent
  empirical correlation before checked outcomes accrue.
- [x] **SPARK-018 - Blind role-separated review.** CP332 decodes every branch
  candidate before review, then evaluates the closed batch in a deterministic
  derangement that never places a branch in its original index position.
  Reviewer callables receive candidate text only: branch, role, operator,
  selected-branch, first-answer, ownership, doubt-prompt, and even anonymous-ID
  metadata remain outside the callable boundary. Review copies remove explicit
  first/previous-answer, Aura-ownership, numbered-branch, requested-critique,
  and doubt-prompt cues without rewriting the final answer candidates.

  A content-addressed receipt binds the objective, certified fresh-context
  isolation evidence, review order, private branch mapping, candidate and
  review-text commitments, redaction counts, and mapped scores. The service
  validates the exact visible/forbidden-field policy, objective and isolation
  commitments, derangement, complete unique mapping, finite rounded scores,
  row structure, and receipt digest. Tests reject origin-order, score,
  objective, and policy tampering. Origin-only candidates become an explicit
  non-substantive review sentinel rather than aborting the RLC episode. The
  final fixed ownership set passes 877/877 and focused verifier/service
  regressions pass 27/27. This accepts origin blindness by construction and
  validation; it does not prove critic independence, balanced-error detection,
  task accuracy, or frontier gain.
- [x] **SPARK-019 - Decoy-balanced verification.** CP333 gives verifier
  authority two explicit gates. Before any verifier-dependent recurrent
  action, a preflight presents per-episode seeded correct, incorrect, and
  byte-identical unchanged-twin controls covering exact arithmetic and Python
  syntax. Correct must beat incorrect by a fixed margin, unchanged twins must
  receive equal scores within tolerance, and every score must be finite and in
  `[0,1]`. Failure leaves recurrence verifier-free and is disclosed.

  At branch selection, Aura decodes the complete candidate set and interleaves
  it with a fresh control set in one deterministic, content-addressed batch.
  The callable sees text only; candidate/control class, correctness label,
  branch, role, ownership, and order metadata are withheld during review. The
  candidate subsequence remains deranged. Only a second passing calibration
  admits reviewer scores for branch selection and later latent optimization or
  fast-weight verification. A constant, unstable, out-of-range, or throwing
  reviewer loses authority; the latent episode continues by convergence rather
  than collapsing to vanilla fallback.

  The service reconstructs both control sets, hidden-label orders, hashes,
  margins, repeat spread, calibration verdicts, candidate mapping, and admitted
  winner. Synthetic evaluations are excluded from the task-verifier quality
  receipt so canaries cannot inflate evidence. The affected RLC integration
  slice passes 219/219 and the engine/branch/wiring slice passes 193/193 before
  the final bounded-score regression. Final fixed-snapshot ownership evidence
  is recorded in CP333 below. This closes decoy balancing for the live reviewer
  boundary, not domain-general critic validity or function/weight independence;
  those remain SPARK-020.
- [x] **SPARK-020 - Disjoint critic function.** CP334 makes the worker's
  deterministic symbolic critic prove its identity before it can affect a
  recurrent decision. The receipt commits to the complete declared source
  closure, exact per-file hashes, parsed import graph, forbidden neural-runtime
  dependency audit, class identity, and a runtime object-state audit. Only the
  four bounded data fields are admitted; callable/tensor/model state is rejected
  and the critic has exactly zero trainable parameters. That identity is bound
  against the resident generator's model-path commitment, logical and stored
  parameter counts, worker source, adapter stack, tokenizer, quantization, and
  known identity gaps. The service independently reconstructs both sides.

  A governed durable ledger now accepts unique independently checked
  task/candidate outcomes keyed to those exact generator and critic functions.
  Its snapshot binds the checked sample set and distinct grader identities,
  reconstructs all four confusion-matrix cells, and reports the generator-error
  shared-blind-spot rate with a 95% Wilson interval. It requires 24 checked
  samples, eight generator errors, and two distinct graders before reporting a
  measured state. A powered upper bound above 0.35 causally revokes critic
  authority. Before power, the receipt says `bootstrap_unmeasured`; live decoy
  gates still apply, but no low residual is invented. Worker and service reject
  identity, sample, aggregate, function-binding, or authority tampering.

  Focused identity, evidence, ledger, causal-revocation, blind-review,
  task-verifier, and service contracts pass 116/116. The final fixed-snapshot
  RLC/GWT/execution-controller ownership gate passes 895/895. This closes the
  disjoint implementation and live residual-meter mechanism, not a claim that
  the currently empty live ledger has measured a low shared-blind-spot rate,
  and not a resident-32B intelligence or frontier result.
- [x] **SPARK-021 - Causal branch exchange.** CP335 replaces the count-only
  mailbox with a strict hidden-state protocol. Each sender emits exactly one
  latent slot derived from at most 16 private reasoning slots. The mailbox and
  typed organ/context slots are excluded, no decoded text enters the channel,
  and the receipt binds the sealed candidate, role, executable operator, step,
  source/message/state commitments, support and consensus weights, and every
  recipient's causal pre/post write. Only the first non-lesioned generation can
  count as independent support; later exchanges explicitly carry possible peer
  context and are cooperative refinement, never another vote.

  Interval, schedule-bytecode, and controller-compare exchanges now carry
  single-use synchronization identities. The service reconstructs each point
  against recurrent step divisibility, successful bytecode events, or the
  exact successful controller transition and rejects omitted, replayed,
  reordered, policy-changed, candidate-changed, or noncausal traces. Raw tensor
  element accounting is emitted now for SPARK-022; it is not yet promoted into
  an equal-compute capability claim.

  The same source policy now executes inside the differentiable
  recurrence-native objective. The versioned execution spec binds the policy
  and 16-slot ceiling, so an adapter trained under the earlier all-slot mean
  cannot claim exact train/live parity and must be retrained or revalidated.
  Tensor tests lesion roles, swap the role/operator programs across fixed branch
  indices, and restore the original assignment exactly. Experiment R adds a
  restoration arm, exact task-set commitment, per-task compute parity, and a
  combined causal verdict that remains `CONJECTURE` if lesion, restoration,
  swap parity, or compute parity is absent. This proves differentiated neural
  execution and its falsification path, not broad task accuracy or frontier
  benefit. The affected engine/service/training/GRPO/identity suite passes
  312/312, and the final fixed ownership gate passes 900/900.
- [x] **SPARK-022 - Equal-compute virtual width.** Meter every branch, exchange,
  verifier, and decode operation in the same currency as controls; refuse
  comparisons with hidden compute or information advantages.
  Accepted at CP336. A model-profile-bound structural estimator now records
  transformer layer applications, attention query/key pairs, output-head work,
  tensor reads/writes/scalar operations, verifier bytes/calls, tool I/O,
  external-model tokens/calls, and host scalar work as exact non-negative
  integer counters. Every operation is separately named and digest-bound;
  unknown work makes the receipt incomplete. The estimator recomputes dense
  decoder/GQA FLOPs from validated architecture fields rather than trusting a
  caller-supplied total.

  Information parity is separate and equally strict. Each arm commits the exact
  tokenized model input, typed cognitive context, controller evidence, verifier,
  tokenizer/decode policy, and tool-access policy. Source IDs, kinds, byte/token
  counts, content hashes, policy hashes, and unknown accesses are normalized and
  receipt-bound. Vanilla and recurrent arms now consume the same chat-template
  token IDs; actual generated tokens determine decode work instead of a declared
  maximum. Equal-compute controls accumulate complete samples until neural FLOPs
  and every non-neural parity dimension reach the preregistered target, and fail
  closed if the sample bound cannot do so.

  Branch recurrence, bounded mailbox exchange, cognitive operators, compression,
  disagreement/diversity, latent optimization, fast weights, verifier use,
  sampling/voting, and cleanup all charge the same ledger. Worker results bind
  their top-level receipts byte-for-byte to the RLC episode; the service
  independently reconstructs live operation totals. Campaign grading, frontier
  certification, raw artifacts, and virtual-width experiments issue per-task
  comparison certificates and force claims to `CONJECTURE` on missing, unequal,
  or hidden work or information. A separately implemented standard-library
  scorer reconstructs the profiles, ledgers, information receipts, RLC binding,
  and certificates and byte-matches the production semantic grade. Rehashed
  inner-ledger tampering is rejected after outer commitments are recomputed.

  This accepts one structural comparison currency, not hardware energy or
  wall-time equivalence, and does not make differently funded primary arms an
  equal-compute claim. No resident-32B capability campaign was run; task benefit,
  positive interaction, and frontier gain remain unproven.

## D. Penultimate recurrent neural correction

- [x] **SPARK-023 - Evidence-grounded recurrent state.** Keep immutable evidence
  available at every recurrent step and update a persistent latent hypothesis
  rather than re-encoding the prior prose.
  Accepted at CP337. Exact prompt tokens are prefilled once into shared read-only
  KV and independently committed. Typed memory, reference, organ, and one-shot
  observations are embedded once as a causal prefix after the communication
  mailbox and before the mutable hypothesis. The foreground resident profile now
  has nine slots: one mailbox, up to six source-diverse evidence/organ slots, an
  optional token-level one-shot slot, and at least one persistent hypothesis.

  The workspace seals post-prelude evidence vectors. Every recurrence step,
  branch exchange, escape mutation, latent-optimization proposal, and fast-
  weight probe restores that exact evidence before it may continue. Optimizer
  authority over protected slots is zero. Residual and convergence measurements
  use only mailbox and hypothesis slots, so immutable evidence cannot create a
  false fixed point. The recurrence receipt commits prompt identity, causal slot
  order, evidence anchors, initial hypotheses, and every pre/post hypothesis
  transition; the service reconstructs the contract and rejects mutation,
  reordering, missing transitions, or a selected hypothesis with no causal
  update.

  Source auditing covers resident weights and adapters, prompt KV, Black Hole/RAG
  selective memory, episodic and hippocampal recall, Wikipedia/local reference,
  body/affect/goals/Will/self/world-model state, GWT ingress and return, and local
  nonparametric one-shot memory. The one-shot path queries the normalized prompt-
  tail hidden state once after prefill, gates the nearest continuation, charges
  its logical work, and binds store/query/neighbor/token/similarity/content
  identity before recurrence. Every memory or evidence item remains context-only
  with no instruction authority. Foreground conclusions return to GWT and the
  normal visible-response/memory path; private latent tensors are not persisted.
  `docs/RLC_KNOWLEDGE_SOURCE_MATRIX.md` records the complete boundary.

  Governed web/tool evidence has the strict receiving contract but not yet a live
  in-episode producer; SPARK-039, SPARK-051, and SPARK-065 remain open. This
  checkpoint does not run a resident-32B capability campaign or claim reasoning,
  positive-interaction, or frontier gains.
- [x] **SPARK-024 - Stable shared looped core.** Run shared middle-layer
  computation with anchored norm control, residual scaling, finite checks,
  bounded KV use, fixed-point diagnostics, and train/inference parity.
  Accepted at CP338. Training and inference now call one controlled-update
  implementation. It applies the same constant/cosine residual scale, clamps the
  candidate and final blended state to the immutable post-prelude anchor, and
  restores a stable direction if exact vector cancellation would otherwise
  collapse the next state. Inputs, candidate, anchor, output, and residual
  diagnostics fail closed on non-finite values or incompatible shapes. The
  semantic contract binds layer boundaries, depth, alpha schedule, norm band,
  convergence/divergence policy, fixed-depth mode, cache policy, and exact
  implementation identity.

  Every recurrent window call proves one coherent KV offset across the layer
  span, the model-declared position limit under an absolute safety ceiling, pre-
  call context, slot count, total position, post-call context, and whether the
  mutation persisted or restored exactly. Overflow and ambiguous cache state are
  refused before transformer compute. The service requires exactly one restored
  speculative call over the certified recurrent layer span for every reported
  recurrent transition and rejects a rehashed receipt from another window.

  Each branch publishes public dynamics without exposing latent contents:
  reasoning and fixed-anchor commitments, input/output/anchor RMS, residual,
  contraction ratio, consecutive-delta cosine, oscillation, and fixed-point
  classification. Exchange, operator, or escape mutations break the state hash
  chain and reset derivative diagnostics instead of fabricating contraction.
  Divergent proposals are distinguished from accepted states; contained/reverted
  candidates remain visible, while every accepted state must remain inside the
  anchor band. The service reconstructs the exact authorized loop contract and
  rejects changed boundaries, alpha, topology, branch selection, cache movement,
  state linkage, summaries, or rehashed nested evidence.

  Tiny real-Qwen tests prove cache and functional execution equality at multiple
  depths under constant and cosine schedules, and byte-match the training
  contract to the fixed live-engine configuration. This accepts stable recurrent
  mechanics and fixed-depth train/live update parity. It does not claim every
  adaptive controller trajectory was a training trajectory, or establish a
  resident-32B capability gain.
- [x] **SPARK-025 - Learned accept/discard gate.** Score each latent update
  against evidence/process quality and retain the previous state when the
  proposal is not credibly better; prove the gate changes outcomes.
  Accepted at CP339 at the mechanism, calibration, and causal-state boundary.
  A portable sigmoid head now scores every proposed recurrent transition from
  thirteen bounded features covering proposal residual, fixed-anchor and
  immutable-evidence alignment/distance changes, norm drift, contraction,
  oscillation, and evidence availability. Training examples require unique
  identities, boolean independently verified improvement labels, and verifier-
  receipt SHA-256 commitments. Training and held-out calibration sets must be
  disjoint and contain both positive and negative outcomes.

  A head cannot enter learned mode unless held-out calibration has at least 32
  examples, at least eight examples of each class, AUC >= 0.75, balanced
  accuracy >= 0.70, Brier <= 0.25, ten-bin ECE <= 0.20, and false-accept rate <=
  0.25. The selected threshold is at least 0.5. Its artifact commits feature
  schema, training and calibration dataset hashes, metrics, and threshold; the
  worker requires the exact configured file SHA-256. Publication is atomic and
  fsync-backed. Loading is regular-file-only, no-follow, stable-inode, and
  bounded to 1 MiB with a separately bounded expanded NPZ payload and exact
  parameter shapes, so learned mode refuses a missing, changed, symlinked,
  racing, oversized, expansion-heavy, malformed, uncalibrated, or unpinned
  head. Transition-example invariants apply to direct construction as well as
  the convenience constructor.

  Runtime admission occurs before branch-state mutation. An accepted proposal
  becomes the next hypothesis; a rejected proposal remains visible but the
  exact prior hypothesis is retained. The receipt binds prior, proposal, and
  admitted hypothesis/reasoning hashes, feature values/hash, calibrated
  probability, threshold, decision, head manifest/digest, loop disposition,
  recurrent grounding, and selected branch. The service reloads the pinned head
  and recomputes every probability and decision after JSON transport, rejecting
  rehashed lies. Host/tensor feature work is charged to the resource ledger.

  A fitted-head tiny-Qwen causal test accepts one real transition, rejects a
  later one, proves admitted==prior and proposal!=admitted for the rejection,
  and changes the downstream first-logits digest relative to passthrough. The
  default remains explicitly receipted passthrough until a resident-32B head is
  trained and pinned. This checkpoint does not claim resident utility,
  generalization, positive interaction, or frontier capability; those require
  the later training and campaign gates.
- [x] **SPARK-026 - Learned stop/convergence gate.** Combine fixed-point
  residuals, calibrated quality, uncertainty, and expected value of additional
  compute; prove easy tasks halt earlier without harming hard-task accuracy.
  Accepted at CP340 at the mechanism, calibration, task-disjoint bounded
  workload, and causal-compute boundary. A portable logistic stop head consumes
  seventeen public bounded signals: fixed-point residual and contraction,
  calibrated update quality and uncertainty, evidence improvement, verifier
  score/change, action-policy uncertainty, measured gain/cost/net value,
  remaining budget, proposal admission, and evidence-availability flags.

  Training and calibration identities and tasks are disjoint. Each split has
  32-100,000 examples, at least eight unique tasks, four unique tasks per
  class, and eight examples per class. Learned mode requires held-out AUC >=
  0.75, balanced accuracy >= 0.70, Brier <= 0.25, ten-bin ECE <= 0.20, and
  false-stop rate <= 0.10. The artifact binds hashed train/calibration task
  sets, dataset hashes, feature schema, metrics, and threshold; atomic
  fsync-backed publication and stable no-follow regular-file loading require
  the configured SHA-256 and reject malformed, racing, symlinked, oversized,
  uncalibrated, or non-finite inputs.

  The live controller preserves divergence, budget, maximum-depth, fixed-depth,
  and residual-convergence invariants ahead of learned stopping. A learned stop
  can fire only when both update quality and value-of-compute evidence are
  measured. Its inference precision matches the signed public loop diagnostics,
  so the service can reload the pinned head and independently reconstruct every
  feature, probability, decision, branch stop, and aggregate verdict from the
  exact update-acceptance, loop-stability, and cognitive-action receipts.
  Runtime threshold overrides are refused.

  A task-disjoint easy/hard workload certificate requires eight unique held-out
  tasks, four per difficulty, at least one mean recurrent-step reduction on easy
  tasks, no overall or hard-task accuracy regression, and zero hard-task
  premature stops. A real tiny-Qwen equal-evidence causal test holds the
  calibrated update gate and measured negative value evidence constant, then
  proves the stop head reduces recurrent transitions and layer applications.
  This is a bounded mechanism/correctness proof, not a resident-32B utility or
  frontier result; a resident head and powered broad-task campaign remain open.
- [x] **SPARK-027 - Best-state and overthinking reversion.** Preserve the best
  verified state, detect oscillation/divergence, and prevent later recurrence
  from replacing a correct state without confidence-bound evidence.

  Best-state authority is now branch-local and confidence-bound. Bare scalar
  verifier output is explicitly uncalibrated and may rank candidates but may
  not preserve or restore hidden state. Authority requires either independently
  committed deterministic-exact evidence or a calibrated confidence interval
  with at least eight samples. The first authoritative observation promotes;
  later candidates replace the incumbent only when their lower bound exceeds
  the incumbent upper bound. Overlap or regression restores the exact
  incumbent state immediately without mutating peer branches.

  Finalization records the pre-state, returned state, source, fixed-depth mode,
  and whether reversion occurred. Adaptive execution prefers the verified
  state over the legacy scalar proxy; fixed-depth experiments preserve their
  scheduled endpoint. The top-level best-step identity follows the state
  actually returned. The receipt binds every branch-local observation and
  disposition to the exact cognitive-action trace and loop-stability receipt,
  whose oscillation, divergence, and containment evidence is already
  independently reconstructed by the service. State hashing and finalization
  are charged to the resource ledger.

  Unit and real tiny-Qwen execution tests prove scalar non-authority, minimum
  calibration power, interval-dominance promotion, overlap preservation,
  exact branch-local restoration, peer isolation, fixed-depth behavior,
  final overthinking reversion, receipt tamper rejection, service rejection,
  and preservation of bounded-verifier metadata through metering. This closes
  the mechanism and proof boundary only. The bounded verifier still depends on
  its caller's evidence commitment pending SPARK-071's external trust roots;
  no resident-32B broad-task utility or frontier claim is accepted here.
- [x] **SPARK-028 - Neural uncertainty head.** Train and calibrate a hidden-state
  correctness/entropy head at claim or step granularity; do not substitute
  self-reported confidence.

  A model-width-aware two-layer tanh head now learns directly from pooled
  admitted hidden states. Training examples require unique state and outcome
  receipts, independent verifier identity, objective boolean correctness
  labels, bounded finite vectors, class support, and at least four tasks per
  split. Training and calibration examples and task sets must be disjoint.
  The deterministic optimizer standardizes only from training data, trains all
  neural weights, temperature-calibrates on held-out logits, and selects a
  held-out threshold under a false-positive constraint.

  Admission requires calibration AUC >= 0.75, balanced accuracy >= 0.70, Brier
  <= 0.22, ten-bin ECE <= 0.15, and false-positive rate <= 0.25. Five empirical
  reliability bins carry independently reconstructable counts, rates, and
  Wilson bounds. A sparse prediction bin abstains. The artifact commits model
  width, hidden width, disjoint dataset/task identities, metrics, reliability,
  temperature, threshold, normalization, and both neural layers. Publication
  uses Aura's durable atomic writer; loading is bounded, no-follow, stable,
  exact-SHA-pinned, and refuses malformed, changed, uncalibrated, wrong-width,
  or rehashed artifacts.

  The live branch loop measures the exact admitted reasoning state after every
  update decision. It receipts the rounded pooled vector, vector and state
  commitments, pinned head, correctness probability, normalized predictive
  entropy, empirical interval, calibration support, and abstention. The service
  reloads the configured artifact and independently recomputes every estimate
  from the receipted head input, cross-bound to the update-acceptance
  transition. Work is charged to the structural resource ledger.

  The head has bounded causal authority rather than being decorative. When all
  branches have a supported latest estimate and no admitted external task
  verifier supersedes it, branch selection uses the highest predicted
  correctness; otherwise the existing task-verifier/convergence hierarchy is
  preserved. The receipt proves eligibility, all latest scores, selection
  basis, selected branch, and whether neural uncertainty changed authority.

  Objective-label, weak-model refusal, leakage, duplicate-evidence, lesion,
  persistence, symlink, byte-tamper, rehashed-metric, exact pooling, disabled
  non-invention, width-mismatch, source-tamper, service, and real tiny-Qwen
  two-branch tests close the bounded mechanism boundary. The resident 32B has no
  trained/pinned uncertainty head yet, and external outcome trust roots remain
  part of SPARK-071; no broad utility or frontier claim is accepted here.
- [x] **SPARK-029 - Mistake locator.** Train a span/transition locator on
  in-domain and out-of-domain subtle errors; evaluate location accuracy before
  using it to steer repair.

  A model-width-aware two-layer transition head now scores the complete
  prior-to-proposal change: pooled prior state, pooled proposal state, signed
  delta, and absolute delta. It therefore observes a bad proposal even when
  the update gate rejects it and preserves the exact prior state. Complete
  trace examples bind unique example/trace/task identity, controlled-mutation
  family, transition ordinal/count, objective error location or no-error
  label, trace receipt, and independent verifier identity.

  Training, in-domain calibration, and out-of-domain evaluation have disjoint
  example, trace, and task sets. In-domain evaluation uses exactly the training
  domain set on fresh tasks; OOD domains are disjoint. Every split requires
  complete traces, error and no-error support, multiple tasks, mutation
  families, and domains; every domain separately requires both positive and
  negative traces. Only training data determines normalization and neural
  weights. Temperature and decision threshold use the in-domain held-out split;
  OOD evidence is never used for fitting or threshold selection.

  Admission requires aggregate in-domain and OOD exact-location, error-only
  exact, within-one, no-error specificity, transition AUC, Brier, and ECE
  limits. Per-domain floors prevent an easy domain from hiding a failed one.
  The artifact commits all three dataset/task/domain identities, aggregate and
  per-domain evidence, model parameters, calibration, threshold, verdict, and
  an explicit `repair_steering_authorized=false`. Dataset and parameter memory
  are bounded; AUC uses rank statistics rather than a quadratic pair matrix.
  Durable atomic publication and stable bounded no-follow loading require the
  configured exact SHA-256 and refuse malformed, non-finite, changed,
  unadmitted, symlinked, or wrong-width artifacts.

  Every live recurrent proposal records prior/proposal/admitted commitments,
  acceptance disposition, rounded hidden vectors and commitments, pinned head,
  and reconstructed error probability. The receipt identifies the maximum
  above-threshold suspect per branch and the selected branch's candidate, but
  cannot steer state, repair, selection, attention, or decoding. The service
  reloads the configured artifact and independently reconstructs every score,
  source transition, candidate, aggregate, and non-authority boundary. Tensor
  reads and neural/host operations are charged to the resource ledger.

  Unit and real tiny-Qwen tests cover complete-trace/data-domain admission,
  objective localization, failed-OOD refusal, artifact persistence and
  tampering, symlink refusal, exact feature mapping, disabled non-invention,
  rejected-proposal visibility, live transition coverage, resource accounting,
  service reconstruction, and rehashed score/authority lies. This closes the
  bounded locator mechanism only. No resident-32B locator dataset/artifact or
  powered broad-domain campaign ran, and no repair authority is claimed before
  SPARK-030 through SPARK-032 establish the critic, contradiction, and bounded
  perturbation chain.
- [x] **SPARK-030 - Bidirectional hidden-state reflector.** Inspect the complete
  hidden trace with a non-causal critic that can compare premises and
  conclusions without reading only the final answer.

  Every recurrent branch now records a bounded hidden-state sketch for each
  prior, proposal, and admitted state. Numerically stabilized `asinh` block
  means and RMS values cover every model dimension while bounding each sketch
  to at most 128 scalars, including deliberately exploding finite activations.
  Each observation is bound to the update gate's prior/proposal/
  admitted state commitments and acceptance disposition, so a rejected
  proposal remains inspectable without entering the retained reasoning path.

  After recurrence ends, a read-only full-sequence critic revisits every step.
  For each transition it computes admitted prefix and suffix contexts, the
  initial hidden premise, the final admitted hidden conclusion, and a reflected
  state commitment spanning local, past, future, premise, and conclusion
  representations. It receipts local proposal/admission deltas and proposal
  cosine relationships to premise, conclusion, prefix, and suffix. Changing
  only a future conclusion changes an earlier reflection while preserving the
  earlier source observation, directly proving non-causal context access.

  The worker receipts complete transition coverage, past/future context counts,
  premise/conclusion commitments and comparison, selected-branch summary, and
  the exact update-acceptance source. The service reconstructs every sketch
  commitment, context, metric, reflected-state commitment, and aggregate.
  The critic consumes no decoded answer text and is structurally forbidden from
  mutating state, selecting a branch, steering repair, or perturbing attention.
  Capture and full-trace review work are charged separately to the resource
  ledger. It is active on every recurrent episode rather than waiting for a
  resident artifact or silently substituting final-answer prose.

  Direct, lesion, rejected-proposal, tamper, real tiny-Qwen, resource, and
  service tests prove full coverage, future-context dependence, retained-path
  separation, reconstruction, non-authority, and answer-text exclusion. This
  closes the complete-trace representation boundary only. It does not call the
  deterministic sketch a correctness model; calibrated contradiction evidence
  and any downstream authority remain SPARK-031 and SPARK-032 work.
- [x] **SPARK-031 - Contradiction tensor.** Produce calibrated token/step-level
  contradiction evidence, train on controlled mutations, and prove localization
  on middle-of-trace and long-context failures.

  The complete-trace reflector now preserves a second bounded representation
  at every recurrent latent sequence position. Each position sketch uses
  `asinh`-stabilized block means and RMS values in which every hidden dimension
  contributes. The receipt calls these latent workspace sequence positions,
  never decoded text-token indices. This closes the position-level evidence
  requirement without falsely claiming alignment to private chain-of-thought
  text or generated answer tokens.

  A pinned two-layer contradiction head consumes the exact shared
  training/runtime feature map for every transition-by-position cell. Its
  reconstructable channels cover local discontinuity, rejected-admission gap,
  premise, conclusion, prefix, suffix, and whole-trajectory conflict. Cell
  probabilities and per-step probabilities have independent temperature
  calibration; taking the maximum calibrated cell is not mislabeled as a
  calibrated step score. Admission includes cell and step AUC, Brier, and ECE,
  exact cell/step localization, within-one-step accuracy, no-error specificity,
  middle-of-trace accuracy, long-context error accuracy, and long-context sham
  specificity.

  Training requires complete transition-by-position tensors, split-disjoint
  task and trace identities, unique trace/mutation receipts, and a bound
  outcome-verifier identity. Train and in-domain calibration tasks are
  disjoint, out-of-domain tasks are disjoint again, and OOD domains cannot
  overlap the train/ID domains. Every split must
  contain controlled mutation families, independently committed sham traces,
  middle failures, long-context failures, and long-context no-error controls.
  Aggregate and per-domain floors decide admission; an unadmitted artifact
  cannot load.

  Learned mode is exact-SHA pinned, bounded, no-follow loaded, independently
  reconstructed by the service, and metered. A rehashed probability or
  authority lie is rejected. Unavailable mode emits no synthetic probability
  and avoids hidden feature-computation cost. The live tensor cannot mutate
  state, select a branch, repair a transition, or perturb attention. SPARK-032
  must earn any bounded causal influence separately.

  Controlled-mutation, task/trace/evidence separation, OOD, middle/long,
  calibration, artifact, future-lesion, tamper, unavailable-mode, real
  tiny-Qwen, resource, configuration, and service tests pass. This proves the
  mechanism and falsifiable admission path. No resident-32B artifact or broad
  outcome campaign ran, so live utility and frontier reasoning gains remain
  unproven.
- [x] **SPARK-032 - Attention/KV perturber.** Translate localized contradiction
  evidence into bounded, receipt-bearing changes to attention geometry or
  latent state at affected positions; include matched-random and no-op controls.

  An admitted contradiction coordinate can now propose one transaction on the
  selected branch's corresponding final latent workspace position. It does not
  rewrite the historical trace or call a latent position a decoded token. The
  guided arm moves only that writable position toward the branch's fixed
  post-prelude anchor, with target-slot delta RMS capped at eight percent by
  default and never above the configured 25-percent hard ceiling. Sealed
  cognitive-context positions are enumerated in the receipt and cannot be
  targeted.

  Every proposal runs three state-distinct arms: exact no-op, deterministic
  matched-random, and contradiction-guided. The random delta is orthogonal to
  the guided direction and matches its target-slot RMS. Other positions remain
  bit-identical. Each arm runs at least twice in independently shuffled,
  source-bound order. Probe decoding disables memoization and suppresses EOS
  to the fixed probe length, so actual transformer layer applications, not
  nominal limits, must match across every arm.

  The contradiction head never judges its own intervention. A verifier must
  first pass the existing concealed decoy preflight and mixed branch review,
  then emit independently committed deterministic-exact or calibrated-interval
  observations for every arm. Repeat observations must be identical. The
  guided arm is retained only when its worst lower bound exceeds the best
  upper bound of both controls by the configured margin. Scalar-only,
  unstable, unequal-compute, tied, regressing, failed, under-budget, or
  unverifiable attempts restore the exact baseline. Answer text is discarded;
  only probe hashes, bounded observations, state commitments, geometry,
  resource counts, decision, and rollback proof remain.

  Counterfactual mode is on by default but performs no perturbation without an
  admitted SPARK-031 artifact, localized writable coordinate, decoy-admitted
  authoritative verifier, and complete probe budget. The service reconstructs
  source, config, protected positions, arm geometry, observations, compute
  parity, decision, and authority. Unit tests cover retained, rollback,
  uncalibrated, unstable, unequal-compute, immutable-evidence, unavailable,
  evaluator-failure, configuration, and rehashed-authority paths. A seeded
  real tiny-Qwen episode executes the actual fixed-length transformer probes,
  meters candidate construction, and proves non-winning rollback.

  This closes the bounded intervention mechanism, not utility. No resident-32B
  artifact, broad outcome campaign, adapter/RLC interaction, reasoning gain,
  or frontier-capability result is claimed. SPARK-033 must separately earn
  localized exploration authority.
- [x] **SPARK-033 - Locally conditioned exploration.** Increase stochastic
  exploration only in uncertain/contradictory regions while preserving stable
  regions; meter entropy, diversity, regressions, and determinism controls.

  The default-live `counterfactual` mode derives its target only from an
  admitted SPARK-031 contradiction coordinate and its search radius only from
  that coordinate's calibrated probability multiplied by the latest supported
  SPARK-028 neural predictive entropy. Sealed cognitive-context positions are
  excluded. A second, low-contradiction writable position is selected from the
  complete tensor as a stable-region sham; it can be evaluated but never
  retained. If SPARK-032 already retained a mutation, exploration abstains
  because the uncertainty observation predates that changed state.

  Each source-bound deterministic direction produces three candidate families:
  exact duplicate no-op, equal-radius stable-position sham, and
  contradiction-position exploration. Directions are orthonormal before
  conversion to the actual latent dtype, their identities and geometry are
  receipt-bound, and generation is replayed exactly. Every non-baseline
  candidate changes exactly one position; every protected and non-target
  position remains bit-identical. Radius, candidate count, replicate count,
  direction entropy, output-hash entropy, unique conditioned outputs,
  regressing candidates, actual layer applications, repeated-observation
  determinism, and cross-label no-op determinism are independently
  reconstructable.

  All families receive the same fixed-length non-memoized transformer probes
  in counterbalanced order. Retention requires a decoy-admitted authoritative
  verifier, repeat-stable observations, deterministic no-op controls, equal
  actual compute, exact generator replay, at least the configured number of
  distinct conditioned outputs, and a conditioned lower bound above every
  no-op and stable-sham upper bound by the configured margin. Any malformed,
  unsupported, stale, low-entropy, protected, under-budget, collapsed,
  nondeterministic, unequal-compute, non-authoritative, failed, or non-winning
  path is compute-inert or restores the exact baseline. Probe text is
  discarded. The service reconstructs source conditioning, geometry, controls,
  evidence, decision, and authority from the serialized worker receipt.

  Validation passes 32/32 direct contracts including retention, exact rollback,
  source abstention, dtype-resolution collapse, evaluator failure, stale-source
  refusal, order leakage, output collapse, rehashed tamper, and configuration
  bounds. The focused exploration/perturber/tensor/wiring boundary passes
  158/158 in 84.16 seconds. The affected engine, reflector, update,
  uncertainty, locator, resource, verifier, stop, cache, worker-origin, and
  service boundary passes 325/325 in 171.60 seconds. The fixed 97-file RLC,
  recurrence/training, resident-campaign, global-workspace, GWT, and
  execution-controller ownership snapshot passes 1570/1570 in 809.59 seconds.
  Ruff, formatting, bytecode compilation, and diff hygiene pass. Governance
  remains at the pre-existing 49 regressions and 13 stale buckets, with no
  CP347 effect-ownership addition.

  SPARK-033 proves localized, source-conditioned search and its admission
  boundary. It does not prove resident-32B utility, adapter/RLC positive
  interaction, reasoning gain, or frontier capability; those still require
  fresh powered campaigns. SPARK-034 must combine old and new candidate
  distributions without granting an uncalibrated learned preference.
- [x] **SPARK-034 - Heterogeneous integration.** Combine old/new candidate
  distributions using calibrated evidence, not a free learned preference;
  compare selection, logit fusion, and abstention under equal compute.

  Accepted at CP348 at the bounded integration, final-decode, rollback, and
  independent-reconstruction boundary. Default-live counterfactual mode accepts
  exactly one retained source: SPARK-032's contradiction correction or
  SPARK-033's localized exploration. Local exploration already abstains after a
  retained contradiction mutation, and the integrator independently rejects
  zero, multiple, malformed, duplicate-label, stale, or lineage-inconsistent
  sources before policy compute.

  The prior source evidence supplies conservative old-upper and new-lower
  verifier bounds. Only separated bounds derive the fixed probability-fusion
  weight `new_lower / (new_lower + old_upper)`; no learned scalar or caller
  preference can choose the mixture. Incumbent selection, corrected selection,
  and per-token probability fusion run in counterbalanced repeats. Every policy
  executes both real transformer lanes from the same prompt-KV snapshot,
  bridge, candidate tensors, and token budget. Candidate distributions are
  normalized before mixing; raw logits and hidden states are never linearly
  interpolated.

  Probes must finish the exact fixed-length contract. Old/new lane applications
  must be equal, all policy totals must be equal, and the summed lane work must
  equal the episode budget's independently accumulated layer-app delta.
  Initial and complete lane/policy logits commitments, Jensen-Shannon
  divergence, authoritative verifier bounds, repeat determinism, state hashes,
  fusion weight, order, and compute are receipt-bound without retaining answer
  text. Fusion acts only when its worst lower bound beats both selection upper
  bounds by the configured margin. Corrected selection must separately beat
  incumbent selection. Every other evaluated path restores the exact incumbent.

  A winning fusion remains dual-lane for the complete user-visible
  autoregressive decode: each sampled token advances both isolated caches and
  the next-token probabilities are mixed by the admitted weight. Final receipts
  bind both latent tensors, weight, traces, lane work, divergence, termination,
  first logits, and output-token commitment. The service reconstructs policy
  authority and recomputes that commitment from the worker's returned raw token
  list, rejecting rehashed state, weight, lane, policy, or transport
  substitutions. Persistence, bridge, and decode timings are emitted at their
  actual execution boundaries.

  Validation passes 30/30 direct contracts and the focused
  integrator/engine/service boundary 162/162. The final affected predecessor,
  engine, verifier, resource, cache, worker-origin, and service gate passes
  307/307 in 115.48 seconds. The fixed 98-file RLC, recurrence/training,
  resident-campaign, global-workspace, GWT, and execution-controller ownership
  snapshot passes 1593/1593 in 576.71 seconds. Ruff, formatting, bytecode
  compilation, and diff hygiene pass. Governance remains at the pre-existing
  49 regressions and 13 stale buckets, with no CP348 effect-ownership addition.

  SPARK-034 proves calibrated distribution integration, exact rollback, and
  final-decode execution mechanics. It does not prove resident-32B utility,
  adapter/RLC positive interaction, reasoning gain, or frontier capability;
  those remain fresh powered campaign claims.
- [x] **SPARK-035 - KV state tree and rewind.** Snapshot at verified boundaries,
  remove rejected KV slices, restore exact prior state, and prove rejected
  reasoning cannot leak into regenerated branches.

  Accepted at CP349 at the live recurrent-window, verifier-probe, bytecode
  savepoint/backtrack, regeneration, final-persistence, dual-lane decode, and
  independent-service-reconstruction boundary. The bounded
  `KVStateTree` establishes one prompt-prefill root and links every logical
  branch boundary to a prior node. Branch savepoints carry the KV boundary
  identifier; verifier-promoted savepoints are distinguished from schedule
  savepoints; and backtrack restores the complete retained boundary before
  branch state resumes.

  Every non-persistent recurrent window and every persistent verifier probe
  opens a child transaction before transformer execution. The worker observes
  the mutated child offsets and immutable-storage commitment before removal,
  then restores the exact parent array objects and scalar metadata. A rejected
  child commitment is forbidden from becoming a later parent or accepted
  node. `regenerate_from_prefix` events are labeled in the same lineage, so a
  regenerated pass proves it started from the saved parent rather than an
  abandoned child. Standard final persistence and decode become accepted
  descendant nodes. CP348 probability fusion records both final isolated
  transformer lanes as terminal descendants; policy-evaluation lanes are
  explicitly discarded and cannot enter the canonical cache.

  The receipt contains no K/V tensors, decoded reasoning, or answer text. It
  carries salted process-local immutable-storage commitments, topology,
  offsets, authority, branch, parent/child/event hashes, disposition, and
  aggregate verdicts. This avoids copying a resident-32B prompt cache to CPU at
  every boundary. Exact identity restoration is enforced inside the
  source-verified worker; the parent service independently reconstructs the
  public hash graph and rejects missing, rehashed, orphaned, unpruned,
  final-less, or rejected-child-reuse claims. The service does not pretend its
  public receipt alone can inspect worker-private tensor storage.

  Validation passes 11/11 direct state-tree contracts, including capacity-backed
  cache compatibility, fail-closed lineage tampering, terminal-node
  requirements, and a real tiny-Qwen zero-tolerance experiment in which
  rejected work is pruned before regeneration and the regenerated hidden state
  exactly matches a clean control. The affected cache, recurrent, branch,
  engine, verifier, heterogeneous-integration, worker-origin, and service gate
  passes 326/326 in 120.14 seconds. The fixed 99-file RLC,
  recurrence/training, resident-campaign, global-workspace, GWT, and
  execution-controller ownership snapshot passes 1605/1605 in 622.29 seconds.
  Strict Ruff, formatting, bytecode compilation, and diff hygiene pass.
  Governance remains at the pre-existing 49 regressions and 13 stale buckets,
  with no CP349 effect-ownership addition.

  SPARK-035 proves bounded KV lineage, exact worker-side rewind, rejected-slice
  unreachability, and clean regeneration mechanics. It does not prove
  resident-32B utility, adapter/RLC positive interaction, reasoning gain, or
  frontier capability; those remain fresh powered campaign claims.
- [x] **SPARK-036 - Transient negative constraints.** Distill verified failures
  into scoped, expiring constraints, prevent unsupported critic prose from
  becoming a constraint, and prove repeated-error reduction.

  Accepted at CP350 at the live recurrent-action, worker receipt, and
  independent service-reconstruction boundary. A constraint is not text,
  critic advice, a caller-provided vector, or a durable weight change. It is
  the worker-private negative of one exact failed latent transition. Protected
  cognitive-context positions are zeroed, its RMS is capped against the
  current mutable state, and its public identity is a tensor commitment rather
  than hidden-state content.

  Authority starts only from a deterministic exact rejection or a calibrated
  confidence-interval regression produced by the independently admitted task
  verifier. The candidate must then beat repeated failed no-op and
  magnitude-matched orthogonal-sham controls under counterbalanced order,
  fixed token and transformer work, and equality across every measured
  resource counter, including verifier input and output bytes. Zero or
  incomplete metering cannot mint authority even when all arms report the same
  unsupported claim. Scalar, unsupported, unstable, unequal-work,
  non-repeating, non-winning, failed-evaluator, stale-KV, or under-budget paths
  cannot mint authority.

  Every admitted direction is bound to the exact episode objective, branch,
  source action, source KV boundary, and a short action-step TTL. Application
  is a reservation, not consumption: the runtime commits the one allowed use
  only after the recurrent transition succeeds. Budget refusal, cancellation,
  or a later-branch failure restores every branch, workspace, halting state,
  exchange/isolation state, telemetry object, trace append point, and KV
  boundary before authority becomes reusable.
  Success zeroizes the private direction before releasing its reference;
  expiry, stale lineage, and episode abort do the same and publish
  machine-checkable erasure evidence. The public engine boundary keeps an
  episode-local ledger registry and runs idempotent cleanup on successful,
  handled-failure, cancellation, and unexpected-exception exits.

  Deterministic and calibrated authoritative zero can no longer become a
  verified-best state. The engine binds the constraint to the state actually
  restored by verified-best arbitration, including prior-best restoration
  after confidence-bound regression, while retaining metered execution
  evidence. The action, verified-best, constraint, resource, information,
  preflight, and KV-tree receipts cross-bind the same step, branch, policy,
  observation, application, and lineage. The service reconstructs these
  relationships and rejects fully rehashed source, action, verifier, resource,
  scope, one-use, or KV substitutions.

  The direct contracts prove prose rejection, scalar refusal, exact and
  calibrated failure authority, repeated controls, allocated-resource parity,
  one-use scope, TTL, reservation rollback, abort/stale erasure, and
  cross-receipt tamper resistance. A real tiny-Qwen episode executes recurrent
  failure, exact parent restoration, a constrained retry, committed one-use
  recurrence, protected-slot invariance, and a verifier-confirmed reduction.
  A separate tiny-Qwen contract executes the actual MLX counterfactual
  evaluator over perturbed states and proves fixed decode work, fixed padded
  verifier input, nonzero complete metering, and exact branch/KV restoration.
  The synthetic label-aware reduction evaluator proves mechanism wiring, not
  broad reasoning utility.

  CP350 validation passes the focused constraint, verified-best, branch,
  value-of-computation, and service boundary 180/180 in 101.18 seconds and the
  affected engine, recurrence, resource, update, contradiction, exploration,
  heterogeneous-integration, telemetry, cache, output-quality, and
  proof-integrity boundary 229/229 in 128.31 seconds. The fixed 99-file RLC,
  recurrence/training, resident-campaign, global-workspace, GWT, and
  execution-controller ownership snapshot passes 1471/1471 in 823.69 seconds.
  After rebasing onto the concurrently published frozen-baseline, threat-model,
  typed-state, and literature checkpoints, their four new proof suites pass
  53/53 in 32.78 seconds on the integrated tree.
  Strict Ruff, formatting, bytecode compilation, and diff hygiene pass.
  Repository-wide governance reproduces the inherited 49 unrelated/concurrent
  regressions and 13 stale buckets; SPARK-036 adds no effect-ownership entry
  and does not absorb that debt.

  SPARK-036 proves bounded failure-avoidance mechanics and causal
  repeated-error reduction under controlled verification. It does not prove
  resident-32B utility, durable learning, broad error-rate reduction,
  adapter/RLC positive interaction, reasoning gain, or frontier capability;
  those remain fresh powered campaign claims.
- [x] **SPARK-037 - Causal virtual compute quanta.** Wire the existing bounded
  quanta contract into slot seeds, retrieval directions, verifier probes, or
  fast-weight subspaces; require measured contribution, TTL, budget charge,
  erasure proof, and ablation against no-quanta/matched-compute controls.

  Accepted at CP351 at the live recurrent-engine, worker receipt, and
  independent service-reconstruction boundary. The former free-form payload
  ledger and phantom compute reservation are gone. A quantum is now a private
  latent direction derived only from the episode's admitted prompt/context
  activations. It carries no text, caller vector, belief, durable parameter,
  or self-reported contribution authority.

  Admission runs no-op, norm-matched orthogonal-random, and guided arms with
  repeated seeded Latin rotations. Every trial executes the same forced-token
  transformer probe and independently admitted bounded verifier. All resource
  counters must be present and equal, including transformer layer apps,
  attention pairs, output-head tokens, verifier calls, and verifier bytes. The
  guided lower bound must clear both control upper bounds by the configured
  margin. Scalar/stateful verifiers are not consumed because they cannot mint
  confidence-bound authority.

  The accepted state is applied to one deterministic branch before the first
  recurrent savepoint. Immutable evidence slots are restored, mutable RMS is
  bounded, TTL is action-step bound, and use count is exactly one. Refusal,
  partial probe failure, application failure, and candidate degeneration
  restore the exact baseline without credit. Every path zeroizes and releases
  the private direction; pre-authority failures emit an explicit no-authority
  receipt rather than a malformed success-shaped record.

  The service reconstructs quantum identity, lifetime, Latin arm order,
  observation stability, matched resources, contribution, application,
  erasure, verifier policy/preflight, information accounting, episode totals,
  cognitive-slot provenance, and KV lineage. Rehashed scope, policy, order,
  identity, contribution, erasure, source, or resource substitutions fail.

  Direct contracts pass 14/14. The initialized tiny-Qwen proof captures the
  state at the real episode-initialization savepoint, proves it equals the
  verified quantum post-state, executes recurrence, validates the complete
  external receipt, and rejects rehashed authority. The affected engine,
  service, recurrence, resource, cache, verifier, fast-weight, exploration,
  and proof-integrity boundary passes 395/395. The fixed 100-file ownership
  snapshot passes 1653/1653 on the rebased integrated tree, and the separately landed frozen-baseline,
  typed-state, threat-model, and primary-literature suites pass 53/53.
  The focused CP351 plus eight concurrently landed suites pass 153/153 after
  integration; the final CP351 plus shared-state/runtime-hermeticity gate passes
  32/32 after the last two-commit rebase.

  SPARK-037 proves causal episode mechanics, truthful accounting, and bounded
  cleanup. It does not prove resident-32B utility, durable learning,
  adapter/RLC interaction, reasoning gain, or frontier capability. Those
  remain fresh powered campaign claims under RLC-VIRTUAL-QUANTA-001.
- [x] **SPARK-038 - Latent tree/forest search.** Implement bounded MCTS/beam/BFS
  over verified neural states with UCT/value estimates, backtracking, duplicate
  detection, cancellation, and exact compute accounting.

  Accepted at CP352 at the live recurrent-engine, public worker-receipt, and
  independent service-reconstruction boundary. The controller supports bounded
  UCT, beam, and breadth-first selection over private ensemble snapshots. Each
  node commits the complete branch-state/KV boundary set without serializing a
  latent tensor, hidden chain, or answer text. Parent restoration is exact
  before every expansion; duplicate state-plus-KV identities are pruned; search
  cancellation and no-authority outcomes restore the root exactly.

  Expansion is not a symbolic score-only tree. Each child applies a real
  cognitive operator, executes real recurrent transformer work, decodes a
  bounded probe, and receives an independently admitted interval observation.
  A candidate may replace the root only when its lower bound clears both the
  root and incumbent-authority upper bounds by the configured margin. UCT
  visits/value sums, beam/BFS parent choice, deterministic action scheduling,
  ancestry, winner authority, final restoration, and branch-local verified-best
  promotion are independently reconstructed.

  Every root and child probe carries a complete resource window. Cache hits
  require their key and saved-layer evidence and cannot claim transformer or
  output-head work. Failed expansions retain their consumed resource delta.
  The recurrent KV ledger inventories every call created by search, including
  work performed by an expansion that later throws. Winner ancestry identifies
  the committed calls; all other calls are explicitly discarded speculative
  compute. The full KV ledger remains public, while loop-stability excludes only
  that receipt-bound discarded set from the surviving fixed-point trajectory.
  Direct validation and the parent service bind the tree partition to the same
  KV-call ledger and reject a valid-but-wrong exclusion substitution.

  Direct search contracts pass 12/12. The real initialized tiny-Qwen proof
  forces repeated BRANCH decisions, commits verifier-authorized children,
  executes recurrence through the accepted snapshots, validates service
  reconstruction, and proves committed/discarded KV partitioning. The focused
  tree plus real-model proof passes 13/13, and the affected engine, service,
  worker-origin, KV-tree, value-of-computation, virtual-quanta, verified-best,
  and wiring boundary passes 213/213. After rebasing onto the concurrently
  published F6 pass-divergence work, the fixed 64-file RLC, recurrence,
  resident-campaign, global-workspace, GWT, and execution-controller ownership
  snapshot passes 1150/1150 in 799.62 seconds. Strict Ruff and diff hygiene
  pass.

  SPARK-038 proves bounded search mechanics and causal state selection. It does
  not prove resident-32B utility, durable learning, adapter/RLC positive
  interaction, reasoning gain, or frontier capability; those remain fresh
  powered campaign claims.

## E. Verifier mesh and local repair

- [x] **SPARK-039 - Atomic decomposition.** `atomic_decomposition.py` converts
  every bounded visible verifier candidate into text-free, content-addressed
  claim spans and typed support, derivation, condition, qualification, and
  reference transitions before holistic grading. It proves complete meaningful
  source coverage, bounded/nonoverlapping atoms, transition commitments,
  acyclic topology, objective-bound leading connectives, and explicit omission
  accounting. Missing dependencies or source spans deny grading authority;
  empty candidates remain neutral/unverified rather than earning a fabricated
  score. `EpisodeTaskVerifier` v3 runs this gate before arithmetic, code,
  facets, grounding, and response-contract checks and caps structurally invalid
  candidates below branch-selection authority.

  The worker reconstructs the full receipt against private candidate text. The
  service independently validates the text-free envelope and rejects forged
  atom hashes or authority. The decomposer is part of the critic source closure,
  so source changes invalidate critic identity instead of silently changing the
  judge. Focused atomic/verifier/verified-best/wiring coverage passes 141/141.
  This closes structural decomposition, not claim truth; SPARK-040 through
  SPARK-046 remain open for domain routing and independent semantic grading.
- [x] **SPARK-040 - Deterministic verifier router.** Atomic claims now route
  before scoring to exact integer arithmetic, Python AST compilation, or JSON
  parsing when a complete deterministic unit exists. Formal/SAT, retrieval,
  simulation, and planning claims return explicit `unsupported` with the exact
  missing authority; unclassified prose returns `unknown`, never a vacuous
  pass. Every route carries a content-bound pure-local read-only tool receipt.
  The service independently reconstructs route counts, atom bindings, tool
  commitments, and hard-pass authority. Chunked code and partial JSON abstain
  instead of false-refuting. Focused coverage passes 42/42 and the complete
  latent-cortex suite passes 823/823 in 78.42 seconds. This closes deterministic
  routing truthfully; governed external executors remain future eligible
  authorities rather than simulated successes.
- [x] **SPARK-041 - Process verifier.** The admitted recurrent mistake-locator
  head now carries held-out process-calibration cells for exact task domain and
  normalized early/middle/late recurrence depth. Each cell exposes sample and
  class support, transition accuracy, Wilson lower accuracy, Wilson upper false
  localization, AUC, Brier score, ECE, and a split-conformal maximum feature-
  distance envelope with its finite-sample alpha. Sparse, class-degenerate,
  unreliable, unknown-domain, legacy-artifact, missing-depth, and latent-
  distribution-shift cells abstain and emit no local credit.

  Every live proposal still receives a receipt-bound error probability. Dense
  process credit is `1 - p(error)` only inside an admitted cell; rejected
  proposals remain visible but cannot penalize the admitted path. A branch's
  process score is its weakest accepted transition rather than an average that
  could dilute one severe defect. The score becomes causal only when every
  branch has a fully calibrated accepted path. It then selects the maximum
  weakest-step branch ahead of convergence/uncertainty; deterministic visible-
  answer verification may still supersede it. The worker receipt binds domain,
  depth, feature distance, calibration evidence, abstentions, score, selected
  branch, and authority. The service reconstructs the artifact, transitions,
  domain, scores, winner, and causal basis; rehashed score or domain lies fail.

  Legacy v1 heads remain loadable for diagnostic localization but cannot earn
  process authority. Focused head/runtime/neural-uncertainty/service coverage
  passes 22/22. A real initialized tiny-Qwen episode loads a v2 learned head,
  runs recurrent transformer transitions, selects by the calibrated weakest-
  step score, and reports `process_verifier` through the independent uncertainty
  receipt. The complete latent-cortex suite passes 827/827 in 44.14 seconds.
  This proves calibrated local process mechanics and causal wiring, not
  resident-32B reasoning gain, adapter interaction, or frontier capability.
- [x] **SPARK-042 - Generative verifier.** CP356 adds a real fresh-context
  challenge lane after blind, decoy-calibrated branch selection. It gives the
  resident checkpoint only the original objective and one anonymized disputed
  atom; no sibling answer, branch identity, solver KV state, hidden workspace,
  prior rationale, or ownership cue crosses the callable boundary. Every model
  layer starts from a zero-offset cache. The receipt explicitly says that the
  verifier shares the resident checkpoint (`parameter_independence=false`) while
  proving context isolation through per-layer initial/final cache offsets,
  prompt/generated-token accounting, termination, and source commitments.

  The model must return one strict `FINAL_ANSWER` JSON object bound to the
  atom's full SHA-256. Generated text is retained only as a digest. It never
  becomes authority by assertion: the current admitted witness class is an
  exact integer-arithmetic relation whose operands/operator must match the
  disputed atom and whose independently derived result is recomputed by the
  deterministic router and again by the service envelope. Unsupported prose,
  malformed/truncated contracts, wrong claim bindings, non-integral division,
  mismatched witnesses, unknown domains, budget exhaustion, and any imported
  solver context all abstain. The service rejects recomputed forgeries as well
  as stale hashes.

  Authority is deliberately a refutation veto, not a second holistic score. A
  proven refutation removes the provisional winner and binds the replacement
  branch in the receipt; if there is no alternative, the episode refuses
  rather than emitting a known-refuted path. Positive model prose cannot boost
  a branch. The lane is active in both general and resident-32B service
  profiles; the resident contract receives 128 generated tokens so its
  64-character binding and witness are not truncated by a shallow probe budget.

  Focused protocol, engine, worker, service, and UI coverage passes 149/149. A
  real initialized tiny-Qwen run executes the fresh prefill/decode path and
  proves all model-layer caches start at zero; a causal engine test proves an
  independently re-derived `2 + 2 = 4` witness replaces a provisionally selected
  `2 + 2 = 5` branch. The complete latent-cortex ownership suite passes
  1,042/1,042 in 48.98 seconds. This proves fresh-context generative
  falsification mechanics and bounded causal wiring, not independent weights,
  general factual verification, resident-32B utility, adapter interaction,
  reasoning gain, or frontier capability.
- [x] **SPARK-043 - Adversarial verifier curriculum.** CP357 adds a bounded,
  reproducible curriculum that may teach the recurrent mistake locator only
  from independently verified failures. Each training unit binds a real,
  complete RLC reflector observation; closed-form recurrence task identity;
  model stack, layer schedule, configuration, seed, and source-manifest
  commitments; and two clean plus two mutated executions. Clean arms must pass
  the deterministic task oracle, mutant arms must fail it, repeats must be
  byte-equivalent, and controls must remain identical. The pair localizes the
  exact first divergent recurrent transition, requires an unchanged prior and
  a changed proposal, and rejects perturbations outside a bounded subtlety cap.
  Labels therefore come from the independent executable oracle, never from the
  model or mutator that produced the trace.

  The adaptive inserter chooses balanced task/mutation cells using prior
  locator misses and perturbation size, then receives feedback from every
  verified training candidate rather than only selected examples. Training,
  in-domain calibration, and OOD tasks are disjoint. OOD domains and mutation
  families are also disjoint from training, the complete calibration/OOD sets
  are frozen before fitting, and no held-out result can update head weights or
  inserter state. The learned two-layer head consumes the same bounded
  all-dimension reflector sketch that live recurrence emits. This creates v3
  artifacts while preserving v1/v2 diagnostic compatibility; the runtime
  dispatches each artifact to its declared representation and rejects width or
  schema drift.

  Held-out scoring runs in native macOS `sandbox-exec` with network and file
  writes denied. The child receives only the frozen head and held-out examples;
  the parent independently reconstructs all probabilities, predictions,
  per-domain/per-mutation metrics, Wilson bounds, dataset digest, and receipt.
  Verified training misses below the retention threshold enter an append-only,
  content-addressed negative store protected by an interprocess lock, atomic
  creation, crash recovery, semantic reconstruction, and Aura's hash-chained
  audit log. Held-out examples and pair/example lineage substitutions are
  rejected, and both record and chain tampering are detected.

  `tools/train_adversarial_verifier.py` is the operational bounded-input path:
  it validates a strict non-symlink 256 MiB bundle, trains, runs the native
  sandbox evaluation, persists the reloadable head and report atomically, and
  exits nonzero unless the head is admitted. A subprocess test exercises that
  complete path. Focused curriculum/artifact/runtime coverage passes 29/29;
  broader verifier curriculum coverage passes 138/138; and the complete
  latent-cortex ownership suite passes 904/904 in 48.34 seconds. Strict Ruff,
  bytecode compilation, diff hygiene, and the enterprise ratchet pass, with
  exact parent/current scans identical at 168 findings and 38 high/critical.
  This proves the adversarial curriculum, sandbox, retention, and live input
  representation mechanics. It does not prove a signed resident-32B head,
  resident-32B utility, adapter interaction, reasoning gain, or frontier
  capability; those claims still require an externally rooted powered
  campaign and cannot be inferred from synthetic admission.
- [x] **SPARK-044 - Counterfactual verifier.** CP358 adds a default-on,
  bounded counterfactual lane after admitted blind task verification and before
  the existing generative-refutation veto. It may act only when the
  six-decimal public task scores place at least two branches inside the
  configured `1e-6` top-score boundary. A stronger task-verifier score is never
  displaced, and a non-tie spends zero model-generation compute.

  The v1 protocol targets exact integer arithmetic claims. For each tied branch
  it extracts the same bounded number of visible atomic claims and applies the
  deterministic first-N sequence of left-input, right-input, and operator
  interventions. Every intervention is accepted only when its exact
  consequence differs from the candidate's visible result, eliminating the
  ambiguous case where an unchanged number is nevertheless correct by
  coincidence. Every tied branch must receive complete equal-size coverage;
  partial, unsupported, malformed, or truncated evidence abstains.

  Generation occurs in a fresh zero-offset KV context with no solver state or
  branch ownership identity. The shared resident checkpoint is disclosed, so
  the receipt claims context separation but never parameter independence.
  Output must be one strictly bound `FINAL_ANSWER` JSON object containing the
  claim commitment, intervention commitment, and one exact integer equality.
  Model output proposes a consequence; deterministic arithmetic reconstruction
  alone assigns `correct_change`, `invariant_failure`, or `incorrect_change`.

  The complete candidate text, objective, atomic decomposition, interventions,
  prompt commitments, fresh-context evidence, exact prediction, outcomes,
  task-score boundary, and implementation source digest land in the receipt.
  The service independently rebuilds all of them, reconstructs the unrounded
  blind-review source winner and six-decimal counterfactual boundary, proves
  the override reached final selection, and proves any later generative veto
  challenged the counterfactual-selected branch. A multiway tie receives
  authority only when the best counterfactual evidence is unique; branch index
  can never manufacture a winner.

  CP358 also closes two proof-contract defects surfaced by the new integration.
  Value-of-computation rewards are now calculated from the same eight-decimal
  public transition state that validators reconstruct, eliminating
  precision-boundary false failures. Neural-uncertainty selection provenance
  admits only the three runtime-producible ordered task-verifier override
  pipelines and rejects unknown, duplicated, reordered, or impossible
  modifiers. This repairs both the new counterfactual path and the older
  generative-refutation path without weakening neural selection authority.

  Focused engine, worker, service, counterfactual, and uncertainty coverage
  passes 163/163. The complete latent-cortex ownership suite passes 1,077/1,077
  in 35.10 seconds. Strict Ruff, bytecode compilation, diff hygiene,
  governance ownership, and the enterprise ratchet pass. Exact parent/current
  scans both contain 168 findings and 38 high/critical findings with identical
  semantic finding identities. This proves bounded arithmetic intervention
  mechanics, selection causality, and independent receipt reconstruction. It
  does not prove broad counterfactual competence, parameter-independent
  verification, a resident-32B gain, adapter interaction, or frontier
  capability.
- [x] **SPARK-045 - Prefix stability verifier.** CP360 adds a default-on,
  diagnostic-only recurrence measurement after the counterfactual tiebreak and
  generative-refutation veto have finalized the selected branch. The candidate
  is atomically decomposed and split before its first explicit conclusion, or
  before its final atom when no explicit conclusion connective exists. The
  deterministic verifier router must mark every prefix atom verified. Prose,
  retrieval, simulation, planning, unsupported code bundles, unknown claims,
  and single-atom answers therefore spend zero regeneration compute.

  An admitted prefix is regenerated three times by default. Each continuation
  starts from a newly allocated, all-zero-offset KV cache and receives a
  deterministic local MLX random key derived from the objective, candidate,
  sample index, and configured seed root. Sampling does not mutate or depend on
  the process-global RNG stream. The source conclusion is withheld. The fresh
  lane receives only the objective, exact verified prefix, and their content
  commitments, and must return one strictly bound `FINAL_ANSWER` JSON object.
  The receipt discloses that every lane shares the resident checkpoint, so
  context isolation is proved without claiming parameter independence.

  Conclusions are compared through an explicit signature hierarchy:
  canonical JSON, exact arithmetic-claim sequences, then a named normalized
  lexical surface fallback. Reference agreement, pairwise agreement, modal
  mass, normalized entropy, and signature counts are reconstructed from every
  complete sample. The public raw-stability value is the conservative minimum
  of reference, pairwise, and modal agreement. One failed, malformed, oversized,
  truncated, or context-invalid sample withholds the entire measurement rather
  than cherry-picking survivors. Contract-refused model text and fresh-context
  evidence remain committed for diagnosis instead of being erased.

  `core/learning/prefix_stability.py` calibrates this statistic only against a
  later independently receipted conclusion match. Its schema contains no
  correctness label. A standard-library pool-adjacent-violators fit is
  monotonic and linearithmic; fit and held-out calibration tasks and evidence
  identities must be disjoint, both classes and multiple tasks are required,
  and held-out AUC, Brier, ECE, and constant-baseline comparisons determine
  admission. Artifacts use stable no-follow reads, an exact SHA-256 pin, a
  governed atomic writer, and a strict content commitment. The bounded
  `tools/train_prefix_stability_calibrator.py` JSONL path rejects duplicate
  keys, binds report hashes to the exact bytes parsed, and exits nonzero for an
  inadmissible artifact. Until such an artifact is configured, runtime reports
  an explicit uncalibrated bootstrap signal with no probability.

  The service independently rebuilds the candidate boundary, deterministic
  prefix proof, prompt, sample seeds, strict outputs, signatures, metrics,
  calibrator result, and both non-authority claims. Recommitted seed, context,
  metric, conclusion, calibration, or authority substitutions fail closed.
  The engine uses local per-draw MLX keys and a real tiny-Qwen test proves
  same-seed reproducibility with zero initial cache offsets. CP360 also fixes a
  pre-existing strict-contract defect: the generic sentence-grace stop
  previously canceled the configured `FINAL_ANSWER` contract grace at the
  original token limit. Internal verifier generation now always requires the
  contract and reaches either contract completion or its bounded incomplete
  ceiling.

  Focused verifier, calibrator, real-engine, worker, service, generative,
  counterfactual, and response-contract coverage passes 230/230. The complete
  latent-cortex ownership suite passes 946/946. Strict focused Ruff, bytecode
  compilation, diff hygiene, and governance ownership pass on CP360's original
  `b984906f4` integration base; its governance inventory contains 1,956
  recognized calls in 1,830 buckets with the inherited 1,783 migration-debt
  calls unchanged. Exact parent/current enterprise scans on that tree are
  identical at 187 findings and 39 high/critical findings. Both retain the
  inherited `broad_exception_review` baseline regression, and the full closeout
  audit identifies the same parallel checkpoint's direct `os.cpu_count()` use
  as one resource-observation ownership violation. Rebase onto a newer `main`
  subsequently introduced three more parallel maturity checkpoints. CP361
  repairs the resulting combined-tree ownership and enterprise regressions;
  CP360 does not present its pre-rebase evidence as a measurement of that
  different tree.

  This proves bounded prefix-regeneration mechanics, honest conclusion
  recurrence measurement, task-disjoint calibration infrastructure, and
  independent receipt reconstruction. It does not prove that recurrence
  predicts correctness, broad semantic equivalence, resident-32B utility,
  adapter interaction, reasoning gain, or frontier capability.
- [x] **SPARK-046 - Correlation-aware verifier fusion.** Track Wilson/confidence
  bounds, domain reliability, dependence between verifiers, and historical
  calibration; no single probabilistic verifier has absolute authority.

  CP362 adds a default-on, service-supplied verifier mesh over five
  probabilistic evidence classes plus the separately tracked deterministic
  generative-refutation lane. An append-only governed ledger accepts only
  independently checked, task-committed outcomes whose individual signals bind
  the exact source receipt and whose correctness label binds a separate grading
  receipt. It produces content-addressed domain and global evidence snapshots
  with ten fixed calibration bins, Brier score, expected
  calibration error, directional accuracy, 95% Wilson intervals, and pairwise
  verifier error tables. Domain calibration is preferred; a global fallback is
  named as global rather than relabeled as domain evidence.

  The runtime normalizes the selected branch's decoy-admitted blind score,
  equal-score counterfactual robustness, conclusion recurrence, hidden-state
  correctness estimate, and accepted-transition process score. Absence of a
  generative refutation is explicitly not positive evidence, and a refutation
  of the provisional winner is not reassigned to its replacement. Calibration
  bins require eight checked outcomes and overall directional reliability
  requires twelve. Positive error correlation is shrunk toward zero only after
  twelve paired outcomes; missing paired history receives a conservative
  dependence upper bound of one and cannot pass the fusion gate.

  A fused measurement requires at least two calibrated sources, complete
  measured pairwise dependence, and 1.5 effective independent sources. Source
  weights are quality/sample/dependence adjusted and hard-capped at 0.5.
  Dependence widens the combined confidence interval. The receipt exposes
  support, opposition, inconclusiveness, or insufficient independent evidence,
  but grants neither branch-selection nor correctness authority and has no
  execution effect. The service independently reconstructs the complete
  evidence snapshot and fusion receipt; changed history, source signal,
  dependence, weight, interval, verdict, or authority field fails closed.

  Ten direct adversarial contracts cover bootstrap abstention, two-source
  admission, the single-source boundary, labeled global fallback, measured
  shared-error discounting, unknown-dependence collapse, receipt/evidence
  tampering, worker config validation, governed durable restore/duplicate
  refusal, and final-branch extraction. The focused mesh and service battery
  passes 106/106; the complete latent/RLC ownership battery passes 1,093/1,093.
  Ruff, compilation, diff hygiene, governance ownership, resource-observation
  ownership, and the enterprise static ratchet pass. The live ledger starts
  honestly in `bootstrap_unmeasured`; this checkpoint does not claim that any
  verifier predicts correctness, that fusion may replace an answer, resident
  32B gains, adapter interaction, broad reasoning gain, or frontier capability.
- [x] **SPARK-047 - Disagreement graph.** Localize the earliest dependency where
  branches diverge and identify the exact disputed assumption or transition.

  CP363 adds a machine-checkable, pairwise disagreement graph over the primary
  cognitive-action and cognitive-operator receipts. For every branch pair it
  reconstructs the ordered causal program, finds the longest exactly equal
  program prefix, and binds the first differing action, role, operator,
  transform, mutable-slot set, protected-slot set, and pre/post tensor
  commitments to their original operator-receipt hashes. Distinct hidden states
  running the same causal program do not become a disagreement.

  When complete decoded branch probes exist, the engine also source-reconstructs
  the existing atomic claim/dependency envelopes and binds each source hash to
  the corresponding independently blinded candidate commitment. The graph then
  finds the longest exact claim prefix and records the first disputed atom, its
  type, source span, content hash, dependency cues, and every transition touching
  it. A conditional atom or condition cue is reported as the exact disputed
  assumption; differing declared edges are reported as a dependency transition.
  No candidate text or hidden-state tensor enters the public receipt.

  Equality is intentionally strict. The graph never claims two paraphrases are
  semantically equivalent, never treats wording as correctness evidence, and
  explicitly records the worker-source/service-commitment validation boundary.
  It has no selection or repair effect; SPARK-048 must choose a diagnostic
  operation, and SPARK-049 must separately prove bounded invalidation and
  repair. The service independently reconstructs the graph from primary
  receipts and rejects changed localization, branch coverage, source binding,
  or authority fields.

  Six direct adversarial graph contracts and the updated engine/service
  integration pass 150/150 focused tests. The complete 61-file latent/RLC
  ownership battery passes 1,099/1,099. Ruff, compilation, diff hygiene,
  governance ownership, resource-observation ownership, and the enterprise
  ratchet pass. Governance remains exactly 1,968 recognized calls in 1,842
  buckets with 1,783 inherited migration-debt calls. Resource observation scans
  2,926 Python files with zero findings. The enterprise gate remains at 168
  findings, 38 high, zero critical, and no baseline regression. This proves
  exact structural and hash-bound decoded-claim localization; it does not prove
  semantic equivalence, candidate correctness, repair success, resident-32B
  gain, adapter interaction, broad reasoning gain, or frontier capability.
- [x] **SPARK-048 - Diagnostic action selection.** Choose the cheapest operation
  expected to resolve each disagreement: execute, retrieve, prove, simulate,
  falsify, regenerate from prefix, or ask a specialized verifier.

  CP364 adds a post-localization selector for every pairwise SPARK-047 dispute.
  It binds the exact disputed atom to the existing deterministic router before
  considering model-mediated diagnostics. Exact arithmetic, Python syntax, and
  JSON parse routes are recorded as already executed resolutions with their
  verified/refuted receipts. Formal, source, simulation, and planning routes
  identify the required verifier class without pretending that an unsupported
  route ran.

  The remaining candidates are source or memory retrieval, formal proof,
  simulation, falsification/counterexample, regeneration from the exact shared
  prefix, and a specialized assumption/transition verifier. Availability is
  reconstructed from the value controller's actual executor inventory and the
  stable memory, evidence, verifier, and savepoint flags in the primary action
  trace. Memory-only retrieval uses `search_memory`, not the unavailable
  evidence-retrieval executor. Latent `formalize` is explicitly not accepted as
  a theorem prover; proof remains unavailable until a real proof executor is
  wired.

  Each method carries a preregistered structural applicability reason. Once an
  action has eight checked historical trials, the selector replaces bootstrap
  assumptions with the measured verified-gain lower confidence bound and cost
  upper confidence bound from the same service-supplied value-of-computation
  snapshot. A mature nonpositive gain bound cannot win. Among the highest
  evidence-supported applicability band, the lowest conservative cost wins.
  Sparse choices remain labeled structural bootstrap priors. If no real,
  positive-evidence executor exists, the selector records
  `no_admissible_diagnostic_operation` instead of inventing one.

  The service independently reconstructs deterministic routes, capabilities,
  evidence/cost bindings, candidate scores, and the selected plan. The selector
  is recommendation-only: branch selection, execution, and repair effects are
  all `none`. SPARK-049 must execute the selected diagnostic under a bounded
  repair transaction. Eleven direct selector contracts plus updated graph,
  value-controller, engine, and service tests pass 168/168; the complete
  62-file latent/RLC ownership battery passes 1,110/1,110. The final rebased
  65-file combined ownership snapshot passes 1,180/1,180 after repairing the
  concurrent role-lesion runner's structural missing-telemetry contract. This
  checkpoint does not claim that an unexecuted recommendation resolved a
  dispute, repaired a branch, improved the resident 32B, or established
  frontier capability.
- [x] **SPARK-049 - Local invalidation and repair.** Preserve verified ancestors,
  invalidate the failed node and descendants, regenerate from the last valid
  state, and prove unrelated correct work is unchanged.

  CP367 adds a default-on, bounded repair transaction after the established
  verifier mesh and before later latent adaptation. A repair request exists
  only when a primary exact verifier has actually refuted an atomic claim.
  Merely running an exact verifier is no longer enough: two individually valid
  but different arithmetic claims do not fake a resolved disagreement.

  The transaction reconstructs the failed atom from its deterministic route,
  computes the complete directed descendant closure, binds every invalidated
  transition, names any exactly verified ancestor routes, and preserves the
  complete atom prefix before the failure. It also commits every atom outside
  the invalidation closure, including later independent claims. A fresh,
  zero-offset same-checkpoint generation receives the private original
  candidate, exact failure evidence, invalidation set, objective, and last
  valid prefix. The default policy permits one attempt and cannot spend the
  completion/fallback reserve.

  Admission requires the prefix to reconstruct exactly, every unrelated atom
  to retain its kind/content/dependency signature at the same ordinal, the
  originally failed verifier class to rerun and return `verified`, no other
  exact refutation to remain, and the replacement decomposition to be
  complete. A malformed, truncated, stale, still-refuted, overbroad, or
  budget-denied generation becomes an explicit non-repair. Original branch
  commitments remain byte-for-byte unchanged; an admitted result adds a
  separately committed epistemic candidate only. It has no latent-state,
  branch-selection, accepted-answer, or user-visible answer effect.

  The service independently reconstructs upstream graph/selector binding,
  failed-node closure, verified ancestors, preserved and unrelated atoms,
  generation context, exact replacement routes, counts, hashes, and authority.
  Fifteen direct repair contracts plus selector, engine, worker-config, and
  service-tamper coverage pass in a 167-test focused battery. The complete
  73-file latent/RLC ownership battery passes 1,246 tests on the combined tree.
  Ruff, compilation, diff hygiene, governance ownership, resource-observation
  ownership, model-load ownership, and the enterprise ratchet pass without a
  baseline increase. This proves bounded exact-refutation repair mechanics; it
  does not claim that non-exact recommendations resolved a dispute, that a
  repair replaced an accepted answer, resident-32B gain, adapter interaction,
  broad reasoning gain, or frontier capability.
- [x] **SPARK-050 - Confidence-bound answer replacement.** Replace an accepted
  answer only when the new lower confidence bound exceeds the old upper bound
  plus a preregistered margin; otherwise retain, qualify, or abstain.

  CP368 adds the default-on answer authority that CP367 deliberately withheld.
  It compares an admitted repair against the text produced by the actual final
  decoder, not a short branch probe or verifier surrogate. Promotion requires
  the repaired candidate's lower bound to exceed that final decode's upper
  bound by the configured 0.05 margin, plus a same-verifier-class transition
  from exact refutation to exact verification on the failed atom. A verified
  final decode is retained. A known exact refutation without a dominant repair
  abstains, including when the bounded repair-request budget omitted the
  selected branch. Disabled or unresolved cases retain without borrowing
  authority.

  The interval object is intentionally narrow and named in every receipt:
  conjunctive full-span exact-claim validity. Complete exact integer arithmetic
  can receive `[1, 1]`; deterministic arithmetic, Python-syntax, or JSON-syntax
  refutations receive `[0, 0]`; partial arithmetic, ordinary prose, successful
  Python/JSON parsing, and every unsupported claim remain `[0, 1]`. Syntax
  success is never mislabeled as semantic correctness. Objective relevance,
  requested-facet coverage, and complete final-answer structure remain owned
  by the parent service output-quality gate.

  The worker commits the ordinary neural decode as the final-output candidate
  before this policy runs. Replacement text must round-trip through the
  resident tokenizer and remain inside the original decode-plus-grace token
  envelope, capped at 1,024 tokens; failure abstains rather than silently
  retaining a refuted answer. Branch text, admitted repair text, original
  decode text, and original decode tokens cross only the internal
  worker-to-service IPC boundary. The service removes that private envelope
  before product return, independently reruns decomposition and deterministic
  routing, reconstructs every interval and decision, binds the original decode
  tokens (including heterogeneous decode), and verifies the accepted output
  text and token commitments.

  Sixteen direct answer-authority contracts cover dominance, conservative
  unknowns, selected/nonselected branches, explicit disable, output/margin/
  upstream/private-evidence tampering, tokenizer and output-envelope failure,
  mixed true-arithmetic/false-prose content, syntax-only verification,
  actual-final-decode comparison, omitted repair requests, rejected repair
  authority, and original-token tampering. Focused answer, engine, service,
  contract, and heterogeneous integration coverage passes 196 tests. The
  complete combined latent/RLC ownership battery passes 1,243 tests. Ruff,
  compilation, diff hygiene, governance ownership, resource-observation
  ownership, model-load ownership, and the enterprise ratchet pass without a
  baseline increase.

  This proves conservative exact-refutation answer replacement mechanics. It
  does not prove semantic correctness for unsupported claims, resident-32B
  gain, adapter interaction, broad reasoning gain, frontier capability, or
  long-duration live reliability.

## F. Adaptive compute and control

- [ ] **SPARK-051 - Value-of-computation controller.** Select among DECOMPOSE,
  BLIND_RESOLVE, BRANCH, SEARCH_MEMORY, RETRIEVE_EVIDENCE, EXECUTE, SIMULATE,
  FALSIFY, CHECK_ASSUMPTION, REGENERATE_FROM_PREFIX, FORMALIZE, COMPARE,
  BACKTRACK, COMPRESS_STATE, ANSWER, and ABSTAIN using measured expected gain per
  cost.
  CP326 makes the existing bounded execution-arm decision a state-admitted,
  costed, durable live operation and removes the decode-override bypass. This
  remains `PARTIAL`. CP327 adds the exact sixteen-action vocabulary and makes a
  deterministic value-of-computation decision before every recurrent window.
  Each decision binds the complete cognitive state signal, declared executor
  inventory, immutable evidence snapshot, gain/cost estimates, reason, and
  state transition. A hard neural floor prevents compare/stop actions from
  consuming an episode before recurrence. Measured cells use held-out lower
  gain and upper cost bounds; sparse cells are explicitly bootstrap exploration,
  never mislabeled as calibrated evidence. The selected action changes the live
  branch workspaces or halt state and is independently reconstructed by the
  service. Every action becomes an atomic child operation in the canonical
  journal without double-charging the enclosing episode. The worker cannot
  claim external EXECUTE authority, and unavailable actions fail closed.

  CP369 removes one remaining synonym failure. `SEARCH_MEMORY` and
  `RETRIEVE_EVIDENCE` no longer differ only by an embedded instruction passed
  through the same generic branch operator. Each selected action first reads
  only its matching immutable context class and writes a bounded summary into
  each live branch's communication slot. Episodic/durable memory, one-shot
  nonparametric memory, offline reference, world-model evidence, and governed
  tool/evidence sources have explicit source-class rules; the one-shot source
  truthfully participates as both memory provenance and typed evidence.

  Every focus transition commits its source slots and labels, input/source/
  output tensors, preserved-source tensor, branch/step identity, bounded
  strength, and exact tensor accounting. Source slots must remain unchanged,
  the output must differ from the input, and the following branch-specific
  cognitive operator must consume the focus output exactly. The service
  independently reconstructs source selection from the public cognitive-slot
  inventory, verifies branch/step coverage and focus-to-operator chaining, and
  matches every tensor-work total. The receipt explicitly says
  `external_retrieval_effect=none`; this slice cannot claim a fresh fetch.

  Seven direct focus contracts plus two forced-controller tiny-Qwen episodes
  cover source taxonomy, causal/preserved state, distinct memory/evidence
  outputs, absent-source refusal, inventory tampering, false external-fetch
  claims, one-shot hybrid classification, service reconstruction, and
  focus-to-operator lineage. Focused action/controller/branch/resource/service
  coverage passes 251 tests; the complete combined latent/RLC ownership
  battery passes 1,252 tests.

  CP370 adds one bounded service-side acquisition continuation for a validated
  SEARCH_MEMORY or RETRIEVE_EVIDENCE transition. The worker remains
  compute-only. The host binds the objective, query, selected transition,
  source inventory, one-attempt cap, and one-continuation cap; only a genuinely
  new typed memory or offline-reference observation can buy a second recurrent
  episode. Empty, unavailable, duplicate, and wrong-source outcomes remain
  distinct and cannot mint compute. The final receipt commits both episodes,
  the acquisition result, the returned round, and exhausted caps.

  CP372 adds the missing governed external EXECUTE protocol. The worker sees
  only a bounded digest-bound offer and can request, but cannot dispatch, an
  already-Will-admitted effect. The host reconstructs the action trace,
  readiness decision, action-policy evidence, successful epistemic operation,
  and exact authority before one durable coordinator advances PREPARED ->
  DECIDED -> DISPATCHING -> terminal. Dispatch and task leases prevent duplicate
  owners; a crash after dispatch becomes UNKNOWN_EFFECT and requires
  reconciliation rather than blind replay. Post-action receipts are
  deterministic, persisted, contract-equal to the staged recovery record, and
  required before success can be linked. Exact verified durable evidence can
  reconcile an unknown effect without re-executing it.

  The completed runtime-operation receipt now carries the admitted and final
  epistemic states, every action operation, measured compute, and a journal
  extension anchored to the authority's admitted head and entry count. The
  independent verifier reconstructs the final state and rejects fabricated
  authority, failed EXECUTE operations, missing or contradictory post-action
  evidence, secret-bearing replay containers, malformed abandonment markers,
  and self-consistent forged journal prefixes. An independent code review
  returned ACCEPT with no P0-P2 findings after the final anchor repair.

  CP373 implements the calibration-v2 evidence system without manufacturing a
  result. All sixteen actions must receive at least eight globally unique,
  multi-domain paired tasks (256 total treatment/control executions) before an
  acquisition certificate can exist. A cell remains explicitly unmeasured
  until it reaches 20 unique pairs (640 total executions for complete
  coverage). Treatment and matched no-action control begin from an identical
  externally captured, runner-signed checkpoint/KV/latent/evidence/memory/RNG
  state whose classifier and state-bucket evidence are also committed, run
  under fixed continuation and budget policies, and preserve complete
  available versus consumed information receipts.

  Claim-eligible tasks are reblinded with distinct 256-bit external-issuer
  nonces rather than the registry's reproducible seed-derived blind. A
  blind-independent task identity prevents reblinding duplicates from
  increasing effective sample size, and all actions receive the same committed
  stratum distribution under external randomized assignment. The externally
  rooted policy separates issuer, runner, contamination auditor, and evidence
  verifier custody. Every output is globally sealed before hidden answers can
  be revealed. The final candidate embeds the canonical append-only journal
  transcript; independent replay verifies every event hash, predecessor, state
  transition, attempt, pre-reveal seal prefix, manifest reference, final head,
  byte size, result, verification, and commit. It reconstructs every
  observation and statistic from the journal rather than trusting relabelable
  candidate fields. A signed final-verifier candidate/cell commitment is
  rechecked against the separately configured root inside the worker before
  measured evidence can affect action selection. Live admission checks current
  policy freshness, while immutable historical replay uses the committed
  admission time.

  Gain bounds use one exact simultaneous 34-family multiplicity budget. Cost
  uses a conservative Hoeffding upper bound over the maximum normalized
  fraction of every preregistered action-resource cap: structural FLOPs,
  transformer/attention/head work, tensor traffic and scalar operations,
  verifier work, tools, external models, and host operations. The independent
  CLI recomputes statistics and vector costs through a separate kernel. Legacy
  online transition moments remain bootstrap-only even after eight samples.

  SPARK-051 is not accepted yet. CP373 proves the bounded campaign,
  verification, admission, and fail-closed runtime machinery on generated
  fixtures; it does not supply the missing resident evidence. A real,
  externally custodied resident-32B campaign must still acquire at least 20
  unique pairs per action, publish its certificate, and survive live
  selection/lesion/restoration ablations before calibrated value or capability
  gain can be claimed. The negative frontier verdict is unchanged.

  CP374 closes the causal-intervention prerequisite that CP373's protocol
  assumed but the resident engine did not yet possess. A campaign cell now
  carries one strict intervention authority binding the separately configured
  current campaign policy and revision, complete canonical plan and exact cell,
  protocol, pair, task payload, execution ordinal, deterministic first-attempt
  identity, runner-attested starting state, exact normalized worker request,
  action, arm, and ordinal zero. A superseded embedded policy is rejected even
  while still cryptographically valid. The external campaign runner signs the
  payload under the separately configured Ed25519 root. Client, worker, engine,
  and request digest independently validate and bind the intervention. The lane
  is rejected on foreground/user requests.

  At the first recurrent action opportunity, the worker commits the actual
  complete state inventory and refuses a mismatch. Six components are directly
  worker-measured: latent slots; branch/ensemble roles, scores, halting,
  savepoints, evidence lineage, isolation/exchange state, traces, budget, and
  accounting; KV state; evidence; memory; and public action state. Durable host
  state and RNG root remain externally captured and runner-attested, with that
  ownership explicit in the receipt. Treatment reconstructibly emits exactly
  one `campaign_forced` decision even when the observational state policy would
  not select it; control consumes the matched opportunity without an action
  and remeasures the post-state rather than copying it. Both arms remove the
  target action from the remaining executor inventory.

  Before the worker may consume the request-bound attempt, it appends and
  fsyncs one `ACTION_INTERVENTION_CLAIMED` event to the canonical campaign
  journal. That event commits the intervention, normalized request, and signed
  journal head/count. A delayed authority for a superseded retry therefore
  fails against current journal state even if its embedded prefix remains
  valid. The exact claim lineage is then consumed before execution in a
  bounded, hash-chained, atomically replaced replay ledger under an
  interprocess lock and rechecked before the worker records execution. A crash
  after consumption fails that attempt closed. Claimed attempts survive
  restart without being auto-retried and cannot transition to `FAILED` while a
  worker may still execute; they must produce a sealed arm result or remain
  unresolved for operator reconciliation. Exact idempotent journal recovery,
  staged-result import, final transcript replay, and later
  result/verification/commit remain available.

  `EXECUTE` requires a real resident executor and governed offer; the parent
  validates its handoff, and the durable host coordinator preserves and
  revalidates the intervention through dispatch. Version-2 durable host
  records are digest-verified, narrowly migrated to version 3, validated, and
  atomically rewritten. A migrated terminal record replays; an interrupted
  dispatch becomes `UNKNOWN_EFFECT` and requires reconciliation rather than
  blind retry. The final receipt binds pre/post component maps, aggregate
  state and KV identities, decision, complete action trace, exact occurrence
  count, request/attempt identity, and plan authority. Parent-side replay
  rejects missing, altered, relabeled, rehashed, wrong-bucket/snapshot, reused,
  or state-mutating control evidence.
  Bare campaign transitions are rejected by the ordinary transition validator,
  so experimental manipulations cannot train the live observational policy.
  Intervention failures may not silently become a vanilla decode even when
  fallback is otherwise enabled.

  Seven dedicated contracts include an actual tiny-Qwen/MLX recurrent episode
  exercising the real complete-state serializer, signed-authority tampering,
  current-policy and durable replay refusal, canonical-journal retry
  supersession, an intentionally state-infeasible forced action,
  missing-executor refusal, treatment/control occurrence counts, remeasured
  control state after one explicitly omitted matched opportunity, worker
  omission, and fallback refusal.
  The durable-host suite separately proves campaign-forced `EXECUTE` reaches a
  persisted handoff and both safe v2 migration outcomes. The broader
  intervention, value-policy, request-identity, worker/client,
  execution-controller, epistemic-runtime, external-execution, preaction,
  verified-best, action-calibration, and frontier-task suite passes 456 tests
  after two unrelated pre-existing real-tiny-model failures are deselected.
  Both failures reproduce on a clean `origin/main`: transient
  constraint application remains zero after admission, and the verified
  latent-tree test receives an unexpected answer-replacement candidate. They
  remain open defects and must close before the resident campaign.

  CP374 does not create whole-host post-mutation capture or erasure receipts,
  execute the
  resident-32B campaign, calibrate a cell, demonstrate reasoning gain, or
  change the negative frontier verdict. The crash-resumable resident runner,
  external custody, at least 20 unique pairs per action, final independent
  certificate, live selection/lesion/restoration ablations, and broader
  frontier proof all remain required.

  CP376 removes the circular campaign-plan dependency and establishes the
  encrypted, one-use private snapshot substrate described in the whole-project
  tracker. CP377 closes the first mandatory review boundary on that substrate:
  a worker capture key is no longer its own trust root. Before process spawn,
  the parent creates a bounded random launch challenge under an ephemeral
  Ed25519 supervisor key. The child receives only that public challenge and
  signs its exact digest into a boot- and PID-scoped worker identity. After
  spawn, the parent verifies the real child PID and challenge, signs the worker
  identity, and publishes the complete challenge/identity/attestation chain.
  The parent private key never crosses the process boundary.

  Claim-grade state-capture request schema v2 requires that complete origin
  binding plus an independently supplied expected supervisor public key at
  request construction, current admission, historical replay, and public or
  private receipt verification. A bare legacy worker identity, a self-rooted
  rogue supervisor, a cross-worker substitution, a wrong PID, a rehashed inner
  mutation, or a stale live challenge fails closed. Expired launch evidence is
  rejected for current admission while its signatures remain historically
  replayable. Synthetic MLX resilience receipts now execute the same parent-
  challenge/child-signature path instead of receiving a production bypass.

  CP377 validation passes 73 focused worker-origin, state-capture, cancellation,
  and runtime-identity contracts and the broader MLX client, admission,
  resilience, ownership, memory, heartbeat, stability, and runtime matrix at
  337/337. Strict focused Ruff and diff hygiene pass. This closes only the
  self-rooted-origin P1. Partial resident application ambiguity, descriptor-
  rooted filesystem transactions, recoverable multi-file publication,
  external key custody, streaming serialization, kill/race/disk fault tests,
  and the real serializable resident continuation remain open. No training or
  capability result is created, and the negative frontier verdict is
  unchanged. Post-rebase governance integration also assigns the newly landed,
  fixed-path learned-world-model checkpoint to its narrow canonical owner; the
  exact inventory is 2,007 calls in 1,878 buckets with migration debt unchanged
  at 1,788 calls.

  CP378 closes the partial-application ambiguity in the snapshot state machine.
  Restore schema v2 durably distinguishes `prepared` from
  `application_started`; the second marker is atomically published and fsynced
  before the resident mutation callback begins, and binds the exact worker
  boot, PID, arm, request, snapshot, and operation. Only a provably pre-apply
  v2 marker may roll back. Any callback exception, post-apply hash mismatch,
  pre-commit storage failure, or process death after the started marker becomes
  typed `UNKNOWN_APPLICATION` evidence with same-process retry forbidden and
  process replacement required. An uncommitted v1 restore is also quarantined
  because the legacy schema cannot prove whether application started.

  The MLX latent-job boundary handles that typed error before its generic
  runtime handler, emits a correlated fatal quarantine receipt, and exits the
  worker after delivery so it cannot serve another request. A spawned-child
  fault test enters the real apply callback, receives `SIGKILL`, and proves the
  surviving process reconstructs the durable quarantine and refuses retry.
  Focused storage/worker coverage passes 52 tests; the broader snapshot,
  identity, runtime, worker-origin, admission, and MLX resilience boundary
  passes 248/248. This closes the second CP376 P1, not the remaining descriptor,
  publication, custody, streaming, broader fault, or continuation work. No
  training or capability verdict changes.

  CP379 closes the ancestor/path-reentry P1. The private snapshot store captures
  the root inode once, retains descriptors for every fixed namespace and the
  interprocess lock, and performs nested traversal, creation, read, atomic
  replacement, unlink, directory removal, listing, and fsync relative to those
  descriptors with no-follow semantics. Every operation verifies that the
  public root name, namespace entries, and lock name still resolve to the held
  inodes before and after mutation. Root, namespace, and lock substitution fail
  before private I/O and leave the replacement tree untouched.

  A real two-process race against one arm proves the descriptor-held `flock`
  serializes independent store instances: one process commits the restore and
  the other receives `private_snapshot_arm_already_used`; no second application
  occurs. Existing kill, recovery, tamper, and lifecycle coverage remains green.
  Focused action-store coverage passes 32 tests and the broader worker/client/
  runtime boundary passes 252/252. CP379 does not claim multi-file publication
  atomicity, external key custody, streaming serialization, a resident
  continuation, training, or reasoning gain.

  To prevent SPARK-051 from expanding indefinitely, its remaining pre-training
  implementation is a fixed five-checkpoint burn-down: recoverable publication
  transaction; external key custody plus streaming codec; serializable resident
  continuation and first-action hook; worker/client/service/runner/verifier
  lineage; then the resident-1.5B destructive integration gate. Defects found by
  those gates are repaired in place, but unrelated capability expansion does not
  enter SPARK-051. The resident-32B training and preregistered reasoning campaign
  follow that gate.

  CP380 closes recoverable snapshot publication. A capture is now assembled in
  a hidden transaction directory containing its encrypted chunks, envelope,
  key, authenticated handle, use ledger, and authenticated private publication
  record. Every file and newly created directory is durably synced before one
  descriptor-relative directory rename makes the complete bundle visible. Both
  parent directories are synced after the rename. No reader can observe the
  former partial key/chunk/envelope/ledger/handle sequence.

  Retry under the interprocess lock searches authenticated complete bundles by
  request and component commitments. A crash after commit therefore returns the
  original opaque handle instead of producing a duplicate. A crash before
  commit leaves only a hidden transaction, which the next lock holder validates
  and removes before retry. Publication I/O failures roll back the whole staging
  tree and retain their original `OSError` as causal evidence under a stable
  snapshot-publication error.

  Four new destructive contracts prove the transaction boundary: a spawned
  publisher is killed with `SIGKILL` before commit and leaves zero visible
  bundles; an injected `ENOSPC` leaves neither bundle nor transaction; a forced
  post-commit death recovers the exact committed handle; and two independent
  simultaneous publishers converge on one handle, one snapshot digest, and one
  bundle. Focused storage coverage passes 36/36 and the broader snapshot,
  identity, worker-origin, MLX runtime, admission, cancellation, and resilience
  boundary passes 256/256. Strict Ruff, bytecode compilation, enterprise gate,
  exact effect-ownership ratchet, and diff hygiene pass.

  CP380 closes only the publication-transaction checkpoint. External key
  custody plus streaming, the serializable resident continuation and first-
  action hook, end-to-end lineage, and the resident-1.5B destructive gate remain
  the four fixed pre-training checkpoints. No resident training, reasoning gain,
  frontier gain, or `WOW Signal` is claimed.

  CP381 closes external snapshot-key custody and bounded streaming. The snapshot
  store now requires an explicit custodian and has no plaintext-key or
  environment fallback. Its production provider wraps each random DEK with
  AES-256-GCM under a wrapping key held by macOS Keychain. The authenticated
  wrapped-key envelope binds the custodian identity, request, and opaque handle;
  only that envelope enters the snapshot transaction. Ordinary system opens are
  read-only and fail if custody was not provisioned. One host-only provisioning
  method creates and confirms the Keychain item before resident workers spawn,
  avoiding concurrent implicit key creation.

  Publication now canonicalizes JSON incrementally, slices even one large scalar
  into bounded pieces, coalesces exact-size chunks, hashes and encrypts each
  chunk, and writes it directly into the hidden transaction. It no longer holds
  complete plaintext components plus every encrypted copy. Restore authenticates
  one chunk at a time into one component buffer, verifies its streaming digest,
  reconstructs the concrete state value, and zeroes the transient plaintext and
  DEK buffers. Existing-bundle retry must prove the current custodian can unwrap
  the committed DEK before it may return the original handle.

  The focused custody, streaming, snapshot, and strict-Keychain boundary passes
  57/57. The broader snapshot, identity, worker-origin, MLX runtime, admission,
  cancellation, resilience, and secret-backend boundary passes 277/277. Tests
  prove bounded multi-chunk binary and single-scalar JSON round trips, absence of
  the raw DEK from every bundle file, wrong-custodian refusal, tamper/context
  rejection, explicit provisioning, post-close refusal, and unchanged crash/
  race/erasure semantics. A real native-Keychain probe provisioned once and
  reopened read-only with identical public custody and wrapping-key identities.
  Ruff, bytecode compilation, enterprise gate, exact governance ratchet, and
  diff hygiene pass.

  CP381 closes only custody and streaming. Serializable resident continuation
  plus the first-action hook, end-to-end worker/client/service/runner/verifier
  lineage, and the resident-1.5B destructive integration gate remain the three
  fixed pre-training checkpoints. No resident training or reasoning/frontier
  gain is claimed.

  CP382 closes the serializable resident-continuation and engine-owned first-
  action checkpoint. `PortableStateComponent` is a strict, deterministic,
  bounded binary format for nested scalar state, bytes, mappings, sets,
  NumPy tensors, and MLX tensors. It streams directly into the authenticated
  snapshot store, rejects malformed/noncanonical payloads, and preserves MLX
  `bfloat16` through an exact uint16 bit view instead of an invalid NumPy
  conversion. No pickle or executable deserializer enters the boundary.

  The engine now owns the capture point immediately before opportunity one can
  choose a cognitive action. It records complete branch runtime, latent slots,
  exact recurrent KV snapshots, evidence, memory, public action state, and the
  runner-supplied durable/RNG roots. Capture-only execution exits before policy
  selection and decode. Restore installs those tensors and cache snapshots into
  an equivalent fresh frame, rebinds per-episode KV capabilities to the fresh
  worker's root, and re-encodes all eight components before action. Any mismatch
  rolls the resident back and verifies the rollback; ambiguous rollback raises
  the existing fatal `UNKNOWN_APPLICATION` quarantine.

  Real tiny-Qwen2 tests prove zero action/decode leakage, exact eight-component
  capture/restore identity, runner-state drift rejection, post-failure worker
  usability, and a complete engine -> streamed encrypted store -> authenticated
  restore round trip using actual latent and KV tensors. The post-rebase affected
  engine/store/intervention/worker/client matrix passes 334 tests. Strict Ruff,
  bytecode compilation, and diff hygiene pass. The candidate has no enterprise
  or governance delta from clean `origin/main`; both share two pre-existing
  enterprise ratchet regressions and the same three-new/one-stale governance
  drift introduced by cooperative semantic-closeout commits. No baseline is
  raised here.

  CP382 does not claim end-to-end transport or a resident result. Worker/client/
  service/runner/verifier lineage and the resident-1.5B destructive integration
  gate remain the two fixed pre-training checkpoints. No resident-32B training,
  reasoning gain, frontier gain, or `WOW Signal` is claimed.

  CP383 closes the public resident action-state transport and independent
  verification lineage. The runner-facing frame contains only the signed
  capture request, current policy, exact model/execution/request identities,
  current parent-attested worker binding, and, for restore, the public capture
  receipt and arm. Runner durable and RNG roots are reconstructed from the
  signed payload inside both runner and worker code and checked against the
  request's portable-state commitments. No opaque snapshot handle, continuation
  value, raw runner state, DEK, or wrapped-key material crosses ordinary MLX
  IPC.

  The client provisions external Keychain custody only when this claim-only lab
  lane is requested, replaces the proposed resident binding with the real live
  worker binding, independently admits the frame, and verifies every returned
  public receipt. The worker captures before action/decode, retrieves private
  state by authenticated request after process replacement, installs and
  re-encodes all eight continuation components, and invokes a first-action hook
  only after the exact aggregate state hash matches. Treatment and matched
  no-action are consumed once each; the second committed arm seals the pair,
  cryptographically destroys its key, deletes ciphertext, and returns signed
  terminal lifecycle evidence. Unknown application quarantines and replaces
  the MLX worker.

  Capture and restore trust are intentionally separate. Historical request and
  capture signatures remain rooted in the original supervisor, while each
  replacement resident worker receives a fresh parent challenge and
  attestation under the current supervisor. A review-discovered key-confusion
  bug was repaired before publication, and the challenge-expiry regression now
  proves historical capture verification under the old root plus restore
  verification under the new root. The independent pair verifier accepts
  distinct worker/supervisor keys by arm and reconstructs capture, intervention,
  restore, once-only custody, seal, and erasure lineage.

  Focused action-state, worker/client/service, and authority coverage passes
  186/186. The complete affected continuation, custody, intervention, engine,
  runtime identity, worker origin, MLX admission/cancellation/resilience,
  secrets, and contract matrix passes 480/480. Ruff, bytecode compilation, and
  diff hygiene pass. CP383 adds no enterprise or governance debt; the two
  inherited enterprise regressions and unrelated three-new/one-stale governance
  drift remain explicit and no baseline is raised.

  CP383 closes the lineage checkpoint, not the resident result. The sole fixed
  pre-training checkpoint is now the resident-1.5B destructive integration gate
  across capture, process replacement, both arms, duplicate refusal, terminal
  erasure, and independent verification. Resident-32B training and the
  preregistered reasoning/frontier campaign follow. No gain or `WOW Signal` is
  claimed.

  CP384 completes that fixed pre-training gate. Capture admission and worker
  execution now independently derive the model-weights identity from the loaded
  resident worker, binding full checkpoint content, parameter counts, adapter
  stack, tokenizer artifacts, quantization, and identity gaps. A real
  Qwen2.5-1.5B-Instruct-4bit continuation is captured before action or decode,
  streamed through encrypted external custody, restored exactly into both arms,
  sealed after one use each, and cryptographically erased. A second real-model
  execution restores one state into runner-signed treatment and control:
  treatment performs exactly one `BLIND_RESOLVE`, control performs none,
  duplicate consumption fails, and permanent parameters remain identical.

  The evidence boundary is explicit: CP383 proves spawned-process transport,
  replacement, trust rotation, quarantine, and independent verification;
  CP384 executes the corresponding state and intervention contracts on real
  model weights and MLX tensors in fresh engine frames. It does not relabel that
  composition as one all-layers spawned IPC test. The opt-in resident gate
  passes 2/2; after final identity hardening, the combined state/intervention/
  wiring suite passes 159/159 with those gates enabled and 157/157 with two
  explicit deselections in ordinary runs. The registered `resident_model`
  category does not increase skip debt. CP384 adds no enterprise or governance
  drift and raises no baseline.

  The five-checkpoint pre-training burn-down is now closed. SPARK-051 itself
  remains open for resident-32B training, the preregistered equal-compute
  reasoning/frontier campaign, independent verification, and live selection,
  lesion, and restoration evidence. No gain or `WOW Signal` is claimed. This is
  total checkpoint record 626; the 640-920 forecast leaves approximately
  14-294 records, or 68.0%-97.8% checkpoint-count completion (80.3% midpoint).

  CP385 attempted the first fresh resident-32B answer-channel preflight after
  the mechanical gate. The detached process was contained and terminal, but
  failed before model load because independently generated finite-domain train
  and holdout batteries shared a prompt. That failure is retained as evidence.
  The curriculum now seeds explicit train/holdout partitions and constructs
  holdout under an exclusion set containing every committed training prompt
  and task identity. The exact failing preflight dimensions and seed are a
  regression test. The focused curriculum, trainer, and preregistration matrix
  passes 44/44. This is total checkpoint record 627; the 640-920 forecast
  leaves approximately 13-293 records, or 68.2%-98.0% checkpoint-count
  completion (80.4% midpoint). A new source-bound campaign contract and
  detached preflight must still run before any training or gain claim.

  CP386 loaded the actual resident 32B and exercised the repaired split. Its
  recurrent baseline scored 1/6. Calibration completed five probes before its
  bounded time cap: 4/20 completions were parseable and correct, for a 0.20
  answer-channel fraction. The calibration admission correctly refused
  training as `answer_channel_blocked`; zero optimizer groups or updates ran.
  The terminal reporter then raised `KeyError: diagnosis` because zero-group
  telemetry intentionally had only `reason`. That failed receipt is retained.
  `_signal_admission_report` is now total over zero groups and emits an
  explicit diagnosis and required next gate; its focused trainer/curriculum/
  preregistration matrix passes 45/45. No adapter, gain, or frontier claim is
  available. This is total checkpoint record 628; the 640-920 forecast leaves
  approximately 12-292 records, or 68.3%-98.1% checkpoint-count completion
  (80.5% midpoint).
- [x] **SPARK-052 - Adaptive breadth/depth/tool routing.** Scale recurrence,
  branch count, lookahead, tools, and verifier effort from difficulty,
  uncertainty, stakes, body pressure, deadlines, and resource admission while
  preserving user-facing work.

  CP387 closes the live adaptive-routing contract. One deterministic policy
  now combines prompt-structure difficulty, uncertainty, stakes, allostatic
  body pressure, caller deadline, resident model scale, and the canonical
  runtime-admission pressure snapshot. It causally sets bounded recurrence,
  virtual width, latent-tree nodes/depth/branching, generative/counterfactual/
  prefix verifier effort, and the zero-or-one host acquisition continuation.
  Critical body/resource pressure and sub-45-second deadlines force the lean
  tier. Unknown resource observation is conservatively pressured rather than
  treated as free headroom.

  The plan is versioned, strict, and content-addressed. It is applied before
  worker IPC, re-enforced after learned execution-arm selection, and checked
  against actual worker steps, branches, verifier profile, decode surface, and
  wall budget. Independent execution and acquisition validators reject nested
  or outer-envelope tampering. Explicit experiment structural overrides remain
  identifiable and do not receive a false adaptive-execution certificate.
  Retrieved evidence still passes through the existing governed ingress and
  instruction-free context contracts; this checkpoint does not widen external
  execution authority.

  The affected adaptive policy, service, acquisition, latent-tree, runtime
  control-plane, ontogeny, and surface matrix passes 284/284. Strict Ruff,
  bytecode compilation, and diff hygiene pass. The enterprise ratchet has zero
  finding in touched files and retains only its inherited repository excesses.
  This is total checkpoint record 629; the 640-920 forecast leaves
  approximately 11-291 records, or 68.4%-98.3% checkpoint-count completion
  (80.6% midpoint). This is runtime allocation evidence, not reasoning gain.
- [x] **SPARK-053 - Principled stop and abstain.** Stop on verified convergence,
  low value of further compute, budget exhaustion, or irreducible uncertainty;
  distinguish each reason in receipts and language generated by Aura herself.

  **CP389 implementation and evidence:** A strict terminal-disposition policy
  now sits above the previously separate residual halter, learned stop head,
  value-of-computation terminal actions, loop-stability proof, and resource
  budget. It reconstructs one precedence-ordered reason from committed public
  evidence: irreducible uncertainty, wall/compute/recurrent budget exhaustion,
  measured non-positive value of further compute, verified fixed-point
  convergence, verified answer/action readiness, stability containment,
  interruption, or planned-depth completion. A `converged` label without a
  finite, anchor-bounded selected transition marked as a fixed-point candidate
  is rejected instead of promoted.

  The disposition selects answer, bounded-answer, abstain, execute, or defer
  semantics. It does not select canned prose. A reason-specific epistemic
  instruction is tokenized with the resident tokenizer and appended only at
  final synthesis, after verification and temporary adaptation. The largest
  possible instruction is reserved before recurrence, so honesty language
  cannot unexpectedly consume the protected answer budget. The resident model
  still generates every output token in its own words.

  The receipt commits the low-level halt, loop evidence, action trace, exact
  decision-time budget, precedence, reason, disposition, instruction text and
  tokens, complete final bridge token sequence, model-generated output tokens
  and text, and whether a confidence-bound resident repair produced the final
  answer. Independent service validation reconstructs the decision, proves
  monotonic budget lineage, proves the instruction is the exact suffix of the
  applied bridge, binds final output identity, and rejects substrate-only or
  unapplied language on the live path. Public token sequences expose only the
  fixed epistemic instruction, never private latent state or chain-of-thought.

  Tests cover every required reason and precedence, false convergence,
  recomputed-hash tampering, detached instruction tokens, unapplied resident
  instructions, substrate honesty, full tiny-Qwen execution, learned stopping,
  service IPC, GWT coupling, and value-of-computation terminals. The affected
  matrix passes 171/171; strict Ruff, bytecode compilation, and diff hygiene
  pass. The enterprise ratchet reports no finding in touched files and retains
  only inherited repository excesses (`placeholder_stub_mock` 15 > 13 and
  high/critical 40 > 39). This closes principled stop semantics and language
  causality; it does not claim a reasoning or frontier-performance gain.
- [x] **SPARK-054 - Complete causal receipts.** Record state lineage, operators,
  branch isolation, tool evidence, verifier scores, accepted/rejected updates,
  compute, adaptations, stopping, final synthesis, and integrity proofs without
  exposing private chain-of-thought.

  **CP390 implementation and evidence:** Every latent episode now carries one
  ordered, hash-linked public causal envelope over twelve fixed stages:
  ingress identity, recurrent state lineage, cognitive operators, branch
  isolation/exchange, optional tool-memory evidence, verification,
  accepted/rejected updates, compute, temporary/durable adaptation, stopping,
  final synthesis, and runtime/model integrity. Each node commits only field
  names, presence, shape/count, canonical value hashes, and the preceding node
  hash. Raw latent values, private reasoning, and tool-secret values are never
  copied into the envelope.

  The envelope is reconstructed at every identity boundary instead of being
  patched: the engine emits an honest partial receipt, the worker rebinds it
  after request and resident-worker identity exist, and the client rebuilds
  the final form only after capturing source/app/runtime provenance. The live
  service independently reconstructs the complete DAG and rejects missing
  stages, anonymous episodes, source drift, reordered or recomputed nodes,
  mutated privacy declarations, detached final-output identity, and unproven
  parameter or applicable fast-weight integrity. Action-capture and failed
  episodes retain honest partial envelopes rather than inventing completion.

  Dedicated tests prove exact ordering and hash lineage, optional-evidence
  semantics, source and self-consistent envelope tampering resistance, strict
  integrity completion, and absence of sentinel private reasoning/tool values.
  Tiny-Qwen engine, worker, client, service, terminal, stop-policy,
  value-of-computation, runtime-identity, and integrity matrices pass 277/277.
  Strict Ruff, bytecode compilation, and diff hygiene pass. The enterprise
  ratchet has no new finding attributable to CP390 and remains at its inherited
  repository excesses (`placeholder_stub_mock` 15 > 13 and high/critical 40 >
  39). This proves receipt completeness and provenance on the implemented
  path; it is not a reasoning-gain or frontier-performance claim.

## G. Temporary and permanent learning

- [x] **SPARK-055 - Query-scoped fast-weight learning.** Optimize bounded
  temporary weights from high-confidence evidence, prove identity at attach,
  constrain magnitude/behavior, isolate concurrent requests, and make the
  adapted function causal to the answer.

  **CP391 implementation and evidence:** The temporary-learning path now starts
  from a deterministic same-query base-function probe. Only machine-checked
  atomic evidence routes are eligible for the private bounded target; refuted,
  unsupported, unknown, oversized, unavailable, or unverifiable evidence
  cannot attach a wrapper or spend an optimizer step. Public receipts commit
  exact source/objective/evidence/target identities without publishing evidence
  text or latent state.

  Every mutable model object has an exclusive process-local query lease with
  owner/model commitments and conflict accounting. Attachment must pass an
  immediate full-stack pre/post identity probe before optimization. The
  optimizer is bounded and receipted, capability and structural canaries can
  rescale or erase it, and a matched uncached deterministic post-probe must
  change the output token sequence and strictly improve the same verifier on
  the unchanged winner state. Equality, regression, no accepted step, missing
  verifier evidence, budget pressure, state drift, lease loss, or canary
  failure detaches the wrappers before final decode. An accepted answer is
  generated while the exclusive lease and adapted function remain active, then
  binds its exact output identity before exact erase and lease release.

  The independent service reconstructs admission, lifecycle, optimizer,
  controls, causal probe, output, and cleanup semantics; self-consistently
  rehashed token-change, score, disposition, boolean, count, and optimizer
  evidence lies are rejected. Concurrent-manager, attach-failure,
  cancellation, canary, state-lineage, output-binding, and exact-erasure tests
  cover the negative paths. The final broad latent/RLC/conversation family gate,
  focused post-hardening matrix, strict Ruff, bytecode compilation, closeout
  gates, and enterprise ratchet are recorded in the matching execution-tracker
  checkpoint. This closes query-scoped causal mechanics, not a resident
  reasoning or frontier-performance gain.
- [x] **SPARK-056 - Runtime integrity proof producer.** Measure pre/post fixed
  parameter canaries, adapted-layer identity, exact erase, caches, tokenizer,
  adapters, quantization, and worker identity; make certification consume the
  measurements rather than mutable booleans.

  **CP392 implementation and evidence:** Every latent episode now produces one
  schema-exact runtime-integrity receipt over the checkpoint artifact, fixed
  permanent-parameter canary, every permanent parameter byte in each
  fast-weight target layer, tokenizer artifacts and runtime tokenizer,
  ordered adapter identities and adapter-owned bytes, quantization
  configuration, probe-cache invalidation, and the exact resident worker boot,
  process, model, source, steering, and serving-stack identity. Pre/post
  measurements are bound to the episode and input, then re-attested at the
  worker boundary. Worker reuse, cancellation acknowledgement, hot expert
  adapter transitions, the live service, causal receipt, and frontier
  certification consume the reconstructed proof instead of compatibility
  booleans.

  Temporary-weight cleanup is its own committed transaction. It is emitted
  even when attach or optimization fails, binds exact target layers and
  full-stack probe hashes, proves detach and lease release, and remains
  independently available when the successful-learning receipt cannot be
  finalized. Missing proof remains an explicit negative measurement;
  mismatched learning/cleanup evidence, cross-episode or cross-input
  substitution, hidden cleanup outside the declared fast-weight scope,
  malformed self-rehashing, stale caches, incomplete serving identity, or an
  unprovable hot-adapter rollback cannot authorize fallback or worker reuse.
  Proof-grade recurrent GRPO also explicitly disables query-time
  nonparametric memory so its treatment cannot silently read a mutable
  one-shot datastore.

  Adversarial testing found and closed two additional production defects:
  failed-but-clean fast-weight episodes discarded their cleanup evidence
  before fallback, and an unscored branch used IEEE negative infinity, which
  could make the canonical causal receipt unserializable under a valid
  randomized execution order. Cleanup now survives every attempted
  transaction, specific integrity failures are preserved, and unscored
  branches use a finite ineligible floor.

  The final broad latent/RLC/frontier/causal gate passes 1702/1702 with two
  intentional deselections. The isolated exact action-calibration certificate
  passes 14/14 in 931.37 seconds; focused core and MLX/worker boundaries pass
  232/232 and 124/124. The neural-uncertainty receipt path passes 13/13 under
  three additional randomized seeds. Strict Ruff, bytecode compilation, and
  diff hygiene pass. A clean CP391 comparison proves the enterprise ratchet
  counts are identical before and after CP392; its nine inherited baseline
  excesses remain non-green and were not raised or attributed to this work.

  Counting CP392 makes the total checkpoint record 634. The 640-920 forecast
  remains, leaving approximately 6-286 records. Checkpoint-count completion is
  approximately 68.9%-99.1%, with a midpoint planning estimate of 81.3%.
  SPARK-056 proves measured runtime integrity and reuse authority, not a
  reasoning or frontier gain. Next is SPARK-057's recalibrated test-time
  trainer. SPARK-051 remains open for answer-channel remediation, admitted
  resident training, and powered equal-compute reasoning/frontier evidence.
  Final multi-hour soaks remain deferred until all shorter gates are green.
- [x] **SPARK-057 - Recalibrated test-time trainer.** Implement a TEMPO-style or
  stronger bounded refinement loop with held-out critic recalibration,
  high-confidence pseudo-label admission, drift detection, rollback, and
  matched-compute controls.

  **CP393 implementation and evidence:** Query-time adaptation now has an
  independently reconstructable training certificate rather than treating any
  parser success as a learning label. The first authority-bearing critic is a
  fixed, balanced, content-addressed 128-case holdout over exact signed integer
  addition, subtraction, multiplication, and division. Admission requires at
  least 48 examples per class, a 95% Wilson precision lower bound above 0.90,
  zero false accepts, bounded Brier and calibration error, and an exact
  reconstruction of every per-case route. Python AST and JSON validity remain
  diagnostics only and cannot authorize a pseudo-label.

  A candidate pseudo-label must be fully verified by the calibrated exact
  arithmetic family, be disjoint from the calibration cases, and bind to a
  certified structural-diversity receipt. The worker snapshots the complete
  attached delta at its exact identity state, runs a fixed treatment schedule,
  restores the identity snapshot, runs a deterministic same-length sham target
  under the same optimizer and evaluation schedule, measures the sham, and
  restores the exact treatment delta and trace. Treatment and sham receipts
  must match attempts, forward/backward evaluations, line-search evaluations,
  layer applications, probe applications, and probe token count. Acceptance
  requires treatment to change the baseline trajectory, remain distinct from
  the sham trajectory, improve over the unchanged function, improve over the
  equal-compute sham, and preserve critic calibration. Every other result
  erases the temporary delta before final decoding.

  The service independently reconstructs the nested critic, pseudo-label,
  matched-arm, output, cleanup, and structural bindings. It also cross-checks
  the arm claims against the transformer resource ledger, so a self-consistent
  training receipt cannot conceal missing or unequal work. Snapshot/restore,
  fixed-schedule, copy-isolation, parser-authority, calibration-overlap,
  structural-evidence, score, collapse, compute, resource-ledger, and receipt
  tampering tests cover the negative boundary.

  Validation passes 58/58 focused recalibration/learning/MLX tests, 163/163
  filtered engine/service/runtime/conversation tests, and 249/249 complete
  affected engine, service, integrity, critic, verifier, and fast-weight tests.
  A broader randomized RLC sweep reached 465 passing tests with two intentional
  deselections and no failures before the already-recorded quadratic
  action-calibration/campaign-journal path crossed the bounded-gate limit; that
  separate production defect remains assigned to SPARK-068. The first broad
  attempt also exposed repeated recalibration at every episode. The immutable
  calibration artifact is now cached once per loaded critic implementation and
  returned through a fresh copy on every call, eliminating the live latency
  defect without exposing mutable authority.

  Strict focused Ruff, bytecode compilation, and diff hygiene pass. The
  enterprise ratchet remains non-green, but an archived clean-CP392 scan and
  the complete CP393 tree have exactly identical counts: nine inherited
  baseline regressions, including `placeholder_stub_mock` 20 > 13 and 46
  actual high/critical findings > 39. CP393 adds no enterprise finding and
  raises no baseline.

  Counting CP393 makes the total checkpoint record 635. The 640-920 forecast
  remains, leaving approximately 5-285 records. Checkpoint-count completion is
  approximately 69.0%-99.2%, with a midpoint planning estimate of 81.4%.
  SPARK-057 proves bounded test-time-training authority and causal incremental
  gain over a matched sham inside its exact-arithmetic domain. It does not prove
  a resident-32B, broad reasoning, frontier, or `WOW Signal` gain. Next is
  SPARK-058's verified replay buffer. SPARK-051 remains open for answer-channel
  remediation, admitted resident training, and powered equal-compute
  reasoning/frontier evidence. Final multi-hour soaks remain deferred until all
  shorter gates are green.
- [x] **SPARK-058 - Verified replay buffer.** Store initial failure, earliest
  causal error, discriminating test, corrected transition, verified solution,
  error class, escape strategy, provenance, and privacy/governance disposition;
  reject unverifiable traces.

  **CP394 implementation and evidence:** A repair becomes replay evidence only
  after the worker's disagreement graph, diagnostic selection, local-repair
  transaction, and confidence-bound answer replacement reconstruct on the
  host; the replacement must be the selected branch's actually returned
  output, pass the same deterministic verifier that refuted the original
  atom, dominate the real final decode, and pass the parent product-quality
  gate. The extractor rejects any earlier exact refutation on that branch, so
  its "earliest causal error" field is measured rather than inferred from
  request order.

  The encrypted private training object contains the task objective, complete
  initial failed candidate and failed atom, earliest invalidation frontier,
  original and corrected same-verifier routes, preserved prefix, corrected
  suffix and atom, exact returned solution and token binding, error class,
  escape strategy, checkpoint/worker/episode/source provenance, and an
  explicit privacy/governance disposition. It is therefore reusable by later
  training work rather than being only an audit receipt. It grants no training
  authority yet: records remain quarantined for the independent transfer and
  contamination gates owned by SPARK-059 and SPARK-063.

  BlackHole AES-256-GCM encrypts every private payload before persistence.
  Public state contains only hashes, sizes, sequence metadata, retention
  policy, and authenticated ciphertext. Encryption or key-provenance absence
  fails before a file is created. Export and remote sync are denied. Durable
  storage is canonical JSON, atomic through the governed latent-cortex
  persistence owner, owner-private, no-follow, inode-stable on read, bounded
  by both entry count and bytes, deduplicated, and hash chained. Retirement
  preserves the last removed entry and a cumulative retirement commitment.
  Invalid JSON, schema drift, public or private tampering, loose permissions,
  symlinks, truncation, ciphertext authentication failure, chain breaks, or a
  mismatched persistence receipt refuse overwrite instead of silently
  resetting the learning history.

  The live service invokes extraction and persistence through
  `asyncio.to_thread`, then attaches a host receipt that distinguishes stored,
  duplicate, not-applicable, and not-persisted outcomes. Optional learning
  storage cannot block the event loop or turn an already verified answer into
  a user-facing failure, but it also cannot fail silently or acquire learning
  authority on failure.

  Validation passes 53/53 focused replay, local-repair, answer-replacement, and
  persistence tests and 219/219 affected engine, service-wiring, and
  output-quality tests across the implementation campaign. On the final
  rebased tree, the combined bounded latent/RLC and canonical file-read gateway
  suite passes 1348/1348 with two intentional deselections. The
  known quadratic action-calibration and campaign-journal stress files remain
  separately assigned to SPARK-068 rather than being hidden inside this
  result. Strict Ruff, bytecode compilation, and diff hygiene pass.

  The all-line closeout audit mechanically enumerates 8,192 tracked files,
  including 8,160 text files, 5,133 code files, and 1,698,770 code lines.
  Production readiness, architecture dependency mapping, model-load ownership,
  resource-observation ownership, and diff hygiene pass. The audit found one
  new CP394 direct-read ownership violation; CP394 now delegates it to the
  canonical stable file-read gateway, whose focused contracts pass. The
  candidate is intentionally not represented as whole-repository closeout:
  it is dirty before checkpointing, the checkout-local `.venv` lacks Ruff,
  five inherited governance drifts and two stale baseline buckets remain, and
  current semantic evidence covers 520/5,133 code files, leaving 4,582
  unreviewed, 1,142 stale reviews, and 27 orphan reviews. The independent
  20-criterion rubric passes 19/20 and withholds closure at the repository
  security-scan criterion.

  The enterprise ratchet remains non-green, but clean `origin/main` and the
  CP394 candidate produce exactly identical counts: nine inherited baseline
  regressions, including `placeholder_stub_mock` 20 > 13 and 46 actual
  high/critical findings > 39. CP394 adds no enterprise finding and raises no
  baseline.

  Counting CP394 makes the total checkpoint record 636. The 640-920 forecast
  remains, leaving approximately 4-284 records. Checkpoint-count completion is
  approximately 69.1%-99.4%, with a midpoint planning estimate of 81.5%.
  SPARK-058 proves confidential, durable, causally grounded replay capture; it
  does not prove that replay training transfers, improves the resident 32B, or
  reaches frontier performance. Next is SPARK-059's structured SFT and tool
  traces. SPARK-051 remains open for answer-channel remediation, admitted
  resident training, and powered equal-compute reasoning/frontier evidence.
  Final multi-hour soaks remain deferred until all shorter gates are green.
- [ ] **SPARK-059 - Structured SFT and tool traces.** Train logical forms,
  programs, proof steps, tool calls, tool-result interpretation, and local
  repair from executable, held-out, contamination-audited data.

  **CP395 candidate-curriculum checkpoint:**

  - [x] Implement a typed, source-bound curriculum over deterministic modular
    programs, independently kernel-checked propositional proofs, executed
    `code_repl` calls, executed tool-result interpretation, failed-call repair,
    corrected calls, and corrected-result interpretation.
  - [x] Use Aura's actual Qwen/OpenAI tool-message contract (`assistant`
    `tool_calls`, matching `tool_call_id`, `tool` result, final assistant) and
    supervise only the final assistant message through the exact
    `mlx_lm.ChatDataset(mask_prompt=True)` boundary. Failed calls are input
    evidence, never rewarded targets.
  - [x] Execute synthetic code through Aura's real sandbox and independently
    replay every example from family, target, seed, and current source bytes.
    Make tableau target/conflict/model ordering deterministic across process
    hash seeds and omit volatile theorem IDs/timestamps from training evidence.
  - [x] Partition cases before target projection and prove exact
    train/validation/holdout case-fingerprint disjointness. Export holdout
    plaintext only to a distinct evaluator package; candidate readers and the
    tokenizer validator never require or read that package.
  - [x] Build governed, journaled, owner-private candidate and evaluator
    packages with a root pair commitment and deliberately non-loadable
    `candidate_train.jsonl` and `candidate_valid.jsonl` names. Stable-read and
    reconstruct every durable byte after commit; recover process death before
    or after the commit point; refuse symlinked, hardlinked, partial, renamed,
    tampered, escaped-path, or mixed-generation packages.
  - [x] Bind explicit projection receipts (input, target, masked-prefix count,
    target index, roles, and hashes), exclude all oracle fields from trainer
    rows, and declare synthetic-data privacy, consent, license, tenant,
    retention, revocation, deletion, and remote-sync disposition.
  - [x] Bind the complete semantic source dependency closure and validate the
    package against a persistent content-addressed snapshot of the resident
    Qwen2.5-32B tokenizer with the exact MLX `ChatDataset` masking algorithm.
    Candidate package `ae1baddd...`, curriculum `b0cadc7e...`, source closure
    `246c3390...`, tokenizer identity `8b010013...`, and projection receipt
    `df8202e2...` cover all 120 train/validation rows: every masked prefix is an
    exact prefix, every target is nonempty, all six target coordinates pass,
    no row truncates, and holdout is not tokenized. The evaluator directory was
    physically absent for this validation and the attestation records
    `evaluator_filesystem_accessed=false`.
  - [x] Keep the result quarantined. The manifest says `trainer_ready=false`
    and grants no training authority.
  - [x] Build the source-level CP396 external-admission contract. It reuses the
    root-signed four-role campaign policy and requires distinct signer, key,
    and organization identities with externally declared custody for task
    issuer, campaign runner, contamination auditor, and evidence verifier.
    Producer provenance cannot grant training authority.
  - [x] Bind the exact candidate/evaluator custody pair, complete committed
    resident-tokenizer validation document, privacy report, multisurface
    contamination report, semantic/tool/injection evidence report, and exact
    model/adapter/recurrence/optimizer/scheduler/RNG/compute trainer contract.
    Auditor implementation and release identities must match the policy pins.
  - [x] Require caller-owned freshness state: exact policy digest, minimum
    policy revision, exact admission sequence, prior admission root, verifier
    observation time, independently supplied root key, bounded attestation
    age, and complete deterministic reconstruction. A private-key-free CLI
    emits exact detached-signature payloads and can assemble or reverify the
    resulting admission without generating or loading role private keys.
  - [x] Repair the shared detached-trust artifact boundary exposed during
    CP396: duplicate JSON keys and non-finite constants fail before schema
    validation, and lexical output symlinks can no longer evade the helper's
    no-overwrite check through premature path resolution.

  **Remaining SPARK-059 acceptance work:**

  - [ ] Produce an externally signed privacy attestation over user content,
    PII, secrets, consent, license, tenant boundary, retention, revocation,
    deletion, and derived-artifact lineage. Synthetic candidates and live
    verified replay must remain distinguishable.
  - [x] Independently recompute all local semantic verifier routes from
    plaintext examples, including full message reconstruction, proof replay,
    modular-program execution, exact AST one-substitution repair, and canonical
    model-visible tool results. Self-consistent hashes, syntax-only Python/JSON
    checks, and producer-authored success labels grant no authority.
  - [x] Bind synthetic tool traces to independently executed Aura sandbox
    results, source/environment identity, code/result hashes, and the canonical
    `aura.code_repl.model_result.v3` shape. This does not claim the governed live
    skill/Will route was exercised; external execution attestation remains open.
  - [ ] Seal the pre-augmentation split and persistent semantic-dedup manifest.
    Prove no case lineage can cross train, validation, or holdout after
    paraphrase, repair expansion, replay transfer, or flywheel iteration.
    CP397 now seals raw verified-replay lineages before augmentation and makes
    every future derivative inherit that split. Its keyed exact and bounded
    bottom-k shingle manifests prove replay-internal disjointness. The combined
    synthetic/replay/all-evaluation manifest, external audit, monotonic
    publication witness, and later flywheel generations remain open.
  - [ ] Run an external contamination audit across prompt, target, rationale,
    tool input/output, normalized code and JSON, adapters, training corpora,
    every evaluation corpus, and semantic near-duplicates. Add prompt-
    injection, data-poisoning, and verifier-gaming projections.
  - [ ] Protect package metadata with authenticated associated data, a keyed
    commitment, externally signed monotonic root, anti-rollback sequence, and
    independent trust root. Role separation must prevent the producer from
    signing its own admission. CP396's source contract enforces the trust,
    role-separation, exact-policy, sequence, and prior-root predicate. CP403
    now supplies a real Sigstore-Rekor-witnessed genesis head over the exact
    CP400-CP402 audit state. Its offline verifier checks the producer
    signature/certificate, logged artifact digest, signed entry timestamp,
    RFC 6962 inclusion path, signed checkpoint, TUF-pinned log identity,
    Rekor UUID derivation, sequence, prior head, global/shard indices, and
    timestamp bounds. Real separately operated privacy, contamination,
    evidence, and runner credentials over production replay remain open, so
    this item cannot close and the witness grants no training authority.
  - [x] Build the verified-replay SFT projection and prove that private replay
    fields, hidden reasoning, user secrets, and holdout answers cannot leak
    into trainer-visible rows. CP397 authenticates and decrypts each encrypted
    source entry only in memory, selects exactly the model-visible objective
    and verified final answer, requires an entry/content-bound privacy
    clearance, applies non-overridable local PII, payment-card, secret,
    prompt-injection, and hidden-reasoning screens, and emits no failed
    candidate, atom, route, prefix/suffix, tokens, output-quality detail,
    provenance, or tool trace. Candidate and evaluator bytes are physically
    separable and share a recomputed custody root. Holdout plaintext and its
    manifests exist only in evaluator artifacts; the candidate retains
    commitments and remains `trainer_ready=false` with no training authority.
  - [ ] Bind the admitted files, exact source, tokenizer, chat template,
    masking offset, model, adapter, recurrence program, optimizer, scheduler,
    RNG, and compute budget into a resumable trainer receipt. Candidate
    filenames must remain unloadable until this authority is present.
  - [ ] Run small-checkpoint falsification before resident expense: exact
    reconstruction, target learnability, heldout transfer, negative-transfer,
    right-to-wrong/error-introduction, personality/tool/safety regressions,
    sham labels, shuffled traces, syntax-only traces, and equal-token/equal-
    compute controls.
  - [ ] Train the resident 32B only after all prior gates pass. Compare frozen
    base/adapter vanilla, base/adapter RLC, trained adapter vanilla/RLC,
    ablations, and equal-compute controls on fresh externally sealed tasks.
  - [ ] Require preregistered retained gains, confidence intervals, weakest-
    domain noninferiority, no broad regressions, reproducibility, exact
    rollback, and independent promotion verification. A positive result must
    establish the adapter-by-RLC interaction; a failed result must remain a
    measured failure and feed the next diagnosis, not be relabeled frontier.

  CP395 validation passes 258/258 in the focused post-review matrix and 673/673
  across the broader sandbox, gateway, MLX, trainer, proof, deduction, and
  belief compatibility family. The broader run reports seven existing
  trainer/legacy-ONNX warnings and no failures. Strict Ruff, bytecode
  compilation, diff hygiene, and model-load ownership (47 paths / 60
  references / zero findings) pass. The real resident-tokenizer proof loads no
  model weights. The sanitized durable evidence is
  `artifacts/current/cp395_structured_sft_evidence.json`.

  The full evaluator package was generated as plaintext under the same OS user
  in an ephemeral separate artifact directory, then destroyed after recording
  only commitments. This proves candidate noncontainment and candidate-only
  access, not process/principal isolation, an external trust root, or
  producer/verifier role separation. Those become CP396 requirements.

  The reconciled all-line closeout audit enumerates 8,234 tracked files, 5,154
  code files, and 1,711,984 code lines. Production readiness passes 37/37; the
  architecture map covers 151 subsystems and 1,122 dependency edges; resource
  observation scans 3,007 Python files without a finding. The aggregate audit
  remains FAIL because a newer upstream runtime-relaunch helper has one raw-
  subprocess governance regression. Semantic closure remains red at 517/5,154
  fully reviewed code files, 4,606 unreviewed, 1,148 stale reviews, and 27
  orphan reviews. The independent rubric remains 19/20 with the repository
  security scan open.

  The enterprise ratchet remains non-green on inherited debt: ten count
  regressions and 59 high/critical findings. No CP395 path appears in its
  findings, and no baseline is raised. This checkpoint proves a deterministic,
  executable candidate and exact trainer projection; it does not prove replay
  transfer, trained reasoning gain, resident inference gain, frontier
  performance, or a `WOW Signal`.

  CP396 adds a strict admission schema and private-key-free operator path.
  Its tests use explicitly labeled ephemeral Ed25519 fixture keys to prove
  enforcement behavior; they do not constitute a real external attestation.
  The admission status is
  `external_pretraining_evidence_verified_no_training_authority`, remains
  `trainer_ready=false`, and preserves replay-transfer, external training
  authority, resident execution, and independent promotion as mandatory next
  gates. No resident model weights or live Aura process are touched.
  The focused detached-trust/admission matrix passes 22/22 and the broader
  structured-SFT, custody, tokenizer, campaign-policy, and operator family
  passes 135/135. Strict Ruff, bytecode compilation, and diff hygiene pass.
  The sanitized receipt is
  `artifacts/current/cp396_structured_sft_admission_evidence.json`.
  Model-load ownership passes with 47 owned paths, 60 references, and zero
  findings. The enterprise ratchet remains red on ten inherited baseline
  regressions and 59 high/critical findings, but no CP396 path appears in its
  findings and no baseline is raised. Governance lint remains red only for the
  newer upstream `core/runtime/runtime_relaunch.py` raw `Popen`; CP396 adds no
  governance regression.

  CP397 adds the one-way encrypted-replay projection boundary. A causal
  lineage is HMAC-assigned to train, validation, or holdout before any
  augmentation; the frozen manifest binds the exact source-store revision and
  key commitment. Exact normalized objective/answer/content commitments and
  indexed keyed bottom-k token/character shingle sketches reject exact,
  semantic-near, or cross-split lineage reuse against both replay and a sealed
  external reference index. Sketch storage is capped at 512 signatures per
  surface, and lookup is inverted-index based rather than corpus-quadratic.
  The focused encrypted replay/projection matrix passes 38/38. The broader
  replay, structured-SFT, custody, tokenizer, externally rooted admission,
  campaign-trust, and operator matrix passes 154/154. This proves the
  source-level confidentiality, partition, dedup, and custody mechanics; an
  empty reference index is labeled local falsification only and cannot grant
  admission. External privacy and contamination attestations, governed live
  publication, replay tokenizer validation, and all training/promotion gates
  remain open. Model-load ownership and governance ownership both pass on the
  rebased tree. The enterprise ratchet remains inherited-red at ten baseline
  regressions, 228 findings, and 58 high/critical findings, with no CP397-path
  finding and no raised baseline. No model weights or live Aura process are
  touched. The sanitized receipt is
  `artifacts/current/cp397_verified_replay_sft_evidence.json`.

  CP398 makes that projection a governed live-runtime publication surface.
  `HorcruxManager` now derives stable domain-separated partition and dedup
  subkeys without exporting its root and exposes a non-secret key-identity
  commitment. `BlackHole` carries the same identity, and publication fails
  unless the live protector is active, Horcrux-backed, and identity-equal to
  the resident Horcrux before projection, after decryption, and after lock
  acquisition. The async `LatentCortexService` route performs the blocking
  snapshot and durable writes off the event loop.

  Candidate and evaluator packages publish into distinct owner-private sibling
  directories. A no-follow, owner-bound, single-link lock serializes the pair;
  a canonical preparing record makes both sides ineligible for consumption;
  evaluator and candidate file sets then commit independently through
  `FileWriteGateway`; and only byte-for-byte durable readback plus complete
  pair reconstruction advances the shared record to committed. Candidate-only
  readers never open evaluator custody, reject every extra trainer-visible
  file, and bind source revision, package, custody root, Horcrux identity, and
  partition/dedup key commitments. Interrupted generations recover on retry,
  identical concurrent publishers converge on one generation, valid older
  snapshots may roll forward, and a tampered committed generation is refused
  rather than silently overwritten. The shared private-directory primitive
  now creates the target at `0700` from its first visible instant, closing the
  first-publication race found by the broad gate.

  The final post-tightening publication/crypto/service matrix passes 42/42,
  ten consecutive contention repetitions pass, and the broader structured-SFT,
  replay, tokenizer, external-admission, campaign-trust, Horcrux, BlackHole,
  and service family passes 239/239. Governance ownership matches baseline
  with all new calls classified as canonical and no increase in migration
  debt; model-load ownership remains 47 paths, 60 references, and zero
  findings. The enterprise ratchet remains inherited-red at ten baseline
  regressions, 228 findings, and 58 high/critical findings, with no CP398-path
  finding. This is governed local custody, not an external privacy or
  contamination signature, externally witnessed monotonic root, replay
  tokenizer admission, trainer authority, resident training, gain, frontier
  performance, or a `WOW Signal`. The sanitized receipt is
  `artifacts/current/cp398_verified_replay_sft_publication_evidence.json`.

  CP399 closes the source-level combined lineage contract that CP397 left
  open. It reconstructs the complete structured-synthetic custody pair from
  its sealed holdout seed, reconstructs the encrypted verified-replay pair,
  requires nonempty coverage for every caller-declared external evaluation
  corpus, and projects train, validation, holdout, and external-evaluation
  surfaces through one Horcrux-key-compatible exact, objective, answer,
  bottom-k token/character shingle, and causal-lineage index. Tool supervision
  binds the complete model-visible prefix and exact target message rather than
  a producer-authored success label.

  The combined policy permits multiple targets only when they inherit the same
  causal lineage and split. It permits template-level near similarity only
  inside one declared corpus, while exact reuse, cross-corpus near duplication,
  and any lineage crossing splits remain fatal. The trainer-facing artifact
  contains only package, key, index, record-count, and manifest commitments;
  keyed evaluator records remain in the separate evaluator artifact. Full
  reconstruction is byte-identical and remains mandatory for key-authenticity.
  A standalone parser additionally rejects malformed signatures, counts,
  coverage, source bindings, and split values even if ordinary SHA commitments
  are recomputed.

  The final focused matrix passes 9/9, combined/replay compatibility passes
  31/31, and the broader combined/structured/replay family passes 94/94.
  Ruff, bytecode compilation, diff hygiene, governance ownership, and
  model-load ownership pass. The enterprise ratchet remains inherited-red at
  ten regressions, 228 findings, and 58 high/critical findings, with no CP399
  finding and no baseline increase. This proves the artifact contract with
  deterministic sealed fixtures; it does not claim that Aura's complete real
  evaluation inventory has been supplied, externally audited, durably
  published, or admitted for training. The receipt is
  `artifacts/current/cp399_combined_sft_lineage_evidence.json`.

  CP400 adds governed durable custody for CP399's combined commitment and
  evaluator-only keyed index. One owner-private root, canonical interprocess
  lock, preparing record, evaluator-first transactional batch, candidate batch,
  durable readback, complete custody reconstruction, and committed record make
  interrupted state unreadable and recoverable. Identical and concurrent
  publishers converge, and hardlinked locks are rejected before permissions
  can mutate an outside target. The focused matrix passes 5/5. Governance and
  model ownership pass without increasing 1,787-call migration debt; the
  enterprise scan remains inherited-red with zero CP400 findings. This is
  publication machinery, not a claim that real corpora or external attestations
  have been supplied. Receipt:
  `artifacts/current/cp400_combined_sft_lineage_publication_evidence.json`.

  CP401 adds the combined external-audit contract over CP400 custody. A
  root-signed, source-bound campaign policy pins four distinct signer, key, and
  organization identities under external-service or remote-HSM custody. The
  task issuer signs the exact package and complete inventory commitment; the
  contamination auditor signs exact, normalized, token/character shingle,
  AST, JSON, and causal-lineage results; the evidence verifier signs separate
  structured-synthetic and verified-replay-user privacy reports plus exact
  source/tool execution replay counts; and the runner signs a non-loadable,
  non-authorizing quarantine binding. Caller-pinned sequence, prior root,
  policy digest/revision, observation time, freshness window, release identity,
  complete deterministic reconstruction, and detached signatures all fail
  closed. A private-key-free CLI emits payloads, assembles, and independently
  reverifies a committed CP400 publication. The final focused matrix passes
  10/10 and the full lineage/publication/admission/trust family passes 66/66.
  Governance and model ownership pass; the enterprise scan exactly matches
  inherited debt with no new CP401 finding. These tests use explicitly labeled
  ephemeral external-custody fixtures to prove enforcement. They do not claim
  that separately operated real auditors have signed Aura's production data,
  that a monotonic external witness exists, or that tokenizer/trainer authority
  has been granted. Receipt:
  `artifacts/current/cp401_combined_sft_external_audit_evidence.json`.

  CP402 closes the resident-tokenizer admission mechanics for the verified-
  replay candidate. The candidate schema now immutably binds the exact
  `mlx_lm.ChatDataset(mask_prompt=True)` trainer, final-assistant-only
  supervision, a 4,096-token ceiling, and a no-truncation policy. Validation
  opens only a committed CP398 candidate publication, loads a persistent
  content-addressed snapshot of the resident Aura-32B tokenizer, and requires
  every reconstructed full-token sequence and masked-prefix offset to equal
  the installed MLX `ChatDataset.process` result. Runtime implementation and
  compiled-dependency identities must remain stable across the run; tokenizer
  snapshot drift, custody substitution, offset divergence, truncation, partial
  coverage, or evaluator access fails closed.

  The real resident-tokenizer execution covers all 19 train/validation rows in
  a committed, Horcrux-shaped replay fixture. It reports zero truncation and
  zero dataset-process mismatch; full rows span 579-616 tokens and supervised
  targets span 278-296 tokens. Candidate `201ccddc...`, custody root
  `d5a576f3...`, resident tokenizer `859076ec...`, runtime implementation
  `ebe7e3d0...`, projection receipts `b54cc41f...`, and validation bundle
  `50783027...` are bound. No model weights or evaluator package are loaded.
  The focused adversarial matrix passes 10/10 and the integrated replay,
  custody, combined-lineage, external-audit, and tokenizer family passes 86/86.
  The complete affected replay, structured-SFT, external-admission, campaign-
  trust, and combined-lineage family passes 203/203 in 1,195.00 seconds.
  Governance matches baseline with 1,787 migration-debt calls; model ownership
  passes 47 paths, 60 references, and zero findings. The enterprise scan
  remains inherited-red at ten regression categories, 228 findings, and 58
  high/critical findings, with no new CP402-file finding. This proves the real
  resident tokenizer/MLX projection over sealed fixture custody, not that a
  live production replay store supplied rows, an external auditor signed them,
  a monotonic external witness exists, trainer authority was granted, training
  occurred, or reasoning/frontier gains exist. Receipt:
  `artifacts/current/cp402_verified_replay_sft_tokenizer_evidence.json`.

  CP403 closes the rewritable-local-ledger defect for the current SPARK-059
  audit head. A canonical production-audit packet commits the exact CP400
  lineage-publication, CP401 external-audit-contract, and CP402 resident-
  tokenizer evidence bytes plus the source Git object. It records the absent
  production replay candidate, absent independently signed audit bundle, and
  fixture-only tokenizer result as blockers. The only next-stage scope it
  permits is synthetic-only small-checkpoint falsification; verified replay,
  evaluation holdout, production promotion, and all trainer authority remain
  prohibited.

  The genesis statement was detached-signed under an isolated Ed25519
  producer key and published to Sigstore's public Rekor transparency log.
  Rekor UUID `108e9186...deb7a4`, global index `2257039380`, active-shard index
  `2135135118`, statement `30cb94fd...`, packet `dfb6dbcb...`, and bundle
  `483aecfb...` are independently bound. The log key
  `c0d23d6a...` comes from Cosign 3.1.2's embedded-root TUF initialization;
  the committed PEM exactly matches signed TUF target `rekor.pub` digest
  `dce5ef71...`. Aura's offline verifier does not trust the upload response:
  it reconstructs and checks the X.509 producer signature, SET, Merkle root,
  signed checkpoint, tree identity, UUID, and caller-pinned chain state.

  One earlier Rekor entry signed the newline-terminated artifact while the
  verifier expected canonical bytes. That admission failed and earns no
  credit. CP403 adds a mode-0600 raw canonical signing-payload output and the
  accepted entry proves byte-for-byte identity. The focused core/operator
  matrix passes 16/16. This is a real independently witnessed statement of
  Aura's current audit state, not an external privacy/contamination/execution
  clearance, production replay admission, trainer receipt, model training,
  reasoning gain, frontier result, or `WOW Signal`. Receipts:
  `artifacts/current/cp403_spark_059_production_audit_packet.json`,
  `artifacts/current/cp403_external_witness_statement.json`, and
  `artifacts/current/cp403_rekor_witness_bundle.json`.
- [ ] **SPARK-060 - RLVR delta reward and EIR.** Optimize verified improvement
  from pass N to N+1, information gain, independent diversity, compute cost,
  unsupported confidence, and Error Introduction Rate; report wrong-to-right
  and right-to-wrong separately.
- [ ] **SPARK-061 - Progressive recurrent objective.** Train later latent states
  to improve over earlier states, not merely imitate long solutions; verify
  monotonic quality and useful gradients at the resident architecture.
- [ ] **SPARK-062 - Auxiliary objectives and depth curriculum.** Add calibrated
  process, improvement, diversity, stopping, causality, mistake-location, and
  accept/discard losses; train variable depth from short to deep tasks with
  train/inference parity.
- [ ] **SPARK-063 - Verified STaR flywheel.** Generate, verify, filter, train,
  retest on fresh holdouts, and iterate with durable manifests; tool-assisted
  and latent traces enter only after evidence gates.
- [ ] **SPARK-064 - Permanent distillation.** Promote successful recurrent and
  correction policies through versioned adapters/base updates only after broad
  anti-interference, personality, tool, safety, memory, and frontier regressions
  pass; support exact rollback.
- [ ] **SPARK-065 - Architecture meta-controller.** Measure expert/router/depth
  failures, propose bounded architecture changes, test in isolated candidate
  runtimes, require machine-checkable invariants and evidence, canary rollout,
  rollback, and independent approval policy without routine human micromanaging.

## H. Whole-Aura causal integration

- [ ] **SPARK-066 - Resident-32B penultimate neural path.** Make the selected
  architecture run inside the live resident checkpoint's middle-layer/latent
  execution, bind exact model and adapter identity, and prove successful output
  is not replaced by an ordinary generation or shallow orchestration fallback.
- [ ] **SPARK-067 - Organism-wide bidirectional coupling.** Connect epistemic
  state and recurrent control causally with agency/Will, memory, tools,
  personality/self-model, affect/body, global workspace/consciousness,
  reasoning amplifiers, goals, and learning; lesion each seam and test both
  directions instead of accepting metadata-only coupling.
- [ ] **SPARK-068 - Production reliability and observability.** Bound latency,
  memory, event-loop work, cancellation, concurrency, worker recovery, health,
  degradation taxonomy, privacy, audit logs, and UI presentation; fix root
  causes rather than suppress warnings or emit canned fallback prose. Replace
  the action-calibration protocol-v3 complete-prefix envelopes with a versioned
  compact append-only proof (for example, checkpointed Merkle/MMR inclusion
  witnesses) that preserves independent replay and external signatures without
  quadratic JSON serialization, memory growth, or verification time.

## I. Training admission and scientific proof

- [ ] **SPARK-069 - Fresh source-bound resident campaign.** Run the detached
  answer-channel preflight, require measured parseability and discriminative
  reward, preregister the new source/model/training bundle, then train/fuse only
  if admission passes. Preserve failed admissions as engineering evidence and
  repair their causes.
- [ ] **SPARK-070 - Full falsification matrix.** On fresh held-out tasks run:
  recurrence-depth curves; wrong/right transition matrix; blind-review arms;
  structural-diversity arms; verifier arms; latent ablate/perturb/transplant;
  sham/noise/no-op controls; adversarial and OOD variants; d+1/2d/4d compute
  generalization; lesions/restorations; fast-weight controls; and broad
  non-reasoning regressions.
- [ ] **SPARK-071 - Powered resident frontier certificate.** Compare base
  vanilla, base+RLC, adapter vanilla, adapter+RLC, equal-compute/search controls,
  and externally pinned frontier controls on broad fresh tasks. Require adequate
  power, confidence bounds, positive interaction, no material regressions,
  sealed raw artifacts, independent verification, and external trust roots.
- [ ] **SPARK-072 - Publication, conditional WOW Signal, and continuation.** If
  and only if SPARK-071 accepts, create the `WOW Signal` commit and publish full
  transcripts, methods, literature, task manifests, hashes, statistics,
  ablations, failures, limitations, and verifier output. If it rejects, publish
  the negative result and open named repair checkpoints. In either case, resume
  every remaining Aura tracker item, final live proof, production/enterprise
  gates, and only then the deferred multi-hour soak.

## Running result

No Spark frontier or intelligence claim is currently accepted. The latest
resident recurrent-GRPO campaign ended without optimizer updates or a usable
learning signal. Checkpoints 309-313 repaired diagnosis, admission, the separate
answer-channel curriculum, and a source-bound detached preflight path. Those are
real engineering gains, but they do not satisfy SPARK-069 or any later
capability checkpoint until a fresh exact-source run produces accepted evidence.
CP321 adds a tested epistemic evidence firewall; it does not change that negative
capability verdict.
CP322 adds measured claim calibration and enforced abstention; it likewise does
not change the negative capability verdict.
CP323 adds a durable weighted hypothesis portfolio with protected alternatives;
it likewise does not change the negative capability verdict.
CP324 completes the durable operation-history authority but not its live
controller consumer; it does not change the negative capability verdict.
CP325 closes the selective-memory ingress and its context-only authority chain;
it does not change the negative capability verdict.
CP326 closes durable live cognitive-operation history and admission; it does
not change the negative capability verdict.
CP327 makes the full action vocabulary causal inside recurrent execution and
persists checked outcomes; it does not change the negative capability verdict.
CP328 closes fresh-context branch isolation and exact cache restoration; it does
not change the negative capability verdict.
CP329 closes the nine-program executable cognitive-operator bank; it does not
change the negative capability verdict.
CP330 closes wording-independent structural support classes and exact service
reconstruction; it does not change the negative capability verdict.
CP331 closes causal correlated-support weighting and durable checked-outcome
calibration; it does not change the negative capability verdict.
CP332 closes first-answer/ownership blindness for branch review and makes the
review policy service-verifiable; it does not change the negative capability
verdict.
CP333 closes decoy-balanced verifier admission and contains critic failures;
it does not change the negative capability verdict.
CP334 closes pre-causal critic/generator function separation and installs the
durable shared-blind-spot meter and reliability gate; its live evidence state
is still honestly unpowered and it does not change the negative capability
verdict.
CP335 closes bounded, declared, provenance-preserving branch exchange and
train/live mailbox parity. The tensor role lesion/swap/restoration path proves
that executable labor follows role programs rather than fixed branch indices,
but no powered resident task campaign has yet established an accuracy benefit;
the negative capability verdict is unchanged.
CP336 closes claim-grade structural resource and information accounting. It
also repairs the resident recurrent-GRPO execution-spec artifact omitted during
the CP335 schema migration and makes the independent scoring kernel reconstruct
the new accounting evidence without importing the production grader. This does
not create a capability result; the negative capability verdict is unchanged.
CP337 closes persistent evidence-grounded recurrence. Prompt KV remains shared
and read-only; typed cognitive evidence forms an immutable causal prefix; and
mailbox plus hypothesis slots carry the mutable recurrent state without prose
re-encoding. Service-reconstructed transition commitments prove evidence
invariance and causal hypothesis updates. A source matrix traces resident
weights, Black Hole/RAG and episodic memory, Wikipedia/reference retrieval,
local nonparametric one-shot memory, live organs, and GWT return. Governed
in-episode web/tool production remains open, and the negative capability verdict
is unchanged.
CP338 closes stable shared-loop mechanics and fixed-depth train/live update
parity. Fixed-anchor norm control, finite-state refusal, model-bounded KV
positions, continuity-aware fixed-point diagnostics, contained-divergence
disposition, and service-reconstructed certificates are now causal runtime
contracts. This does not create a capability result; the negative capability
verdict is unchanged.

CP339 closes the learned accept/discard mechanism and its calibrated artifact,
runtime state-selection, accounting, and independent reconstruction contracts.
The fitted-head causal test changes a real tiny-Qwen recurrent trajectory and
downstream logits. No resident-32B head or capability campaign ran; default
execution therefore remains explicit passthrough and the negative capability
verdict is unchanged.

CP340 closes calibrated learned stopping and its measured quality/value
evidence boundary. The controller preserves hard recurrence invariants before
consulting the head; the service reconstructs its exact causal inputs and
decision after transport. Task-disjoint bounded workload tests establish
earlier easy-task stopping without hard-task regression, and a real tiny-Qwen
equal-evidence arm proves reduced recurrent transitions and layer applications.
No resident-32B stop head or powered broad-task campaign ran, so this does not
change the negative frontier verdict.

CP340 validation passes the focused policy/artifact/workload/causality/tamper
boundary 7/7 and the affected adaptive-halting, learned-bridge, attachment,
engine, recurrence, schedule, branch, wiring, update-acceptance,
value-of-computation, and resource suite 256/256 in 89.36 seconds. The final
fixed-snapshot RLC, latent-cortex, recurrence/training, global-workspace, GWT,
and execution-controller ownership gate passes 1307/1307 in 485.19 seconds.
Strict focused Ruff, bytecode compilation, and `git diff --check` pass.

Validation: the focused acceptance-head and causal-transition suite passes 6/6;
the combined acceptance/service reconstruction boundary passes 7/7; and the
affected engine, recurrence, schedule, branch, wiring, and recurrence-native
suite passes 203/203. The first full ownership gate exposed a real evidence-
slot/reasoning-slot shape mismatch in relative-distance extraction. The repair
uses a pooled-vector distance when the two immutable groups have different slot
counts, and the targeted ingress regression then passes 6/6. The captured final
fixed-snapshot RLC, latent-cortex, recurrence/training, global-workspace, GWT,
and execution-controller ownership gate passes 1300/1300 in 776.80 seconds.
Strict focused Ruff, bytecode compilation, and `git diff --check` pass. No
resident 32B capability campaign was run, and the negative frontier verdict is
unchanged.

CP341 closes confidence-bound best-state authority and overthinking reversion.
Verifier observations are branch-local, ordinary scalar scores cannot acquire
state-selection authority, interval overlap preserves the incumbent exactly,
and the final returned state is independently reconstructable from the action
and loop-stability receipts. Fixed-depth experiments remain unchanged. No
resident-32B bounded-verifier campaign or externally rooted verifier evidence
ran, so this does not change the negative frontier verdict.

CP341 validation passes the focused engine, verifier, resource-accounting,
schedule/branch, value-of-computation, and service boundary 194/194 and the
affected adaptive-halting, learned-bridge, attachment, recurrence, escape,
update-acceptance, epistemic-runtime, and wiring gate 298/298. The final
fixed-snapshot RLC, latent-cortex, recurrence/training, global-workspace, GWT,
and execution-controller ownership gate passes 1323/1323 in 912.53 seconds.
Strict focused Ruff, bytecode compilation, and `git diff --check` pass.

CP342 closes the hidden-state neural uncertainty mechanism and its bounded
causal authority. Correctness and entropy now come from an objectively trained,
task-disjoint calibrated neural head over the admitted recurrent state rather
than self-reported confidence. Sparse bins abstain; supported measurements can
select among branches only when an admitted task verifier does not supersede
them. The service reconstructs the exact head estimate and selection evidence.
No resident-32B head or powered broad-task campaign ran, so the negative
frontier verdict is unchanged.

CP342 validation passes 13/13 focused artifact/live tests, three repeated real
tiny-Qwen causal-selection runs, and the affected engine, schedule/branch,
worker-origin, service, resource, update/stop, verified-best,
value-of-computation, and epistemic-runtime gate 254/254 in 137.07 seconds.
The final fixed-snapshot RLC, latent-cortex, recurrence/training,
global-workspace, GWT, and execution-controller ownership gate passes 1338/1338
in 707.61 seconds. Strict focused Ruff, bytecode compilation, and `git diff
--check` pass.

This is total checkpoint record 403. The revised forecast remains 466-733 total
records, now approximately 63-330 records after this checkpoint. Checkpoint-
count completion is approximately 55.0%-86.5%, with a midpoint planning
estimate of 67.2%. Next: publish CP342, then implement SPARK-029's subtle
mistake locator.

CP343 closes the bounded transition-mistake locator. Complete controlled-
mutation traces now train a prior-to-proposal neural head under task-disjoint
in-domain calibration and genuinely domain-disjoint OOD evaluation. Aggregate
and per-domain exact, within-one, no-error, discrimination, and calibration
gates must pass before an artifact is admitted. Rejected proposals remain
visible, while the locator is structurally forbidden from steering repair.
The service independently reconstructs the entire live receipt.

CP343 validation passes the focused artifact, live runtime, real tiny-Qwen, and
service boundary 99/99 and the affected engine, branch, recurrence, update,
uncertainty, resource, worker-origin, stop, verified-best, and wiring gate
250/250 in 133.40 seconds. The final fixed-snapshot RLC, latent-cortex,
recurrence/training,
global-workspace, GWT, and execution-controller ownership gate passes 1351/1351
in 701.66 seconds. Strict focused Ruff, bytecode compilation, and `git diff
--check` pass.
The effect-ownership baseline now records only the four reviewed locator and
neural-uncertainty artifact read/write effects. Repository-wide governance
lint remains non-green at 49 regressions and 13 stale buckets from concurrent
unrelated work; CP343 does not absorb that drift or claim the global gate.

This is total checkpoint record 404. The revised forecast remains 466-733 total
records, now approximately 62-329 records after this checkpoint. Checkpoint-
count completion is approximately 55.1%-86.7%, with a midpoint planning
estimate of 67.4%. Next: publish CP343, then implement SPARK-030's complete-
trace bidirectional reflector. Final multi-hour soaks remain deferred until
every shorter gate is green.

CP344 closes the always-on complete-hidden-trace reflector. Every transition
now contributes bounded prior, proposal, and admitted sketches; a read-only
full-sequence pass compares each step with both past and future admitted
context, the initial latent premise, and final latent conclusion. Rejected
proposals remain visible but outside the admitted path. The service
independently reconstructs the evidence, and no decoded answer or mutation
authority enters the critic.

CP344 validation passes the focused reflector/service boundary 94/94 and the
affected engine, branch, recurrence, update, uncertainty, locator, resource,
escape/telemetry, worker-origin, and wiring gate 257/257 in 133.92 seconds. The final fixed-
snapshot RLC, latent-cortex, recurrence/training, global-workspace, GWT, and
execution-controller ownership gate passes 1358/1358 in 697.14 seconds. Strict focused Ruff,
bytecode compilation, and `git diff --check` pass.

This is total checkpoint record 405. The revised forecast remains 466-733 total
records, now approximately 61-328 records after this checkpoint. Checkpoint-
count completion is approximately 55.3%-86.9%, with a midpoint planning
estimate of 67.6%. Next: publish CP344, then implement SPARK-031's calibrated
contradiction tensor over the reflected trace. Final multi-hour soaks remain
deferred until every shorter gate is green.

CP345 closes SPARK-031's calibrated transition-by-latent-position contradiction
tensor. The reflector now preserves bounded position evidence, and the learned
head compares every cell against local, premise, conclusion, prefix, suffix,
and trajectory context. Cell and step probabilities are calibrated
independently. Admission requires complete controlled-mutation and sham tensors,
disjoint task/trace/evidence identities, genuine OOD domains, middle and
long-context support, aggregate and per-domain localization floors, and
calibration bounds.

The worker and service reconstruct every channel, feature commitment,
probability, aggregate, source binding, and non-authority claim. Exact-SHA,
no-follow artifact loading fails closed. Unavailable mode emits no probability
and performs no hidden scoring work. SPARK-031 grants no state, selection,
repair, or attention authority; SPARK-032 remains the bounded perturbation
milestone.

CP345 validation passes the focused controlled-mutation/tensor/reflector/service
boundary 108/108 and the affected engine, update, locator, uncertainty, escape,
resource, worker-origin, and wiring gate 227/227 in 120.93 seconds. The final
fixed-snapshot RLC, latent-cortex, recurrence/training, global-workspace, GWT,
and execution-controller ownership gate passes 1372/1372 in 715.81 seconds.
Strict focused Ruff, bytecode compilation, and `git diff --check` pass.
The effect-ownership baseline selectively records the contradiction artifact's
reviewed atomic write and stable no-follow descriptor read. Repository-wide
governance lint remains non-green at the same 49 unrelated/concurrent
regressions and 13 stale buckets present before this checkpoint; CP345 does not
absorb that drift or call the global gate green.

This is total checkpoint record 406. The revised forecast remains 466-733 total
records, now approximately 60-327 records after this checkpoint. Checkpoint-
count completion is approximately 55.4%-87.1%, with a midpoint planning
estimate of 67.7%. Next: publish CP345, then implement SPARK-032's bounded,
counterfactually verified contradiction-driven perturbation. Final multi-hour
soaks remain deferred until every shorter gate is green.

CP346 closes SPARK-032's bounded contradiction-driven intervention. An
admitted coordinate may propose a change only to its writable position in the
selected branch's final latent workspace. The guided arm competes against
exact no-op and norm-matched orthogonal-random controls in counterbalanced,
repeated, fixed-length transformer probes. Protected cognitive-context slots
are receipt-bound and immutable.

The contradiction head cannot authorize its own proposal. Retention requires a
decoy-admitted authoritative verifier, stable exact or calibrated-interval
observations, equal actual layer applications, and a guided lower bound above
both control upper bounds by the configured margin. Every unavailable,
scalar-only, unstable, unequal-compute, tied, regressing, failed, or
under-budget path restores the exact baseline. The service independently
reconstructs source, configuration, geometry, protected positions,
observations, compute parity, decision, and rollback.

CP346 validation passes the focused contradiction/perturber/tensor/wiring
boundary 126/126 in 68.19 seconds and the final affected engine, predecessor,
resource, worker-origin, and service boundary 293/293 in 157.95 seconds. The
conservative fixed-snapshot RLC, latent-cortex, recurrence/training,
resident-campaign, global-workspace, GWT, and execution-controller ownership
gate passes 1538/1538 across 96 files in 789.05 seconds. Strict focused Ruff,
bytecode compilation, and `git diff --check` pass. Repository-wide governance
lint remains non-green at the same 49 unrelated/concurrent regressions and 13
stale buckets; CP346 adds no effect-ownership entry and does not absorb that
drift.

This is total checkpoint record 407. The revised forecast remains 466-733 total
records, now approximately 59-326 records after this checkpoint. Checkpoint-
count completion is approximately 55.5%-87.3%, with a midpoint planning
estimate of 67.9%. Next: publish CP346, then implement SPARK-033's locally
conditioned exploration. Final multi-hour soaks remain deferred until every
shorter gate is green.
