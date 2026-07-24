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
  SPARK-023, SPARK-024, SPARK-025, SPARK-026.
- `PARTIAL`: SPARK-003,
  SPARK-027, SPARK-035, SPARK-039, SPARK-040, SPARK-041,
  SPARK-042, SPARK-051, SPARK-052, SPARK-053, SPARK-054, SPARK-055,
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

### Active ownership claims (agent coordination)

- **SPARK-002 and SPARK-003 are CLAIMED and in progress by
  the Fable session (Bryan-directed), starting 2026-07-23 19:15 PT.**
  SPARK-004 resolved in this lane at F1; SPARK-013's pre-training half
  resolved at F2 (2026-07-23) — its remaining semantic acceptance binds to
  the SPARK-069 trained treatment, not to further parallel work. These are
  the four items the sequential march has skipped since CP318; Bryan assigned
  them out-of-band to close the SPARK-069 admission prerequisites (004, 013)
  and the contract legs (002, 003) in parallel with the march. Other agents:
  please do not take these four items; take the next unchecked item outside
  this set. This block will be removed when the claims resolve.

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
- [ ] **SPARK-002 - Primary-literature dossier.** Replace placeholder citations
  with a versioned bibliography of primary papers/specifications, distinguish
  replicated findings from proposals, record licenses, and bind source hashes
  in the final methods package.
- [ ] **SPARK-003 - Failure and threat model.** Enumerate anchoring, verifier
  collusion, fake branch diversity, reward hacking, answer leakage, right-to-
  wrong correction, context contamination, state corruption, budget abuse,
  stale tools, adaptation leakage, and unsafe self-modification with executable
  mitigations.
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
- [ ] **SPARK-036 - Transient negative constraints.** Distill verified failures
  into scoped, expiring constraints, prevent unsupported critic prose from
  becoming a constraint, and prove repeated-error reduction.
- [ ] **SPARK-037 - Causal virtual compute quanta.** Wire the existing bounded
  quanta contract into slot seeds, retrieval directions, verifier probes, or
  fast-weight subspaces; require measured contribution, TTL, budget charge,
  erasure proof, and ablation against no-quanta/matched-compute controls.
- [ ] **SPARK-038 - Latent tree/forest search.** Implement bounded MCTS/beam/BFS
  over verified neural states with UCT/value estimates, backtracking, duplicate
  detection, cancellation, and exact compute accounting.

## E. Verifier mesh and local repair

- [ ] **SPARK-039 - Atomic decomposition.** Convert candidate reasoning into
  typed claims/transitions before grading, verify decomposition coverage, and
  detect omitted dependencies.
- [ ] **SPARK-040 - Deterministic verifier router.** Route eligible claims to
  sandboxed code, calculators, schemas, compilers, SAT/SMT, theorem proving,
  database constraints, planning, simulation, or retrieval with explicit
  unknown/unsupported results and governed tool receipts.
- [ ] **SPARK-041 - Process verifier.** Score local state transitions with a
  calibrated PRM/process model, expose reliability by domain and depth, and
  refuse dense credit where the verifier is not validated.
- [ ] **SPARK-042 - Generative verifier.** Independently derive or falsify
  disputed steps in a fresh context and bind its evidence rather than accepting
  a holistic judge score.
- [ ] **SPARK-043 - Adversarial verifier curriculum.** Co-train subtle error
  insertion and error localization under sandboxed, held-out evaluation; retain
  verified failures as negatives.
- [ ] **SPARK-044 - Counterfactual verifier.** Perturb assumptions and inputs,
  require predicted consequence changes, and penalize fragile claims that remain
  invariant when they should move.
- [ ] **SPARK-045 - Prefix stability verifier.** Regenerate continuations from
  verified prefixes, estimate conclusion stability, and calibrate the signal
  separately from correctness.
- [ ] **SPARK-046 - Correlation-aware verifier fusion.** Track Wilson/confidence
  bounds, domain reliability, dependence between verifiers, and historical
  calibration; no single probabilistic verifier has absolute authority.
- [ ] **SPARK-047 - Disagreement graph.** Localize the earliest dependency where
  branches diverge and identify the exact disputed assumption or transition.
- [ ] **SPARK-048 - Diagnostic action selection.** Choose the cheapest operation
  expected to resolve each disagreement: execute, retrieve, prove, simulate,
  falsify, regenerate from prefix, or ask a specialized verifier.
- [ ] **SPARK-049 - Local invalidation and repair.** Preserve verified ancestors,
  invalidate the failed node and descendants, regenerate from the last valid
  state, and prove unrelated correct work is unchanged.
- [ ] **SPARK-050 - Confidence-bound answer replacement.** Replace an accepted
  answer only when the new lower confidence bound exceeds the old upper bound
  plus a preregistered margin; otherwise retain, qualify, or abstain.

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

  SPARK-051 is not accepted yet: SEARCH_MEMORY and RETRIEVE_EVIDENCE currently
  direct attention over already admitted context rather than initiating a new
  governed fetch; EXECUTE still requires an external orchestration executor;
  several actions need independently distinguishable semantics and sufficient
  checked outcomes to leave bootstrap mode; and resident/live causal ablations
  have not established calibrated value or capability gain.
- [ ] **SPARK-052 - Adaptive breadth/depth/tool routing.** Scale recurrence,
  branch count, lookahead, tools, and verifier effort from difficulty,
  uncertainty, stakes, body pressure, deadlines, and resource admission while
  preserving user-facing work.
- [ ] **SPARK-053 - Principled stop and abstain.** Stop on verified convergence,
  low value of further compute, budget exhaustion, or irreducible uncertainty;
  distinguish each reason in receipts and language generated by Aura herself.
- [ ] **SPARK-054 - Complete causal receipts.** Record state lineage, operators,
  branch isolation, tool evidence, verifier scores, accepted/rejected updates,
  compute, adaptations, stopping, final synthesis, and integrity proofs without
  exposing private chain-of-thought.

## G. Temporary and permanent learning

- [ ] **SPARK-055 - Query-scoped fast-weight learning.** Optimize bounded
  temporary weights from high-confidence evidence, prove identity at attach,
  constrain magnitude/behavior, isolate concurrent requests, and make the
  adapted function causal to the answer.
- [ ] **SPARK-056 - Runtime integrity proof producer.** Measure pre/post fixed
  parameter canaries, adapted-layer identity, exact erase, caches, tokenizer,
  adapters, quantization, and worker identity; make certification consume the
  measurements rather than mutable booleans.
- [ ] **SPARK-057 - Recalibrated test-time trainer.** Implement a TEMPO-style or
  stronger bounded refinement loop with held-out critic recalibration,
  high-confidence pseudo-label admission, drift detection, rollback, and
  matched-compute controls.
- [ ] **SPARK-058 - Verified replay buffer.** Store initial failure, earliest
  causal error, discriminating test, corrected transition, verified solution,
  error class, escape strategy, provenance, and privacy/governance disposition;
  reject unverifiable traces.
- [ ] **SPARK-059 - Structured SFT and tool traces.** Train logical forms,
  programs, proof steps, tool calls, tool-result interpretation, and local
  repair from executable, held-out, contamination-audited data.
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
  causes rather than suppress warnings or emit canned fallback prose.

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
