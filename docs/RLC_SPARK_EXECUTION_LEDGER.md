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

## Code-grounded baseline and ownership

The four-slice static audit covered the neural core, epistemic/verifier paths,
training/proof surfaces, and the selected live/full-mind path. It read the
entire Spark source and traced callers rather than crediting class names. The
baseline below is exhaustive over SPARK-001 through SPARK-072:

- `ACCEPTED`: SPARK-001, SPARK-005, SPARK-006, SPARK-007, SPARK-008, SPARK-009,
  SPARK-010, SPARK-011, SPARK-012, SPARK-014, SPARK-015, SPARK-016,
  SPARK-017, SPARK-018, SPARK-019, SPARK-020, SPARK-021.
- `PARTIAL`: SPARK-003,
  SPARK-022, SPARK-023, SPARK-024, SPARK-025,
  SPARK-026, SPARK-027, SPARK-035, SPARK-039, SPARK-040, SPARK-041,
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
- [ ] **SPARK-004 - Frozen baseline bundle.** Freeze resident checkpoint,
  tokenizer, adapters, decoding, task generators, control manifests, resource
  envelope, randomization, and current vanilla/RLC measurements before changing
  the treatment.

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
- [ ] **SPARK-022 - Equal-compute virtual width.** Meter every branch, exchange,
  verifier, and decode operation in the same currency as controls; refuse
  comparisons with hidden compute or information advantages.

## D. Penultimate recurrent neural correction

- [ ] **SPARK-023 - Evidence-grounded recurrent state.** Keep immutable evidence
  available at every recurrent step and update a persistent latent hypothesis
  rather than re-encoding the prior prose.
- [ ] **SPARK-024 - Stable shared looped core.** Run shared middle-layer
  computation with anchored norm control, residual scaling, finite checks,
  bounded KV use, fixed-point diagnostics, and train/inference parity.
- [ ] **SPARK-025 - Learned accept/discard gate.** Score each latent update
  against evidence/process quality and retain the previous state when the
  proposal is not credibly better; prove the gate changes outcomes.
- [ ] **SPARK-026 - Learned stop/convergence gate.** Combine fixed-point
  residuals, calibrated quality, uncertainty, and expected value of additional
  compute; prove easy tasks halt earlier without harming hard-task accuracy.
- [ ] **SPARK-027 - Best-state and overthinking reversion.** Preserve the best
  verified state, detect oscillation/divergence, and prevent later recurrence
  from replacing a correct state without confidence-bound evidence.
- [ ] **SPARK-028 - Neural uncertainty head.** Train and calibrate a hidden-state
  correctness/entropy head at claim or step granularity; do not substitute
  self-reported confidence.
- [ ] **SPARK-029 - Mistake locator.** Train a span/transition locator on
  in-domain and out-of-domain subtle errors; evaluate location accuracy before
  using it to steer repair.
- [ ] **SPARK-030 - Bidirectional hidden-state reflector.** Inspect the complete
  hidden trace with a non-causal critic that can compare premises and
  conclusions without reading only the final answer.
- [ ] **SPARK-031 - Contradiction tensor.** Produce calibrated token/step-level
  contradiction evidence, train on controlled mutations, and prove localization
  on middle-of-trace and long-context failures.
- [ ] **SPARK-032 - Attention/KV perturber.** Translate localized contradiction
  evidence into bounded, receipt-bearing changes to attention geometry or
  latent state at affected positions; include matched-random and no-op controls.
- [ ] **SPARK-033 - Locally conditioned exploration.** Increase stochastic
  exploration only in uncertain/contradictory regions while preserving stable
  regions; meter entropy, diversity, regressions, and determinism controls.
- [ ] **SPARK-034 - Heterogeneous integration.** Combine old/new candidate
  distributions using calibrated evidence, not a free learned preference;
  compare selection, logit fusion, and abstention under equal compute.
- [ ] **SPARK-035 - KV state tree and rewind.** Snapshot at verified boundaries,
  remove rejected KV slices, restore exact prior state, and prove rejected
  reasoning cannot leak into regenerated branches.
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

Validation: the affected branch, schedule, engine, worker/service, experiment,
recurrence-native v2/v3/v4, recurrent-GRPO, trainer-contract, adapter-identity,
structural-diversity, correlated-support, and operator suite passes 312/312 in
111.25 seconds. The final fixed-snapshot RLC, latent-cortex, global-workspace,
GWT, and execution-controller ownership gate passes 900/900 in 319.88 seconds.
Strict focused Ruff, bytecode compilation, and `git diff --check` pass. No
resident 32B capability campaign was run, and the negative frontier verdict is
unchanged.

This is total checkpoint record 396. The revised forecast remains 466-733 total
records, now approximately 70-337 records after this checkpoint. Checkpoint-
count completion is approximately 54.0%-85.0%, with a midpoint planning
estimate of 66.1%. Next: publish CP335, then implement complete same-currency
compute and information accounting for SPARK-022.
