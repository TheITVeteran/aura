# CP511 RLC review applicability matrix

Status: implementation ledger, not a capability verdict.

## Scope

Four independent read-only passes reviewed every supplied CP498-CP510 advisory
attachment. The passes covered 32 files and 538,482 bytes of advice from
Perplexity, Grok, DeepSeek, Gemini, Copilot, Claude, Kimi, ChatGPT, Meta, and
Antigravity. Recommendations are consolidated below because many files repeat
the same proposal with incompatible measurements or different names.

Evidence was first compared against immutable CP510 (`6224cd19c`). CP511 then
implemented the applicable state-schema findings. Forecasts, simulated charts,
and uncited measurements are not treated as observations.

## Findings that changed the machine

| Finding | CP510 status | CP511 action | Claim boundary |
|---|---|---|---|
| The five-slot local state was not Markov-sufficient for calibration, coding, or mathematics | Confirmed by bounded collision audit | Added a versioned 11-slot semantic schema; calibration retains numerator/denominator context; coding retains case identity and four signed balances | Local recurrent admission is separate from public-prefix replay admission |
| Premise tie-breaking depended on identity absent from the public action/state | Confirmed by source trace | Added public row/name ranks and retained incumbent identity for exact equal-score behavior | Premise closure must pass the regenerated audit |
| Mathematics needs a sufficient statistic whose size grows with input | Confirmed; fixed-width widening is not a proof | Mathematics is excluded from the semantic-state canary | Requires addressable work memory or a recompiled bounded algorithm before admission |
| Public-prefix replay can recover state without using recurrent state | Confirmed | Replay remains a separately named, disabled canary arm | Replay success is not recurrent-state success |
| Saved controller geometry was implicit | Confirmed | State schema, width, register order, grounding, process-tape width, loss weights, checkpoint loading, evaluation, and receipts are frozen in campaign identity | Legacy five-slot checkpoints stay loadable; incompatible bootstrap fails closed |
| Frozen evaluator dropped useful diagnostics | Confirmed | Added final-active-state, recovery, sustained-recovery, terminal-correctness, terminal-self-stability, and per-register aggregates | Calibration metrics remain open below |

## Recommendation ledger

| Recommendation | Status after CP511 | Action or reason |
|---|---|---|
| Closed-loop rollout from exact public initial state with no gold state after step zero | Implemented | Preserve recursive-state and counterfactual-gold tests |
| Train the same hard state commitment used at runtime | Implemented | STE remains deployment-matched |
| Scheduled sampling as the primary repair | Refuted | The direct objective is already closed-loop; prior matched evidence did not improve |
| Replace STE with Gumbel-Softmax or soft rollout now | Refuted | Estimator lesions did not identify BPTT as the blocker; revisit only after schema-qualified T1 competence |
| Remove post-terminal padding from execution loss | Implemented | Terminal stutter remains a separate structural diagnostic |
| Opcode-conditioned hidden/output experts | Implemented, bounded | Keep; do not expand until a matched semantic-schema run isolates residual specialization error |
| PCGrad as the primary repair | Refuted | CP509 measured near-zero conflict and PCGrad regressed; mean is the canary default |
| Markov-identifiability certification | Implemented for bounded cohorts | Admissions now distinguish local recurrent closure from prefix replay closure |
| Blindly increase register count to 7, 8, or 16 | Refuted | Semantic sufficiency, not width alone, determines admission |
| Minimal semantic state for calibration | Implemented | Numerator and denominator remain available through later transitions |
| Minimal semantic state for coding | Implemented for declared generator bound | Case identity plus four signed balance pairs are retained |
| Minimal semantic state for premise auditing | Implemented | Row/name identity and tie behavior are public and local |
| Fixed-register mathematics repair | Refuted | Current DP statistic is not fixed-width representable across growing inputs |
| Addressable recurrent work memory for mathematics | Open, required | Build bounded read/write memory, compiler, masks, and lesions before restoring mathematics |
| Ordered public-prefix recovery | Implemented but untrained | Keep as external-history recovery; evaluate disabled/forced/lesioned separately |
| Recovery from corrupted state | Partial | One-step target is computed from the corrupted state; coherent off-manifold continuation, empirical error harvesting, recoverability classes, and invalid-state latch remain |
| Primitive coverage curriculum | Open, required | Certify opcode, operand, boundary, composition, and length coverage; unique prompts are insufficient |
| Family/horizon/register-balanced optimization | Partial | Balanced family batches and weakest-register term exist; canary enables them and must report macro groups |
| State/history/tape lesions | Partial | Existing arms need topology-matched shuffle, wrong-prefix, processor, replay-only, constant-state, identity-copy, and complete-machine nulls |
| Matched-capacity control | Open, required | Train an initialization-matched, equal-budget control; post-hoc expert averaging is not sufficient |
| Per-task gain and regression sets | Open, required before causal claim | Emit converted, preserved, regressed, and lesion-erased task identities |
| Recovery, terminal, and per-register aggregates | Implemented | Persist by family and horizon in the frozen evaluator |
| ECE, Brier, and reliability by register/depth | Open | Freeze bins/minimum support, preserve probabilities, and add to evaluator |
| Trace-perturbation no-oracle proof | Open | Add static target-flow guard and randomized nonzero train/runtime parity |
| Move exact microcode behind label-authority boundary | Partial | Candidate runtime still contains exact transition helpers; add import/dataflow sentinel before capability adjudication |
| Hash-chained independently replayable process certificate | Open | Define canonical events and a separate interpreter after the semantic tape schema is frozen |
| Execute process once and cache it for answer tokens | Open | Add immutable `ProcessExecution`; prove logits equivalent before resident latency claims |
| Answer-sufficient bridge | Open | Final semantic state is not sufficient for every family; bridge from declared state plus intact process memory and test gold/shuffle/lesion arms |
| Sealed train/development/admission/replication cohorts | Partial | Split cohorts and draw replication seeds only after checkpoint commitment |
| Paired statistics and power | Partial | Exact paired tests exist; sample size must be frozen from pilot discordance, not generic proportions |
| Equal-compute ordinary-decode baseline | Open for the repaired mechanism | Freeze generation, verifier, latency, and parameter budgets before answer-level comparison |
| Direct 1.5B-to-32B tensor copy | Refuted | Transfer contracts and retrain width-specific tissue; do not pretend tensor geometry is portable |
| Resident-32B transfer and runtime shadow | Deferred | Requires replicated 1.5B transition and answer gains with causal lesions and no regressions |
| WOW/frontier claim | Not established | Requires powered, fresh, multi-domain, equal-compute resident evidence and independent verification |

## Advice not adopted as fact

- Accuracy projections, timelines, dollar costs, gradient SNR, mutual
  information, contraction constants, attention entropy, memory estimates, and
  simulated improvement charts without executable artifacts remain hypotheses.
- `A16 > A1` is not a universal recurrence requirement. The relevant measures
  are trajectory exactness, retention after a correct predecessor, recovery
  after error, and improvement over matched controls.
- Prefix collision-freedom on a finite generated cohort does not prove a
  universal Markov state.
- Stability without correctness is not success. CP509's terminal self-stability
  and terminal-correctness diverged sharply.
- A public action compiler or exact verifier may supervise training and score
  evidence; it may not remain inside the measured candidate path.

## Ordered execution

1. Seal CP511 semantic schema and bounded identifiability evidence.
2. Run the fresh three-family semantic-state canary with replay disabled.
3. Add primitive coverage and coherent off-manifold recovery before a larger run.
4. Build addressable work memory and recompile mathematics.
5. Add matched controls, complete lesions, calibration metrics, and gain sets.
6. Prove the structured state/process result reaches free-decoded answers.
7. Replicate on sealed fresh 1.5B cohorts.
8. Retrain width-correct resident-32B tissue, shadow it, and independently verify.
9. Reserve `WOW Signal` for the powered resident result, never for a canary.

## CP513-CP514 addendum

CP513 implemented the primitive-coverage admission required by ordered step 3.
The admission freezes task identity disjointness, opcode and operand support,
state and action masks, depth support, and zero exact-program overlap. It is a
development cohort gate. Because it inspects holdout structure, it is not an
untouched replication verdict and cannot support a WOW claim.

The r9 semantic canary reached step 208 under the older objective. Its step-192
evaluation had perfect action parsing but zero T1 state-value exactness, zero
active-trajectory exactness at every depth, and a best heldout relative gain of
-0.1525. The checkpoint and signed stop receipts remain preserved under
`/Users/bryan/.aura/training-campaigns/cp512-semantic-transition-canary-r9-1p5b-20260815`.
No positive capability claim follows from that run.

CP514 found that the older closed-loop objective used the student's committed
state as input but retained the reference trajectory's next state as the label
after the student left that trajectory. That made the supervised relation
path-dependent rather than one stationary transition law. CP514 changes every
training transition target to the exact next state of the state actually
committed by the student, uses the reference trace after initialization only as
a consistency check, injects coherent alternate trace states for recovery, and
makes the explicit invalid state absorbing. It also reports local-transition
accuracy separately from nominal verified-trace-position agreement. The r9
checkpoint must not be resumed or migrated into this changed objective.

Ordered step 3 is now implemented at the contract level. Its empirical closeout
requires a fresh source-bound canary from a new initialization. Mathematics and
addressable work memory remain ordered step 4.

## CP515 addendum

CP515 begins ordered step 4 by replacing the disproven fixed-register
mathematics representation with an exact bounded sparse work memory. Its
address is `(selected_count, last_value, total_sum)` and each cell retains an
exact multiplicity plus a canonical witness. The declared registry bound of
ten input values and selections through width four yields a derived maximum of
386 live addresses; overflow is an explicit refusal rather than silent
truncation. A configuration no-op aligns the memory trace with the canonical
public action program, after which one stationary Markov update consumes each
sorted public value.

The compiler reads only public objective literals. An independent brute-force
oracle verifies all registered difficulties in tests, while the verifier answer
is used only as a compilation consistency check. Private addresses,
multiplicities, and witnesses survive a create-once dataset freeze through the
new v2 source schema; v1 artifacts remain loadable without memory supervision.
Public receipts expose bounded shape and cryptographic commitments but not the
private cells.

This proves a sufficient, serializable supervision state for the bounded
mathematics task. It does **not** prove that learned recurrent tissue can write,
read, or execute that memory autonomously. Mathematics remains excluded from a
capability claim until learned read/write heads, occupied-cell masks, matched
lesions, and teacher-free decoded evaluation are implemented and pass.

## CP516-CP518 addendum

CP516 implements a teacher-removed bounded mathematics memory machine. A
generic sparse address bus retains and merges cells, while neural tissue learns
the write predicate and final result-admission predicate. Runtime starts from an
empty memory and rolls in only its own hard decisions. It has no compiler,
verifier, expected answer, or private trace. Incorrect writes remain observable
rather than being blocked by the training-time semantic validator. No-write,
always-write, no-read, rotated-routing, and reset-each-step lesions are explicit
execution modes; a fresh seed-identical tissue is the matched-capacity control.

CP517 made the frozen canary directly executable. CP518 records its create-once
certificate at `cp517_mathematics_memory_canary.json`. The measured source was
clean commit `3ef63788a271efae45e35d8b2d1f3a2c74c02d4e`. Training used 120 tasks;
evaluation used 300 fresh tasks across all three registered difficulties.
Teacher-removed treatment was 300/300 exact. The initialization-matched control,
no-write, no-read, and reset-memory arms were each 0/300; rotated routing was
2/300 and always-write was 17/300. The outer receipt hash and all arm receipts
validate.

This is positive causal evidence for bounded neural predicate acquisition plus
student-rolled addressable memory. It closes the fixed-register mathematics
representation blocker. It is still not a free-decoded language result, a
resident-32B result, a multi-domain general reasoning result, or WOW. The next
gates are durable tissue packaging, independent reload/replay, process-to-answer
emission, matched ordinary-decode comparison, and fresh replication.
