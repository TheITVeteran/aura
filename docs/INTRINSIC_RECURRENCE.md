# Intrinsic recurrence

Status: Guide · Reviewed against the tree 2026-08-13

The live front of the [Recursive Latent Cortex](RECURSIVE_LATENT_CORTEX.md)
programme. This page covers the pivot from *talking to* a recurrent workspace
to *being* recurrent, and the training work that follows from it.

Read [RECURSIVE_LATENT_CORTEX.md](RECURSIVE_LATENT_CORTEX.md) first if you
haven't — this page assumes the frozen-loop result.

---

## The finding that caused the pivot

The RLC as originally built was a model that **talked to** a recurrent
workspace. Reading its own `_persist_and_score`: the answer tokens traverse
`layers[prelude:coda]` exactly once, at every depth setting. Only the four
slot positions were ever recurred.

So the computation producing the answer always received the base checkpoint's
64 layers — identical to vanilla. "Depth" changed nothing except what got
written into a scratchpad the answer attends to.

The measurement follows from the architecture, and should have been predicted
from it:

```
RLC depth 1 / 2 / 4 / 8 : 25 / 29 / 25 / 25 %   (8× compute, flat)
vanilla greedy          : 21 %
```

Slots are causal and worth a few points as a prior. **Depth was worth nothing
because no depth was ever applied to the answer's own computation.** That is
the whole explanation for the frozen-loop negative result, and it is an
architectural fact rather than a tuning failure.

`core/learning/intrinsic_recurrence.py` (CP226) applies depth where it
actually matters. The real token stream re-enters the middle block `T` times:

```python
h = layers[:prelude](h)                     # prelude, once
for t in range(T):
    h = layers[prelude:coda](h)             # the loop
h = layers[coda:](h)                        # coda, once
```

Effective depth becomes `prelude + T*(coda-prelude) + (L-coda)`. A 64-layer
checkpoint runs **160 layers deep at T=4, with the same weights**. This is the
Ouro / LoopLM architecture, retrofitted onto a checkpoint that was not
pretrained for it — which is the only reason the stabilizers exist at all.

### The load-bearing safety property

> At `T=1` this is bit-identical to the base forward pass.

Recurrence is added by increasing `T` from a known-good starting point, never
by a cutover. Every campaign carries a `T=1` no-recurrence anchor arm for
exactly this reason, and a preflight that finds the anchor missing blocks the
launch before any model loads.

A checkpoint pretrained without recurrence has no reason for its middle block
to be a stable map — iterating it can drift in norm until the coda receives
activations outside anything it was trained on. `anchor_injection` and
`renormalize` exist for that failure mode, and **both default to OFF** so the
plain loop is what gets measured first.

---

## Teaching process instead of output

The first training attempt was answer-only SFT. It taught the model to *stop
reasoning*: recurrence itself became the damage. Output-only transfer is now
rejected quickly and by contract (CP393).

What replaced it is a typed program the recurrence executes and is supervised
on. The controller emits, per recurrent step, a structured action against a
canonical schema (`core/learning/recurrent_action_schema.py`,
`aura.recurrent_action_target.v2`):

| Slot | Meaning |
|---|---|
| `opcode` | which operation this step performs |
| `arg0`–`arg5` | typed operands |
| `terminal` | whether the program halts here |

The narrow opcode vocabulary that was proven first covers exact machine
semantics — `copy value`, `add/multiply/subtract modulo`, `boolean
not/and/or/xor`, `register affine`. CP394 extended it with seven broader
process meanings (`frontier traverse / enumerate / simulate / infer /
schedule / calibrate / audit`), values 9–15.

Because the targets are typed and the operations are exactly checkable, the
supervision signal is a verifier rather than a preference model. There is
nothing to game: a step either computed the right typed state or it didn't.

Alongside the action schema sit a state schema
(`recurrent_state_schema.py`), literal grounding bound to real tokenizer digit
token ids (`recurrent_literal_grounding.py`), an opcode grounding contract
(`recurrent_opcode_grounding.py`), and an answer-emission contract
(`recurrent_answer_emission.py`). `unified_intrinsic_recurrence.py` runs depth,
memory, correction, and halting on **one** resident-transformer trajectory —
additive and identity-initialized, so one iteration remains the base forward
until learned controller parameters are admitted.

---

## What has been established

Working on a 1.5B vehicle so the resident 32B stays live. Every item below is
bounded to what it measured; none of them authorize a broad claim.

**Trained recurrence is not inert.** Package-depth trained parameters beat
their exact initialization control 3/7 to 2/7 (CP368). Small, but measured
against the right control — the same architecture with the same decode
budget, differing only in whether the parameters were trained.

**The serving policy was the bug, not the tissue.** CP368's campaign failed
not because the controller was useless but because fixed depth four discarded
a correct depth-one answer and produced no new success over ordinary decode.
Decode now produces two separately attested candidates from the same trained
controller — depth one and the package-qualified depth — and public exact
verification considers both. A deeper decode can *add* a success that depth
one missed; it can never *erase* a shallow answer already proven correct
(CP376). Source labels for the shallow arms are rejected unless the resource
receipt records recurrence depth one.

**Typed recurrent serving authority is durable, and narrow.** The resident 32B
holds `qualified_typed_only` authority for the `khop`, `modular`, and
`register_trace` families at task depths 1, 2, 4 with recurrence depth four
(CP357). Two independent cold loads each decoded 9/9 typed cases exactly; the
durable pointer reopened, rollback completed, and both canary authorities
expired after their requests without exposing token outputs.
`ordinary_chat_authorized=false` and `arbitrary_reasoning_authorized=false`
are the standing boundary.

**Proven tissue can be extended without ambiguous inheritance.** CP396
recovered the exact CP232 parent immutably. CP399 built a certified
append-only semantic migration for the seven new opcode meanings. CP400 then
corrected CP399's own rule: the state, action, and literal codebooks are
*learned tissue*, and requiring equality against a fresh initialization would
have discarded the exact parent CP396 had just recovered. The migration now
starts from all 51 exact parent tensors and overwrites only
`controller.action_value_embeddings[opcode, 9:16]`. Everything else is
verified byte-exact, and independent evaluation recomputes the merge and
checks the reconstructed step-zero controller hash.

**Small-checkpoint falsification ran before resident expense.** The retained
five-arm run passed heldout likelihood transfer against base and every
equal-work control, and passed all teacher-forced regression families. The
generated behavior gate **failed** at 1/12 trained canaries, including three
zero-token answers. That completes the experiment requirement; it does not
authorize resident training.

## What is not established

Stated plainly, because the ledger states it at every checkpoint:

- No broad behavioral gain.
- No resident-32B reasoning improvement.
- No static fusion.
- No frontier performance, and no `WOW Signal`.

The next bounded milestone is the 1.5B adaptation over train depths
`1,3,4,5,6,8,10` with held-out `12,16`, followed immediately by the
already-frozen four-arm behavioral canary.

**This page is a snapshot; the ledger is the record.** It was reviewed at
CP409, and checkpoints land faster than a narrative page can track. Everything
above is *mechanism and status*, which changes slowly. For what happened most
recently, read the tail of
[RLC_SPARK_EXECUTION_LEDGER.md](RLC_SPARK_EXECUTION_LEDGER.md) and
[AURA_EXECUTION_TRACKER.md](AURA_EXECUTION_TRACKER.md) — both append-only.
What will *not* have changed silently is the "not established" list: every
checkpoint in this programme restates it explicitly, so if a broad gain is ever
claimed it will be claimed in a named checkpoint rather than drifting into
being true.

---

## The code

| Path | What it is |
|---|---|
| `core/learning/intrinsic_recurrence.py` | The loop itself, `RecurrentDepthPlan`, the `T=1` identity property |
| `core/learning/unified_intrinsic_recurrence.py` | Depth + memory + correction + halting on one trajectory |
| `core/learning/unified_intrinsic_objective.py` | Structured state/action/initial-state losses and accuracy breakdowns |
| `core/learning/recurrent_action_schema.py` | Typed action targets, the opcode vocabulary |
| `core/learning/recurrent_state_schema.py` | Typed state targets from an execution trace |
| `core/learning/recurrent_opcode_grounding.py` · `recurrent_literal_grounding.py` | Tokenizer-bound grounding contracts |
| `core/learning/recurrence_checkpoint_migration.py` | The certified append-only codebook migration |
| `core/learning/recurrent_grpo.py` | The GRPO training path |
| `core/learning/recurrent_sft_falsification.py` | The small-checkpoint falsification battery |
| `tools/train_unified_intrinsic_recurrence.py` | The trainer entry point |
| `tools/run_unified_recurrent_broad_canary.py` | The frozen behavioral canary |
| `tools/run_unified_recurrent_shadow_lifecycle.py` | Cold-load shadow lifecycle and rollback |

Evidence is frozen under `artifacts/closeout/latent_cortex/`. The
checkpoint-by-checkpoint narrative is in
[AURA_EXECUTION_TRACKER.md](AURA_EXECUTION_TRACKER.md) — append-only, read the
tail as current.

**One training protocol at a time.** It is memory-hungry, and launching a
second beside the resident 32B will take the host down.
