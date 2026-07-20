# CP226 — Making the 32B itself recurrent: measured, and it fails untrained

## What was wrong before

`_persist_and_score` passed the ANSWER tokens through `layers[16:48]`
exactly once at every depth setting; `_advance_recurrent_states` recurred
only the four slots. The computation producing the answer always received
the base checkpoint's 64 layers — identical to vanilla. "Depth" only
changed what got written into a scratchpad the answer attended to.

So `25 / 29 / 25 / 25` across an 8x compute range was not a disappointing
result. It was the only result that architecture could produce.

## What was built

`core/learning/intrinsic_recurrence.py`: the real token stream re-enters
the middle block T times. Effective depth = prelude + T*window + coda.
T=1 is the unmodified forward pass by construction.

## What it measured (32B, fused resident checkpoint, n=24)

    identity_check: t1_vs_vanilla_gap = 0.0

    T=1    64 layers   reasoning 12%   answered 79%   (vanilla: 12% / 79%)
    T=2    96 layers   reasoning  8%   answered 71%   final_delta 0.55
    T=4   160 layers   reasoning  0%   answered  4%   final_delta 0.50
    T=8   288 layers   reasoning  0%   answered  0%   final_delta 0.32

    0 diverged. Every state finite. The loop kept moving.

**Untrained retrofit recurrence destroys this checkpoint.** Not by
numerical blowup — by output collapse. By T=4 the model answers 4% of the
time; by T=8 it emits nothing parseable. (T=8 ran in 80.8s vs T=4's 491.8s
because it generated nothing.)

Most likely mechanism: `layers[48:64]` has never seen a state that passed
through `layers[16:48]` eight times. That input is far out of distribution
and the coda's output distribution collapses.

## What this does and does not establish

DOES refute: "make a frozen checkpoint recurrent and it gets deeper for
free." Depth cannot be retrofitted without training.

DOES NOT refute: recurrence-native models. Ouro/LoopLM work — but they are
PRETRAINED to iterate. That now looks like the load-bearing difference
rather than an incidental one.

ESTABLISHES: a trustworthy instrument. `t1_vs_vanilla_gap = 0.0` on the
live 32B proves the recurrent forward IS the base model at T=1, so every
other cell is readable. The previous architecture could not even ask this
question.

## Confound owned

Vanilla scored 12%/79% here vs CP223's 21%/100% because this probe used
bridge `"\n\nFINAL_ANSWER: "` while CP223 used `assistant_answer`. This
cue elicits answers WORSE. The within-run comparison is unaffected —
vanilla and T=1 share the bridge and agree exactly — but absolute numbers
are not cross-comparable with CP223 and should not be quoted as such.

## The obstacle training must clear

1.5B sweep, float32: `cos(pass 1, pass 2) = 0.9994`. The state moves ~42%
in magnitude while barely rotating. Re-running the same window adds another
increment along nearly the same ray, because the residual stream is
dominated by accumulated magnitude. Per-iteration weight deltas
(depth-conditioned LoRA, now on the same clock) make each pass a different
function — that is the mechanism that has to turn repetition into
deepening, and it is now measurable either way.
