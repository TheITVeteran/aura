"""The floor: the stack must never answer worse than ordinary decode.

Bryan's contract, and it should have been the design contract from the
start: by default the system does as well as vanilla; no improvement means
neutral; improvement means gain; it must NEVER return a lower-quality answer.

That is only structurally true if, when nothing is promoted, the public
answer is the answer ordinary decode would have produced. Every knob on
which the incumbent decode differs from the control is a way the floor can
be breached, so each one is enumerated here rather than discovered later by
losing a battery to it.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
TOOLS = REPO_ROOT / "tools"
for path in (str(REPO_ROOT), str(TOOLS)):
    if path not in sys.path:
        sys.path.insert(0, path)

import run_rlc_reconciliation_sweep as sweep  # noqa: E402


# What `_run_vanilla` actually does: greedy, no penalty, stop on the first
# complete FINAL_ANSWER. Anything the product arm does differently is a way
# its "neutral" path stops being neutral.
_CONTROL_DECODE = {
    "decode_temperature": 0.0,
    "decode_top_p": 1.0,
    "decode_repetition_penalty": 1.0,
    "decode_max_tokens": 512,
    "decode_contract": "final_answer_v1",
    "decode_min_tokens": 0,
}


def test_the_product_arm_decodes_like_the_control():
    """Divergences found the expensive way: a 1.25 repetition penalty the
    deployed system does not use, which fights step-by-step arithmetic because
    that reasoning repeats phrasing and digits by construction."""
    full = sweep._build_config(8, 16, "applied", 512, profile="full")
    mismatched = {
        knob: (expected, getattr(full, knob))
        for knob, expected in _CONTROL_DECODE.items()
        if getattr(full, knob) != expected
    }
    assert not mismatched, (
        "the product arm's decode differs from ordinary decode, so its "
        f"neutral path is not neutral: {mismatched}"
    )


def test_ordinary_decode_owns_the_answer_until_something_beats_it():
    """The structural floor. Recurrence, branch selection, verification and
    learning all still run and stay receipted -- they simply do not own the
    output until a gain gate promotes them."""
    full = sweep._build_config(8, 16, "applied", 512, profile="full")
    assert full.decode_incumbent_policy == "vanilla_incumbent"
    assert full.answer_replacement_enabled is True


def test_the_ablation_is_allowed_to_break_the_floor():
    """The mechanism arm deliberately lets the latent path own the answer: a
    degraded episode must not silently serve a vanilla answer and be read as a
    recurrent result. It is an instrument, not the product, and it is the only
    arm permitted to score below ordinary decode."""
    mech = sweep._build_config(4, 16, "suppressed", 512, profile="mechanism")
    assert mech.decode_incumbent_policy == "latent"


def test_every_arm_declares_which_side_of_the_floor_it_is_on():
    """No arm may be ambiguous about whether it is bounded below by vanilla."""
    bounded = {"full", "full_oracle"}
    unbounded = {"mechanism", "ordinary", "ordinary_best_of_3"}
    for arm in sweep.ARMS:
        assert arm.profile in bounded | unbounded, arm


def test_the_last_divergence_is_crossed_rather_than_assumed():
    """Bridge tokens -- including the 27-token terminal disposition -- are
    prepended to the answer decode even under vanilla_incumbent, so the
    product's honest floor is "vanilla plus disposition", not "vanilla". The
    deployed system does this, so the product arm keeps it; the matched arm
    with it suppressed measures what it costs. Discovering that after a
    negative result would cost the whole battery a second time."""
    by_name = {a.name: a for a in sweep.ARMS}
    product = by_name["full_stack"]
    matched = by_name["full_stack_nodisp"]
    assert product.profile == matched.profile == "full"
    assert product.steps == matched.steps
    assert product.max_tokens == matched.max_tokens
    # The disposition is the ONLY thing that differs between them.
    assert product.policy == "applied"
    assert matched.policy == "suppressed"


def test_the_floor_and_the_gain_are_not_mutually_exclusive():
    """The deadlock that made a win impossible.

    decode_incumbent_policy governs WHO owns the answer; answer_replacement
    governs WHETHER a better candidate may take it. Wiring the second to the
    first meant:

      latent            -> may promote, but no floor at all
      vanilla_incumbent -> floor holds, promotion force-disabled

    so no configuration could both stay at or above ordinary decode AND
    improve on it. The product arm must be the combination the engine's own
    incumbent comment promises: ordinary decode owns the answer until an
    independent gain gate promotes something better.
    """
    full = sweep._build_config(8, 16, "applied", 512, profile="full")
    assert full.decode_incumbent_policy == "vanilla_incumbent"
    assert full.answer_replacement_enabled is True

    import inspect

    from core.brain.llm.latent_cortex import engine as engine_mod

    src = inspect.getsource(engine_mod)
    assert "enabled=self.config.answer_replacement_enabled," in src, (
        "the promotion gate must not be re-coupled to decode_incumbent_policy"
    )
    assert (
        "self.config.answer_replacement_enabled\n                        and latent_decode_authorized"
        not in src
    )
