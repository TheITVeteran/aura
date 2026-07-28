"""The latent cortex is an enhancement. It must never cost the person the turn.

Measured live on the desktop path the moment the latent lane started running at
all (its budget contract had been rejecting every foreground turn, so it had
been silently declining and an ordinary generation had always run):

    Recursive Latent Cortex exhausted the single resident owner
    (receipt_contract_failed:terminal_disposition_unproven,
     answer_replacement_unproven,fast_weight_learning_receipt_unproven);
    refusing a late ordinary generation. stage=complete input_tokens=1699
    progress={'elapsed_s': 59.868063, 'stage': 'complete'}

...and the person received "I couldn't get to an answer I'd stand behind on that
one." Every turn died that way.

The episode ran to its TERMINAL stage and released the resident model. What
failed afterwards was its receipt contract — it could not prove what it had
done. The owner was free the whole time.
"""

from __future__ import annotations

from core.phases.response_generation import ResponseGenerationPhase

_exhausted = ResponseGenerationPhase._latent_owner_exhausted

LIVE_REASON = (
    "receipt_contract_failed:terminal_disposition_unproven,"
    "answer_replacement_unproven,fast_weight_learning_receipt_unproven"
)
LIVE_RECEIPT = {
    "episode_id": "4cf6963cc4164ec59fc45d3062a8fcb7",
    "last_stage": "complete",
    "input_token_count": 1699,
}


def test_a_proof_failure_after_a_clean_completion_falls_back():
    assert _exhausted(LIVE_REASON, LIVE_RECEIPT) is False, (
        "a completed episode that merely could not PROVE itself left the model "
        "owner free; refusing the fallback costs the person their answer"
    )
    for stage in ("completed", "finished", "COMPLETE"):
        assert _exhausted(LIVE_REASON, {**LIVE_RECEIPT, "last_stage": stage}) is False


def test_a_genuinely_consumed_owner_still_refuses():
    """The honest refusals must survive: these really did eat the model slot."""
    for reason in (
        "latent_timeout:90s",
        "latent_integrity:state_mismatch",
        "worker_identity_failed:swapped",
        "runtime_identity_deadline_exhausted",
        "runtime_identity_unbound",
    ):
        assert _exhausted(reason, LIVE_RECEIPT) is True, reason

    # A receipt failure that died PART WAY through did consume the owner.
    assert _exhausted(
        LIVE_REASON,
        {"episode_id": "x", "last_stage": "decode", "input_token_count": 10},
    ) is True


def test_no_episode_at_all_falls_back():
    """The pre-existing decline path — nothing ran, so nothing was consumed."""
    assert _exhausted("latent_declined", {}) is False
    assert _exhausted("runtime_operation_authority_mismatch", {}) is False
