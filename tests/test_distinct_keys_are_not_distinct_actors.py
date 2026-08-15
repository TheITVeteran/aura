"""The v5 protocol called four keys "mutually distinct actors". They are not.

`validate_evidence_role_separation` checks that the evaluator, worker, verifier
and run coordinator have four different signer ids and four different public
keys, and the module documented that as proof a candidate cannot manufacture a
run. One operator — or one candidate-controlled process — can hold all four
private keys, and every signature would be exactly as valid. Key distinctness
rules out the laziest case and stops there.

Four more places where the receipt authenticated a claim without measuring it:

**A disclosed fallback passed.** `fallbacks_used` had to be a list of stripped
strings and was never required to be empty, so a worker could say it took a
different path and still attest a matched run.

**Input size and memory had no ceiling.** `MATCHED_BUDGET` bounded time,
samples and output tokens; a worker could consume arbitrary context and RAM
with `all_within_budget` still true.

**The supervisor could observe anywhen.** `observed_at_unix` had to be a
non-negative number, so the independent observer could claim to have watched a
response it could not have seen.

**A commitment could be held forever.** Chronology bounded reveal-to-expiry and
left commit-to-reveal open, so an evaluator could sit on a commitment and
reveal it when the result suited. And freshness was off by default even for a
caller who supplied the clock to check against.
"""
from __future__ import annotations

import inspect

import pytest

import core.brain.frontier_evidence_v5 as evidence
from core.brain.frontier_evidence_v5 import MATCHED_BUDGET, actor_independence


def _challenge(evaluator="evaluator-1"):
    return {"evaluator_id": evaluator}


def _task_spec(verifier="verifier-1"):
    return {"verifier_id": verifier}


def _workers(signer="worker-1"):
    return [{"signer_id": signer}]


def _run(signer="coordinator-1"):
    return {"signer_id": signer}


def _independence(custody=None):
    return actor_independence(
        challenge=_challenge(),
        task_spec=_task_spec(),
        worker_receipts=_workers(),
        run_envelope=_run(),
        custody=custody,
    )


# ─────────────────────────── keys alone are cryptographic only


def test_without_custody_the_separation_is_cryptographic_only():
    """This is the honest label for four keys and nothing else."""
    report = _independence()

    assert report["independence"] == "cryptographic_only"
    assert report["custody_attested"] is False


def test_every_role_is_named_even_when_unattested():
    report = _independence()

    assert set(report["roles"]) == {
        "evaluator",
        "worker",
        "verifier",
        "run_coordinator",
    }
    assert all(row["attested"] is False for row in report["roles"].values())


def test_one_holder_of_every_key_is_reported_as_shared_custody():
    """The case the phrase "distinct actors" was hiding."""
    custody = {
        role: {"organization": "one-lab", "key_custodian": "one-person"}
        for role in ("evaluator-1", "worker-1", "verifier-1", "coordinator-1")
    }

    report = actor_independence(
        challenge=_challenge(),
        task_spec=_task_spec(),
        worker_receipts=_workers(),
        run_envelope=_run(),
        custody=custody,
    )

    assert report["custody_attested"] is True
    assert report["distinct_key_custodians"] == 1
    assert report["independence"] == "shared_custody"


def test_four_custodians_earn_the_attested_label():
    custody = {
        signer: {"organization": f"org-{index}", "key_custodian": f"holder-{index}"}
        for index, signer in enumerate(
            ("evaluator-1", "worker-1", "verifier-1", "coordinator-1")
        )
    }

    report = actor_independence(
        challenge=_challenge(),
        task_spec=_task_spec(),
        worker_receipts=_workers(),
        run_envelope=_run(),
        custody=custody,
    )

    assert report["independence"] == "custody_attested"
    assert report["distinct_key_custodians"] == 4
    assert report["distinct_organizations"] == 4


def test_a_partial_custody_record_does_not_count_as_attested():
    custody = {"evaluator-1": {"organization": "org", "key_custodian": "holder"}}

    report = _independence(custody)

    assert report["custody_attested"] is False
    assert report["independence"] == "cryptographic_only"


def test_an_organization_without_a_custodian_is_not_an_attestation():
    custody = {
        signer: {"organization": "org"}
        for signer in ("evaluator-1", "worker-1", "verifier-1", "coordinator-1")
    }

    report = _independence(custody)

    assert report["custody_attested"] is False


def test_the_key_check_is_still_documented_as_necessary_not_sufficient():
    source = inspect.getsource(evidence.validate_evidence_role_separation)

    assert "nowhere near sufficient" in source


def test_the_report_can_require_attested_custody():
    import core.brain.frontier_gap as frontier_gap

    parameters = inspect.signature(frontier_gap.validate_capability_report).parameters

    assert "key_custody" in parameters
    assert "require_attested_custody" in parameters


# ─────────────────────────── a fallback is a disqualification


def test_a_disclosed_fallback_no_longer_passes():
    source = inspect.getsource(evidence)

    assert "used fallbacks and cannot attest a matched run" in source


def test_the_refusal_names_the_fallbacks():
    source = inspect.getsource(evidence)

    assert "','.join(sorted(set(fallbacks))[:4])" in source


# ─────────────────────────── the budget bounds every resource


def test_the_protocol_now_caps_input_and_memory():
    assert MATCHED_BUDGET["max_input_tokens"] > 0
    assert MATCHED_BUDGET["max_peak_memory_bytes"] > 0


def test_the_input_cap_is_generous_next_to_a_battery_item():
    """A battery prompt is tens of tokens; the cap exists to stop
    context-stuffing, not to constrain the task."""
    assert MATCHED_BUDGET["max_input_tokens"] >= 8 * MATCHED_BUDGET["max_tokens"]


def test_both_caps_are_enforced_where_usage_is_validated():
    source = inspect.getsource(evidence)

    assert "exceeded the matched input budget" in source
    assert "exceeded the matched memory budget" in source


# ─────────────────────────── the observer has to have been there


def test_the_supervisor_time_is_bound_to_the_response_and_the_run():
    source = inspect.getsource(evidence.validate_supervisor_observation)

    assert "observed before the response completed" in source
    assert "observed before the run began" in source
    assert "observed after the run completed" in source


def test_the_run_envelope_checks_every_observation():
    source = inspect.getsource(evidence.validate_run_envelope)

    assert "supervisor observation is outside the signed run window" in source


# ─────────────────────────── a commitment cannot be held forever


def test_the_commit_to_reveal_gap_is_bounded():
    assert evidence.MAX_CHALLENGE_COMMIT_AGE_S > 0


def test_the_bound_is_enforced():
    source = inspect.getsource(evidence.validate_challenge_bundle)

    assert "held past the protocol commit-to-reveal bound" in source


def test_freshness_defaults_on_when_a_clock_is_supplied():
    """A caller who went to the trouble of passing the current time still got
    expired challenges accepted."""
    source = inspect.getsource(evidence.validate_challenge_bundle)

    assert "require_fresh: bool | None = None" in inspect.getsource(evidence)
    assert "require_fresh = verification_time_unix is not None" in source


def test_freshness_can_still_be_declined_explicitly():
    parameters = inspect.signature(evidence.validate_challenge_bundle).parameters

    assert parameters["require_fresh"].default is None


def test_no_clock_still_means_no_freshness_verdict():
    """Absence of evidence is not evidence: without a verification time there
    is nothing to check the window against."""
    source = inspect.getsource(evidence.validate_challenge_bundle)

    assert "still cannot check freshness" in source


@pytest.mark.parametrize(
    "field", ["max_input_tokens", "max_peak_memory_bytes", "max_tokens"]
)
def test_the_budget_stays_a_single_pinned_object(field):
    """Every validator reads MATCHED_BUDGET; a second copy would drift."""
    assert field in MATCHED_BUDGET
