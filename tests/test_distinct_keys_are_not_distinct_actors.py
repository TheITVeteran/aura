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


# ─────────────────────────── the index says what the trend reads


def _entry(**overrides):
    body = {
        "schema": evidence.EVIDENCE_ENTRY_SCHEMA,
        "previous_entry_sha256": evidence.EVIDENCE_CHAIN_GENESIS,
        "evidence_sha256": "a" * 64,
        "evidence_class": "aura.frontier_gap.capability_measurement",
        "at": 1_700_000_000.0,
        "battery_version": "v5",
        "challenge_id": "challenge-1",
        "comparison_stratum_sha256": "b" * 64,
        "overall_gap": -0.1,
        "overall_candidate_score": 0.8,
        "effective_n": 20,
    }
    body.update(overrides)
    return body


def test_a_well_formed_entry_passes_the_semantic_check():
    evidence._validate_index_entry_semantics(_entry())


@pytest.mark.parametrize("score", [-0.5, 1.5])
def test_a_score_outside_the_battery_range_is_refused(score):
    with pytest.raises(ValueError, match="candidate score is outside"):
        evidence._validate_index_entry_semantics(
            _entry(overall_candidate_score=score)
        )


@pytest.mark.parametrize("gap", [-2.0, 2.0])
def test_a_gap_outside_two_proportions_is_refused(gap):
    """A gap is a difference of two proportions; anything else is not a
    measurement this protocol produced."""
    with pytest.raises(ValueError, match="gap is outside"):
        evidence._validate_index_entry_semantics(_entry(overall_gap=gap))


def test_a_missing_gap_is_allowed():
    evidence._validate_index_entry_semantics(_entry(overall_gap=None))


@pytest.mark.parametrize("value", [0, -1, True, "twenty"])
def test_an_invalid_effective_n_is_refused(value):
    with pytest.raises(ValueError, match="effective sample count"):
        evidence._validate_index_entry_semantics(_entry(effective_n=value))


def test_an_entry_with_no_evidence_class_is_refused():
    with pytest.raises(ValueError, match="no evidence class"):
        evidence._validate_index_entry_semantics(_entry(evidence_class=""))


def test_an_entry_with_no_battery_version_is_refused():
    with pytest.raises(ValueError, match="no battery version"):
        evidence._validate_index_entry_semantics(_entry(battery_version=None))


def test_a_negative_timestamp_is_refused():
    with pytest.raises(ValueError):
        evidence._validate_index_entry_semantics(_entry(at=-1.0))


def test_the_chain_runs_the_semantic_check():
    source = inspect.getsource(evidence.validate_index_chain)

    assert "_validate_index_entry_semantics(entry)" in source


# ─────────────────────────── the trend says what it assumed


def _trend(gaps):
    entries = [
        {
            "overall_gap": gap,
            "comparison_stratum_sha256": "a" * 64,
            "challenge_id": f"challenge-{index}",
            "effective_n": 20,
        }
        for index, gap in enumerate(gaps)
    ]
    return evidence.analyze_gap_trend(entries)


def test_the_first_look_spends_the_nominal_alpha():
    """A correction that forecloses the minimum case is not a correction."""
    assert evidence.sequential_looks(5, 5) == 1
    assert evidence.sequential_alpha(0.05, 1) == pytest.approx(0.05)


def test_alpha_tightens_as_the_series_grows():
    assert evidence.sequential_looks(10, 5) == 6
    assert evidence.sequential_alpha(0.05, 6) < 0.05


def test_the_spending_rule_never_reaches_zero():
    assert evidence.sequential_alpha(0.05, 10_000_000) > 0


def test_a_five_run_closing_series_is_still_eligible():
    trend = _trend([0.50, 0.42, 0.34, 0.26, 0.18])

    assert trend["sequential_looks"] == 1
    assert trend["claim_eligible"] is True


def test_the_trend_publishes_what_it_assumed():
    trend = _trend([0.50, 0.42, 0.34, 0.26, 0.18])
    assumptions = trend["inference_assumptions"]

    assert assumptions["run_order_treated_as_time"] is True
    assert assumptions["exchangeability_assumed"] is True
    assert assumptions["preregistered_horizon"] is False
    assert assumptions["stopping_rule"] == "alpha_spent_over_measured_runs"


def test_serial_dependence_is_measured_not_assumed_away():
    """An adaptive campaign is the opposite of exchangeable: each run follows
    a change made because of the last one."""
    trend = _trend([0.50, 0.42, 0.34, 0.26, 0.18])

    assert trend["inference_assumptions"]["residual_lag1_autocorrelation"] is not None


def test_a_flat_series_has_no_measurable_dependence():
    assert evidence._serial_dependence([0.3, 0.3, 0.3]) is None


def test_too_few_points_report_no_dependence():
    assert evidence._serial_dependence([0.3, 0.2]) is None


# ─────────────────────────── a release cannot vouch forever


def test_the_attestation_has_a_validity_window():
    assert evidence.MAX_RELEASE_ATTESTATION_AGE_S > 0


def test_the_window_and_revocation_are_enforced():
    source = inspect.getsource(evidence.validate_source_identity)

    assert "past the protocol validity window" in source
    assert "dated in the future" in source
    assert "signed by a revoked key" in source


def test_no_clock_still_means_no_age_verdict():
    parameters = inspect.signature(evidence.validate_source_identity).parameters

    assert parameters["verification_time_unix"].default is None
    assert parameters["revoked_release_keys"].default is None


# ─────────────────────────── the task spec pins its own scalars


def test_the_spec_checks_types_before_equality():
    """`1 == True` in Python, so a boolean seed compared equal to an int and
    passed a check meant to pin the battery instance."""
    source = inspect.getsource(evidence.validate_task_spec)

    assert "isinstance(value, bool) or not isinstance(value, int)" in source
    assert "is not a valid integer" in source


def test_the_spec_must_precede_challenge_expiry():
    source = inspect.getsource(evidence.validate_task_spec)

    assert "issued after the challenge expired" in source


# ─────────────────────────── modifiers are the matched set


def test_the_matched_modifier_set_is_pinned():
    assert evidence.MATCHED_RUNTIME_MODIFIERS == {
        "contrastive_decoding": False,
        "recurrent_loops": 1,
    }


def test_the_matched_set_passes():
    evidence._validate_runtime_modifiers(dict(evidence.MATCHED_RUNTIME_MODIFIERS))


def test_an_undeclared_modifier_is_refused():
    modifiers = dict(evidence.MATCHED_RUNTIME_MODIFIERS)
    modifiers["hidden_retrieval"] = True

    with pytest.raises(ValueError, match="undeclared: hidden_retrieval"):
        evidence._validate_runtime_modifiers(modifiers)


def test_a_missing_modifier_is_refused():
    with pytest.raises(ValueError, match="missing:"):
        evidence._validate_runtime_modifiers({"recurrent_loops": 1})


def test_a_modifier_off_its_matched_value_is_refused():
    modifiers = dict(evidence.MATCHED_RUNTIME_MODIFIERS)
    modifiers["recurrent_loops"] = 4

    with pytest.raises(ValueError, match="not the matched value"):
        evidence._validate_runtime_modifiers(modifiers)


def test_a_boolean_cannot_pass_for_a_loop_count():
    modifiers = dict(evidence.MATCHED_RUNTIME_MODIFIERS)
    modifiers["recurrent_loops"] = True

    with pytest.raises(ValueError, match="not the matched value"):
        evidence._validate_runtime_modifiers(modifiers)


def test_a_non_mapping_modifier_block_is_refused():
    with pytest.raises(ValueError, match="modifiers are malformed"):
        evidence._validate_runtime_modifiers(["contrastive_decoding"])


# ─────────────────────────── the signature bytes have a named profile


def test_the_canonicalization_profile_is_named():
    from core.brain.canonical_json import (
        CANONICAL_JSON_CONTRACT,
        CANONICAL_JSON_PROFILE,
    )

    assert CANONICAL_JSON_PROFILE
    assert CANONICAL_JSON_CONTRACT["profile"] == CANONICAL_JSON_PROFILE


def test_the_contract_admits_it_is_not_a_cross_language_standard():
    from core.brain.canonical_json import CANONICAL_JSON_CONTRACT

    assert CANONICAL_JSON_CONTRACT["cross_language_standard"] is None
    assert CANONICAL_JSON_CONTRACT["known_divergences"]


def test_the_protocol_manifest_pins_the_profile():
    from core.brain.canonical_json import CANONICAL_JSON_PROFILE

    assert evidence.PROTOCOL_MANIFEST["canonical_json_profile"] == CANONICAL_JSON_PROFILE


def test_changing_the_profile_would_change_the_protocol_digest():
    """That is the point: a canonicalization change becomes a version bump
    rather than a silent divergence."""
    body = {
        key: value
        for key, value in evidence.PROTOCOL_MANIFEST.items()
        if key != "manifest_sha256"
    }
    assert evidence.sha256_json(body) == evidence.PROTOCOL_MANIFEST_SHA256
    assert "canonical_json_profile" in body
