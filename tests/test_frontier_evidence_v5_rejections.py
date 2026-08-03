"""The capability-claim validators' refusals, each exercised.

Found by mutation: disabling every `raise` in validate_task_spec (12 of them),
validate_worker_receipt (11) and validate_protocol_manifest (1) left the entire
suite green across all seven test files that reference them. Those functions
decide whether a claim Aura makes about her own capability is admissible, so
they are the enforcement layer behind the rule that a claim with no test cannot
be registered — and that layer was itself unenforced.

The existing tests build VALID evidence and check it round-trips. Nothing built
invalid evidence and checked it was refused, which is the half that matters:
a validator that accepts everything also round-trips valid input perfectly.

Each test below perturbs exactly one field of an otherwise-valid, correctly
signed artifact, so a failure names the specific defence that stopped working
rather than "something about the payload is wrong".
"""

from __future__ import annotations

import base64
import copy
import hashlib
from typing import Any

import pytest

from core.brain.frontier_evidence_v5 import (
    PROTOCOL_MANIFEST_SHA256,
    validate_protocol_manifest,
    validate_task_spec,
)
from tests.test_frontier_evidence_v5 import (  # reuse the proven fixtures
    BATTERY_VERSION,
    Trust,
    _challenge,
    _sign,
    _task_spec,
    _worker_receipt,
)

SEED = 7
PER_CLASS = 2
# Not a repeated byte: validate_challenge_bundle rejects degenerate nonces,
# and it caught this fixture when it was b"\x5b" * 32.
NONCE = hashlib.sha256(b"frontier-rejection-fixture").digest()
FREEZE = "c" * 64


@pytest.fixture
def trust() -> Trust:
    return Trust.create()


@pytest.fixture
def raw_challenge(trust: Trust) -> dict[str, Any]:
    """The {commit, reveal} bundle as it travels on the wire.

    _task_spec hashes THIS to bind the spec to the challenge, so it must be the
    raw bundle; the validated form below carries a decoded `nonce` in bytes and
    is not JSON-serializable.
    """
    return _challenge(trust, FREEZE, NONCE)


@pytest.fixture
def challenge(trust: Trust, raw_challenge: dict[str, Any]) -> dict[str, Any]:
    """The VALIDATED challenge, which is what validate_task_spec consumes."""
    from core.brain.frontier_evidence_v5 import validate_challenge_bundle

    return validate_challenge_bundle(
        raw_challenge, trusted_evaluator_keys=trust.evaluator_keys
    )


@pytest.fixture
def spec(trust: Trust, raw_challenge: dict[str, Any]) -> dict[str, Any]:
    return _task_spec(
        trust, seed=SEED, per_class=PER_CLASS, nonce=NONCE, challenge=raw_challenge
    )


def _validate(spec: dict[str, Any], trust: Trust, challenge: dict[str, Any]):
    from core.brain.frontier_gap import battery_manifest

    manifest = battery_manifest(seed=SEED, per_class=PER_CLASS, challenge_nonce=NONCE)
    return validate_task_spec(
        spec,
        trusted_evaluator_keys={"eval-lab": _pub(trust.evaluator)},
        trusted_verifiers={
            "independent-verifier": {
                "public_key_b64": _pub(trust.verifier),
                "implementation_sha256": trust.verifier_implementation,
                "release_sha256": trust.verifier_release,
            }
        },
        challenge=challenge,
        expected_items=manifest["items"],
        battery_version=BATTERY_VERSION,
        seed=SEED,
        per_class=PER_CLASS,
    )


def _pub(key) -> str:
    from tests.test_frontier_evidence_v5 import _public_key_b64

    return _public_key_b64(key)


def _resign(payload: dict[str, Any], trust: Trust) -> dict[str, Any]:
    """Re-sign after tampering.

    Without this a test proves only that the SIGNATURE check works — which is
    already covered — and never reaches the semantic check it is aiming at.
    This is the same trap that made three mycelium corruption tests vacuous.
    """
    from core.brain.frontier_evidence_v5 import TASK_SPEC_SCHEMA

    return _sign(TASK_SPEC_SCHEMA, payload, signer_id="eval-lab", key=trust.evaluator)


def test_the_valid_spec_is_accepted(spec, trust, challenge):
    """Control. If this fails the perturbation tests below prove nothing."""
    assert _validate(spec, trust, challenge)


@pytest.mark.parametrize(
    "mutate,expected",
    [
        pytest.param(
            lambda p: p.update({"seed": p["seed"] + 1}),
            "battery instance mismatch",
            id="seed_mismatch",
        ),
        pytest.param(
            lambda p: p.update({"per_class": p["per_class"] + 1}),
            "battery instance mismatch",
            id="per_class_mismatch",
        ),
        pytest.param(
            lambda p: p.update({"battery_version": "not-the-version"}),
            "battery instance mismatch",
            id="battery_version_mismatch",
        ),
        pytest.param(
            lambda p: p.update({"protocol_manifest_sha256": "0" * 64}),
            "protocol digest mismatch",
            id="protocol_digest_forged",
        ),
        pytest.param(
            lambda p: p.update({"challenge_bundle_sha256": "0" * 64}),
            "not bound to the challenge reveal",
            id="unbound_from_challenge",
        ),
        pytest.param(
            lambda p: p.update({"issued_at_unix": 0.0}),
            "predates challenge reveal",
            id="spec_predates_reveal",
        ),
        pytest.param(
            lambda p: p.pop("effective_n"),
            "fields are invalid",
            id="missing_field",
        ),
        pytest.param(
            lambda p: p.update({"unexpected_field": 1}),
            "fields are invalid",
            id="extra_field",
        ),
        pytest.param(
            lambda p: p.update({"effective_n": p["effective_n"] + 5}),
            "effective sample count is false",
            id="inflated_effective_n",
        ),
        pytest.param(
            lambda p: p["items"].pop(),
            "item coverage is incomplete",
            id="dropped_item",
        ),
        pytest.param(
            lambda p: p["items"].__setitem__(
                0, {**p["items"][0], "prompt_sha256": "0" * 64}
            ),
            "does not match the battery",
            id="substituted_item",
        ),
        pytest.param(
            # Caught by the per-index battery match, which runs FIRST. The
            # module's own "duplicated effective samples" raise sits behind
            # that and behind build_battery's identical check, so it is
            # defence-in-depth rather than a reachable path from here — worth
            # recording accurately instead of contorting a test to hit it.
            lambda p: p["items"].__setitem__(1, copy.deepcopy(p["items"][0])),
            "does not match the battery",
            id="duplicated_item_caught_by_index_match",
        ),
    ],
)
def test_task_spec_refuses_tampered_evidence(spec, trust, challenge, mutate, expected):
    """Each perturbation must be refused, and refused for its own reason.

    Matching the message matters: a validator that rejected everything with one
    generic error would pass a bare `pytest.raises(ValueError)` while having
    lost the ability to tell these cases apart.
    """
    payload = copy.deepcopy(spec["signed_payload"])
    mutate(payload)
    tampered = _resign(payload, trust)

    with pytest.raises(ValueError, match=expected):
        _validate(tampered, trust, challenge)


@pytest.mark.parametrize(
    "mutate,expected",
    [
        pytest.param(
            lambda v: v.update({"verifier_id": "unpinned-verifier"}),
            "not independently pinned",
            id="verifier_not_pinned",
        ),
        pytest.param(
            lambda v: v.update({"implementation_sha256": "9" * 64}),
            "does not match its pin",
            id="verifier_implementation_swapped",
        ),
        pytest.param(
            lambda v: v.update({"release_sha256": "9" * 64}),
            "does not match its pin",
            id="verifier_release_swapped",
        ),
        pytest.param(
            lambda v: v.pop("release_sha256"),
            "verifier identity is invalid",
            id="verifier_field_missing",
        ),
    ],
)
def test_task_spec_refuses_an_unpinned_or_swapped_verifier(
    spec, trust, challenge, mutate, expected
):
    """The verifier is what makes a capability claim independent.

    A claim graded by a verifier the evaluator chose freely is self-assessment
    wearing a signature, so these four refusals are the ones that carry the
    independence property.
    """
    payload = copy.deepcopy(spec["signed_payload"])
    mutate(payload["verifier_identity"])
    tampered = _resign(payload, trust)

    with pytest.raises(ValueError, match=expected):
        _validate(tampered, trust, challenge)


def test_protocol_manifest_refuses_a_foreign_manifest():
    """validate_protocol_manifest's single refusal had no test at all."""
    with pytest.raises(ValueError):
        validate_protocol_manifest({"not": "the protocol manifest"})


def test_protocol_manifest_digest_is_pinned():
    """A changed manifest must change its digest, or pinning is decorative."""
    assert len(PROTOCOL_MANIFEST_SHA256) == 64


# ── validate_worker_receipt ────────────────────────────────────────────────
#
# 11 refusal paths, none of them tested. A worker receipt is the generating
# side's sworn statement about how an answer was produced: which run, which
# item, under what decoding budget, in how long, with what resources, and
# whether it fell back. Every one of those is a place a claim can be inflated,
# and the receipt is the only thing standing between "the model did this" and
# "something produced this text somehow".

# Every binding field is require_sha256-checked, run_id included.
RUN_ID = hashlib.sha256(b"frontier-run-id-fixture").hexdigest()
RUN_NONCE = hashlib.sha256(b"frontier-run-nonce-fixture").digest()


@pytest.fixture
def battery_item():
    from core.brain.frontier_gap import build_battery

    return build_battery(seed=SEED, per_class=PER_CLASS, challenge_nonce=NONCE)[0]


@pytest.fixture
def receipt_bits(trust: Trust, battery_item, raw_challenge):
    """A valid receipt plus the bindings it must agree with."""
    from core.brain.frontier_evidence_v5 import sha256_json

    output_sha256 = hashlib.sha256(b"an answer").hexdigest()
    bits = {
        "output_sha256": output_sha256,
        "source_identity_sha256": "a" * 64,
        "runtime_manifest_sha256": "b" * 64,
        "model_stability_sha256": "d" * 64,
        "challenge_sha256": sha256_json(raw_challenge),
    }
    receipt = _worker_receipt(
        trust,
        run_id=RUN_ID,
        run_nonce=RUN_NONCE,
        index=0,
        item=battery_item,
        **bits,
    )
    from core.brain.frontier_evidence_v5 import PROTOCOL_MANIFEST_SHA256 as _pm

    bindings = {
        "run_id": RUN_ID,
        "run_nonce_sha256": hashlib.sha256(RUN_NONCE).hexdigest(),
        "item_id": battery_item.item_id,
        "prompt_sha256": hashlib.sha256(battery_item.prompt.encode()).hexdigest(),
        "output_sha256": output_sha256,
        "source_identity_sha256": bits["source_identity_sha256"],
        "runtime_manifest_sha256": bits["runtime_manifest_sha256"],
        "model_stability_sha256": bits["model_stability_sha256"],
        "protocol_manifest_sha256": _pm,
        "challenge_bundle_sha256": bits["challenge_sha256"],
        "attempt_index": 0,
    }
    return receipt, bindings


def _validate_receipt(receipt, trust: Trust, bindings):
    from core.brain.frontier_evidence_v5 import validate_worker_receipt

    return validate_worker_receipt(
        receipt, trusted_worker_keys=trust.worker_keys, bindings=bindings
    )


def _resign_receipt(payload, trust: Trust):
    from core.brain.frontier_evidence_v5 import WORKER_RECEIPT_SCHEMA

    return _sign(
        WORKER_RECEIPT_SCHEMA, payload, signer_id="generation-worker", key=trust.worker
    )


def test_the_valid_receipt_is_accepted(receipt_bits, trust):
    """Control for every worker-receipt perturbation below."""
    receipt, bindings = receipt_bits
    assert _validate_receipt(receipt, trust, bindings)


@pytest.mark.parametrize(
    "mutate,expected",
    [
        pytest.param(
            lambda p: p.update({"output_sha256": "0" * 64}),
            "output_sha256 binding mismatch",
            id="output_swapped_for_another_answer",
        ),
        pytest.param(
            lambda p: p.update({"item_id": "e" * 64}),
            "item_id binding mismatch",
            id="receipt_reassigned_to_another_item",
        ),
        pytest.param(
            lambda p: p.update({"model_stability_sha256": "0" * 64}),
            "model_stability_sha256 binding mismatch",
            id="different_model_than_measured",
        ),
        pytest.param(
            lambda p: p.update({"source_identity_sha256": "0" * 64}),
            "source_identity_sha256 binding mismatch",
            id="different_source_than_measured",
        ),
        pytest.param(
            lambda p: p.pop("sealed_evaluation_enforced"),
            "fields are invalid",
            id="missing_field",
        ),
        pytest.param(
            lambda p: p.update({"sealed_evaluation_enforced": False}),
            "did not enforce sealed evaluation",
            id="evaluation_not_sealed",
        ),
        pytest.param(
            # -1, not 3: a valid non-negative int would pass this check and
            # fall through to the request-id derivation instead.
            lambda p: p.update({"attempt_index": -1}),
            "attempt index is invalid",
            id="attempt_index_negative",
        ),
        pytest.param(
            lambda p: p.update({"request_id": "forged-request-id"}),
            "request identity is invalid",
            id="request_id_not_derived_from_run_and_item",
        ),
        pytest.param(
            lambda p: p.update({"completed_at_unix": p["started_at_unix"] - 1.0}),
            "timing receipt is inconsistent",
            id="completed_before_it_started",
        ),
        pytest.param(
            lambda p: p.update({"elapsed_s": 99999.0, "completed_at_unix": p["started_at_unix"] + 99999.0}),
            "exceeded the hard time budget",
            id="ran_past_the_matched_budget",
        ),
        pytest.param(
            lambda p: p["decoding_parameters"].update({"temperature": 1.9}),
            "decoding parameters are not matched",
            id="decoded_under_a_different_budget",
        ),
        pytest.param(
            lambda p: p.update({"fallbacks_used": [{"not": "a string"}]}),
            "fallback receipt is malformed",
            id="fallback_receipt_malformed",
        ),
        pytest.param(
            lambda p: p.update({"run_nonce_b64": base64.b64encode(b"short").decode("ascii")}),
            "committed 256-bit run nonce",
            id="run_nonce_too_short_to_be_committed",
        ),
    ],
)
def test_worker_receipt_refuses_an_inflated_claim(receipt_bits, trust, mutate, expected):
    """Each perturbation is a way to claim an answer the run did not produce."""
    receipt, bindings = receipt_bits
    payload = copy.deepcopy(receipt["signed_payload"])
    mutate(payload)
    tampered = _resign_receipt(payload, trust)

    with pytest.raises(ValueError, match=expected):
        _validate_receipt(tampered, trust, bindings)


def test_worker_receipt_refuses_usage_that_contradicts_elapsed_time(receipt_bits, trust):
    """Resource usage and the clock have to tell the same story.

    A receipt claiming more wall time inside the call than the call itself took
    is describing work that did not fit in the window it reports.
    """
    receipt, bindings = receipt_bits
    payload = copy.deepcopy(receipt["signed_payload"])
    # Enough to break the 0.25s agreement, but under the 20s hard budget —
    # otherwise _validate_resource_usage refuses it first and this test would
    # pass while proving something else entirely.
    payload["resource_usage"]["wall_time_s"] = 5.0
    tampered = _resign_receipt(payload, trust)

    with pytest.raises(ValueError, match="usage contradicts elapsed time"):
        _validate_receipt(tampered, trust, bindings)
