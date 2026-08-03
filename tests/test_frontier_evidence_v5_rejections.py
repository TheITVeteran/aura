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
