"""CP126: a boolean in a params dict is not authorization.

Every privileged actuator gated on `params["_aura_authorized"]`, which the
ActuatorRegistry injects after the AuthorityGateway approves. Any direct caller
could set that key themselves. This pins the shared verifier that closes that
class across all of them (8900fa05, 27651212, 9f94bf4d, 251ada47, bdb4255d,
5ce6b589, 5acd8c38, …).
"""
from __future__ import annotations

import pytest

from core.actuators.authority import (
    actuator_authorization,
    current_authorization,
    verify_actuator_authority,
)
from core.runtime.capability_tokens import (
    get_capability_token_store,
    reset_capability_token_store,
)


@pytest.fixture(autouse=True)
def _fresh_tokens():
    reset_capability_token_store()
    yield
    reset_capability_token_store()


# ── the core defect: a fabricated flag is not authorization ────────────────


def test_flag_without_registry_context_is_refused():
    ok, reason = verify_actuator_authority({"_aura_authorized": True}, actuator="web_fetch")
    assert ok is False
    assert "not authorization" in reason


def test_missing_flag_is_refused():
    ok, reason = verify_actuator_authority({}, actuator="web_fetch")
    assert ok is False and "requires ActuatorRegistry" in reason


def test_flag_inside_registry_context_is_accepted():
    with actuator_authorization("web_fetch"):
        ok, reason = verify_actuator_authority({"_aura_authorized": True}, actuator="web_fetch")
    assert ok is True and reason == ""


def test_context_for_a_different_actuator_is_refused():
    # An authorization for web_fetch must not authorize process_supervisor.
    with actuator_authorization("web_fetch"):
        ok, reason = verify_actuator_authority(
            {"_aura_authorized": True}, actuator="process_supervisor"
        )
    assert ok is False and "authorization is for 'web_fetch'" in reason


def test_wildcard_context_authorizes_any_actuator():
    with actuator_authorization("*"):
        ok, _ = verify_actuator_authority({"_aura_authorized": True}, actuator="anything")
    assert ok is True


def test_context_is_cleared_on_exit():
    with actuator_authorization("web_fetch"):
        assert current_authorization() is not None
    assert current_authorization() is None


# ── capability tokens are verified, not just carried ──────────────────────


def test_unknown_token_is_refused():
    with actuator_authorization("web_fetch"):
        ok, reason = verify_actuator_authority(
            {"_aura_authorized": True, "_capability_token_id": "cap-forged"},
            actuator="web_fetch",
        )
    assert ok is False and "unknown" in reason


def test_valid_token_is_accepted():
    token = get_capability_token_store().issue(capability="web_fetch", scope="s")
    with actuator_authorization("web_fetch"):
        ok, reason = verify_actuator_authority(
            {"_aura_authorized": True, "_capability_token_id": token.token_id},
            actuator="web_fetch",
        )
    assert ok is True, reason


def test_token_scoped_to_another_capability_is_refused():
    token = get_capability_token_store().issue(capability="git_operation", scope="s")
    with actuator_authorization("web_fetch"):
        ok, reason = verify_actuator_authority(
            {"_aura_authorized": True, "_capability_token_id": token.token_id},
            actuator="web_fetch",
        )
    assert ok is False and "scoped to 'git_operation'" in reason


def test_expired_token_is_refused():
    token = get_capability_token_store().issue(capability="web_fetch", scope="s", ttl_s=-1.0)
    with actuator_authorization("web_fetch"):
        ok, reason = verify_actuator_authority(
            {"_aura_authorized": True, "_capability_token_id": token.token_id},
            actuator="web_fetch",
        )
    assert ok is False and "expired" in reason


def test_revoked_token_is_refused():
    store = get_capability_token_store()
    token = store.issue(capability="web_fetch", scope="s")
    store.revoke(token.token_id)
    with actuator_authorization("web_fetch"):
        ok, reason = verify_actuator_authority(
            {"_aura_authorized": True, "_capability_token_id": token.token_id},
            actuator="web_fetch",
        )
    assert ok is False and "revoked" in reason


# ── every privileged actuator now routes through the verifier ─────────────


def test_all_privileged_actuators_use_the_shared_verifier():
    import pathlib

    expected = [
        "core/actuators/process_supervisor.py",
        "core/actuators/doc_ingest.py",
        "core/actuators/web_actuators.py",
        "core/actuators/git_pkg_actuators.py",
        "core/actuators/code_execution_actuator.py",
    ]
    for path in expected:
        src = pathlib.Path(path).read_text()
        assert "verify_actuator_authority" in src, f"{path} still trusts the raw flag"
        assert 'params.get("_aura_authorized")' not in src, (
            f"{path} still reads the raw boolean directly"
        )
