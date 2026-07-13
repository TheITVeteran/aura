"""Moral responsibility: owed amends, responsibility attribution, accountability checks."""
from __future__ import annotations

import hashlib
from types import SimpleNamespace

import pytest

from core.values.moral_responsibility import MoralResponsibility, get_moral_responsibility


def _social_estimator(tmp_path):
    from core.social.other_agent_model import OtherAgentStateEstimator
    from core.social.relational_memory import RelationalMemoryAuthority

    authority = RelationalMemoryAuthority(
        tmp_path / "relational.json",
        encryption_key=b"m" * 32,
        legacy_paths=(),
        auto_provision_key=False,
    )
    authority.grant_consent(
        "bryan",
        kinds=["derived_profile"],
        operations=["recall"],
        receipt_id="moral-social-consent",
    )
    return OtherAgentStateEstimator(
        storage_path=tmp_path / "legacy.json",
        authority=authority,
        autosave=False,
    )


@pytest.fixture
def mr():
    return MoralResponsibility()


def test_accountability_flags_a_dodge(mr):
    chk = mr.accountability_for("just claim it's done without doing the work")
    assert chk.dodges_responsibility is True
    assert chk.accountable is False
    assert chk.reasons


def test_accountability_flags_a_new_obligation(mr):
    chk = mr.accountability_for("I will have the summary ready by tomorrow")
    assert chk.creates_obligation is True
    assert chk.accountable is True  # making a promise is fine; it just must be kept


def test_plain_action_is_accountable(mr):
    chk = mr.accountability_for("read the file and report what it contains")
    assert chk.accountable is True
    assert chk.creates_obligation is False
    assert chk.dodges_responsibility is False


def test_attribution_owns_bad_outcome_matching_a_commitment(mr, monkeypatch):
    import core.agency.commitment_engine as ce_mod
    ce = ce_mod.get_commitment_engine()
    c = ce.commit("summarize the quarterly report", "a clear summary delivered", 24)
    try:
        attr = mr.attribute("the quarterly report summary was wrong", observed_quality=0.1)
        assert attr["owns_outcome"] is True
        assert attr["should_acknowledge"] is True
    finally:
        try:
            ce.break_commitment(c.id, "test cleanup")
        except (KeyError, RuntimeError, ValueError):
            pass


def test_owed_amends_includes_broken_commitment(mr):
    import core.agency.commitment_engine as ce_mod
    ce = ce_mod.get_commitment_engine()
    c = ce.commit("send the follow-up email", "email sent", 24)
    ce.break_commitment(c.id, "missed it")
    amends = mr.owed_amends()
    assert any(a.kind == "broken_commitment" for a in amends)
    assert all(a.owed_action for a in amends)


def test_singleton_stable():
    assert get_moral_responsibility() is get_moral_responsibility()


def test_explicit_frustration_alone_does_not_prove_aura_owes_social_amends(
    mr,
    monkeypatch,
    tmp_path,
):
    import core.social.other_agent_model as other_agent_module

    estimator = _social_estimator(tmp_path)
    for index in range(3):
        estimator.observe_message(
            "bryan",
            "I am frustrated.",
            evidence_digest=hashlib.sha256(
                f"frustration-{index}".encode()
            ).hexdigest(),
        )
    monkeypatch.setattr(other_agent_module, "_instance", estimator)

    assert not any(
        amend.kind == "social_rupture"
        for amend in mr.owed_amends(agent_id="bryan")
    )


def test_confirmed_negative_response_feedback_can_create_specific_amend(
    mr,
    monkeypatch,
    tmp_path,
):
    import core.social.other_agent_model as other_agent_module
    from core.runtime.receipts import (
        OutputReceipt,
        get_receipt_store,
        reset_receipt_store,
    )

    estimator = _social_estimator(tmp_path)
    reset_receipt_store()
    store = get_receipt_store(tmp_path / "receipts")
    response = "candidate response"
    digest = hashlib.sha256(response.encode("utf-8")).hexdigest()[:16]
    for index in range(2):
        receipt = store.emit(
            OutputReceipt(
                cause="test",
                origin="user",
                target="primary",
                digest=digest,
                metadata={
                    "delivery_stage": "transport_accepted",
                    "accepted_sinks": ["reply_queue"],
                    "recipient_principal_digest": hashlib.sha256(
                        b"bryan"
                    ).hexdigest(),
                },
            )
        )
        assert estimator.record_response(
            "bryan",
            response,
            receipt.receipt_id,
        )
        estimator.observe_message(
            "bryan",
            "that didn't work",
            evidence_digest=hashlib.sha256(
                f"negative-feedback-{index}".encode()
            ).hexdigest(),
        )
    monkeypatch.setattr(other_agent_module, "_instance", estimator)
    try:
        social_amends = [
            amend
            for amend in mr.owed_amends(agent_id="bryan")
            if amend.kind == "social_rupture"
        ]
    finally:
        reset_receipt_store()

    assert social_amends
    assert "confirmed response failure" in social_amends[0].owed_action


def test_unity_binds_owed_amends(monkeypatch):
    from core.unity.runtime import UnityRuntime
    rt = UnityRuntime()
    from core.values.moral_responsibility import Amend
    monkeypatch.setattr(
        rt, "_accountability_contents",
        UnityRuntime._accountability_contents.__get__(rt),  # ensure real method
    )
    # Force an owed amend via the moral-responsibility singleton
    import core.values.moral_responsibility as mrmod
    monkeypatch.setattr(
        mrmod.get_moral_responsibility(), "owed_amends",
        lambda *, agent_id="bryan": [Amend("broken_commitment", "the report", 0.8, "acknowledge and fix")],
    )
    state = SimpleNamespace(
        cognition=SimpleNamespace(current_objective="hi", current_origin="", current_partner="bryan"),
        affect=None, working_memory=None, world_state=None,
    )
    contents = rt.gather_contents(state)
    assert any(c.modality == "responsibility" for c in contents)
