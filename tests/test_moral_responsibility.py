"""Moral responsibility: owed amends, responsibility attribution, accountability checks."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from core.values.moral_responsibility import MoralResponsibility, get_moral_responsibility


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
        except Exception:
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
