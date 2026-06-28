"""Unity, not federation: nociception / social / epistemic engines bind into ONE mind-moment.

These prove the engines are members of the unified state the cognitive cycle computes — they
contribute BoundContent to gather_contents — rather than being consulted as side islands.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from core.unity.runtime import UnityRuntime


def _state(objective="", partner=""):
    cognition = SimpleNamespace(
        current_objective=objective, current_origin=partner, current_partner=partner,
    )
    return SimpleNamespace(cognition=cognition, affect=None, working_memory=None, world_state=None)


def test_felt_state_binds_when_in_pain(monkeypatch):
    from core.affect.nociception import get_nociception_engine, DamageChannel
    eng = get_nociception_engine()
    eng.reset()
    eng.register_damage(DamageChannel.MEMORY_CORRUPTION, 0.9)

    contents = UnityRuntime().gather_contents(_state(objective="hello"))
    interoception = [c for c in contents if c.modality == "interoception"]
    assert interoception, "nociception did not bind into the unified contents"
    assert interoception[0].source == "nociception"
    assert interoception[0].salience > 0.4
    eng.reset()


def test_no_felt_state_when_healthy():
    from core.affect.nociception import get_nociception_engine
    get_nociception_engine().reset()
    contents = UnityRuntime().gather_contents(_state(objective="hello"))
    assert not [c for c in contents if c.modality == "interoception"]


def test_epistemic_stance_binds_with_warranted_confidence():
    # A speculative objective must enter the bound state as a LOW-confidence content, so the
    # whole mind holds it tentatively — calibration as state, not as a prompt instruction.
    contents = UnityRuntime().gather_contents(
        _state(objective="will superintelligence definitely arrive by 2045")
    )
    epi = [c for c in contents if c.modality == "epistemic"]
    assert epi, "epistemic stance did not bind into the unified contents"
    assert epi[0].confidence <= 0.4   # warranted confidence is low → the mind is uncertain
    assert epi[0].source == "epistemic_calibration"


def test_formal_objective_is_not_low_confidence_noise():
    contents = UnityRuntime().gather_contents(_state(objective="2 + 2 = 4"))
    # a formal claim, if it binds at all, must not enter as low-confidence uncertainty
    epi = [c for c in contents if c.modality == "epistemic"]
    assert all(c.confidence >= 0.5 for c in epi)


def test_social_binds_when_interlocutor_known():
    from core.social.other_agent_model import get_other_agent_model
    oam = get_other_agent_model()
    # seed a few observations so the estimate has real confidence
    for _ in range(6):
        oam.observe_signal("bryan", presence=0.8, affiliation=0.6, threat=0.4)

    contents = UnityRuntime().gather_contents(_state(objective="hi", partner="bryan"))
    social = [c for c in contents if c.modality == "social"]
    assert social, "other-agent estimate did not bind into the unified contents"
    assert social[0].source == "other_agent_model"
    # confidence is our ACTUAL certainty about them, not a guess asserted as fact
    assert 0.0 < social[0].confidence <= 1.0


def test_unknown_interlocutor_does_not_fabricate_social_content():
    contents = UnityRuntime().gather_contents(_state(objective="hi", partner="self"))
    assert not [c for c in contents if c.modality == "social"]


def test_self_audit_binds_when_a_draft_overclaims(monkeypatch):
    # A leading draft that overclaims must surface an honesty (metacognition) content so the
    # mind holds it critically — the adversarial auditor is in the moment, not an island.
    rt = UnityRuntime()
    overclaiming = SimpleNamespace(
        content="This is definitely, certainly, undeniably proven and always true."
    )
    monkeypatch.setattr(rt, "_draft_inputs", lambda: [overclaiming])

    contents = rt.gather_contents(_state(objective="claim something"))
    audit = [c for c in contents if c.modality == "metacognition" and c.source == "adversarial_audit"]
    assert audit, "adversarial auditor did not bind a verdict on the overclaiming draft"
    assert audit[0].confidence < 1.0  # risk lowered the held-confidence


def test_self_audit_silent_on_measured_draft(monkeypatch):
    rt = UnityRuntime()
    measured = SimpleNamespace(content="This likely helps, based on the benchmark we ran.")
    monkeypatch.setattr(rt, "_draft_inputs", lambda: [measured])
    contents = rt.gather_contents(_state(objective="claim something"))
    assert not [c for c in contents if c.source == "adversarial_audit"]
