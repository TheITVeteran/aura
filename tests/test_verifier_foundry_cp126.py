"""CP126 verifier foundry — ground-truth trust root and ledger/memory coherence."""
from __future__ import annotations

import pytest

from core.brain.verifiers.foundry import TRUSTED_GRADE_SOURCES, VerifierFoundry


@pytest.fixture
def foundry(tmp_path):
    f = VerifierFoundry(root=tmp_path / "foundry")
    yield f
    f.close()


def _record(foundry, *, verifier="v1", domain="planning", hard_pass=True):
    return foundry.record_verdict(
        verifier=verifier, domain=domain, hard_pass=hard_pass,
        score=1.0 if hard_pass else 0.0, checked=True, task_key="t1",
    )


class TestGradeTrustRoot:
    """cd3bd98e: only a trusted ground-truth channel may grade."""

    def test_untrusted_source_is_refused(self, foundry):
        vid = _record(foundry)
        assert foundry.grade_verdict(vid, truth_pass=True, source="whoever") is False
        # The verdict stays pending: nothing was graded.
        assert vid in foundry.pending_verdicts()

    def test_empty_source_is_refused(self, foundry):
        vid = _record(foundry)
        assert foundry.grade_verdict(vid, truth_pass=True, source="") is False

    def test_trusted_sources_are_accepted(self, foundry):
        for source in sorted(TRUSTED_GRADE_SOURCES):
            vid = _record(foundry)
            assert foundry.grade_verdict(vid, truth_pass=True, source=source) is True

    def test_source_is_case_insensitive(self, foundry):
        vid = _record(foundry)
        assert foundry.grade_verdict(vid, truth_pass=True, source="AUDIT") is True

    def test_a_verdict_is_graded_once(self, foundry):
        vid = _record(foundry)
        assert foundry.grade_verdict(vid, truth_pass=True, source="audit") is True
        # A second channel cannot move reliability by re-grading.
        assert foundry.grade_verdict(vid, truth_pass=False, source="human") is False

    def test_production_channels_are_trusted(self):
        # The live callers in frontier_gap and experience_engines.
        assert "frontier_battery" in TRUSTED_GRADE_SOURCES
        assert "prediction_resolution" in TRUSTED_GRADE_SOURCES


class TestLedgerDivergenceFailsClosed:
    """1cd91c2e + 9d1a2e8c: memory must not govern admission alone."""

    def test_admission_refuses_while_events_are_unpersisted(self, foundry):
        foundry._unpersisted_events.add("vf-orphan")
        decision = foundry.domain_admitted("code")  # a SEED domain, normally admitted
        assert decision.admitted is False
        assert decision.reason == "ledger_divergence"

    def test_is_alive_reports_the_divergence(self, foundry):
        assert foundry.is_alive() is True
        foundry._unpersisted_events.add("vf-orphan")
        assert foundry.is_alive() is False

    def test_clean_foundry_admits_seed_domains(self, foundry):
        assert foundry.domain_admitted("code").admitted is True
