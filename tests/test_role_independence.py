"""tests/test_role_independence.py — a complete receipt chain is not a witness.

Operationally: this measures whether the four roles in a governed decision —
proposer, authorizer, criterion-setter, evidence-interpreter — were filled by
distinct sources, and names the collapses.

The gap, stated by an outside reviewer and correct as stated: even with every
action passing through UnifiedWill and AuthorityGateway, the same architecture
and model-generated context can propose an action, authorize it, define what
success means, and interpret the evidence. A model can be consistently wrong at
every stage and still emit a complete, signed receipt chain. `WillDecision`
records `source` and no other role, so nothing in a receipt could ever have
shown this.

These tests pin the distinction. None of them claims a witnessed decision is a
correct one — that is the failure being named, and reproducing it one level up
would be worse than not trying.
"""

from __future__ import annotations

import pytest

from core.governance.role_independence import (
    CONSEQUENTIAL_SCOPES,
    IndependenceReport,
    Role,
    RoleAttribution,
    analyse,
    from_mapping,
)


def test_one_source_in_every_role_is_circular():
    """The reviewer's scenario, exactly."""
    report = analyse(
        RoleAttribution(
            proposer="cognitive_engine",
            authorizer="cognitive_engine",
            criterion="cognitive_engine",
            interpreter="cognitive_engine",
            effect_scope="privileged_mutation",
        )
    )

    assert report.verdict == "circular"
    assert report.fully_circular is True
    assert report.consequential is True
    assert "could have reported failure" in report.describe()


def test_the_proposer_grading_its_own_work_is_not_witnessed():
    """Three parties involved and the verdict still is not "witnessed".

    This is the pair that matters. Every other collapse degrades a decision;
    this one removes the possibility of it being reported as a failure.
    """
    report = analyse(
        RoleAttribution(
            proposer="model",
            authorizer="authority_gateway",
            criterion="preregistration",
            interpreter="model",
        )
    )

    assert report.self_graded is True
    assert report.verdict == "self_graded"
    assert report.distinct_sources == 3, "three real parties, and it still does not count"
    assert (Role.PROPOSER, Role.INTERPRETER) in report.collapsed


def test_four_distinct_sources_are_witnessed():
    report = analyse(
        RoleAttribution(
            proposer="cognitive_engine",
            authorizer="authority_gateway",
            criterion="preregistration",
            interpreter="deterministic_verifier",
        )
    )

    assert report.verdict == "witnessed"
    assert report.distinct_sources == 4
    assert report.collapsed == ()


def test_nothing_recorded_is_reported_as_unattributed_not_as_fine():
    """Silence must not read as independence.

    Every decision in this repository is currently unattributed, and the report
    should say so rather than default to the flattering answer.
    """
    report = analyse(RoleAttribution())
    assert report.verdict == "unattributed"
    assert report.witnessed is False


def test_a_lone_proposer_is_unwitnessed():
    report = analyse(RoleAttribution(proposer="cognitive_engine"))
    assert report.verdict == "unwitnessed"


def test_an_unfilled_role_is_not_attributed_to_the_proposer():
    """Guessing would manufacture false comfort in the other direction."""
    attribution = RoleAttribution(proposer="ce", interpreter=None)
    assert attribution.get(Role.INTERPRETER) is None
    assert analyse(attribution).self_graded is False


def test_blank_and_whitespace_count_as_unattributed():
    """An empty string is not a source, and must not compare equal to one."""
    attribution = RoleAttribution(proposer="   ", interpreter="")
    assert attribution.get(Role.PROPOSER) is None
    assert attribution.filled() == {}
    assert analyse(attribution).verdict == "unattributed"


def test_consequential_scopes_are_flagged():
    """Reading a file unwitnessed is unremarkable; mutating the world is not."""
    for scope in ("privileged_mutation", "external_effect", "irreversible", "unknown"):
        assert scope in CONSEQUENTIAL_SCOPES
        assert analyse(RoleAttribution(proposer="x", effect_scope=scope)).consequential

    assert not analyse(RoleAttribution(proposer="x", effect_scope="read_only")).consequential


def test_an_unknown_scope_is_treated_as_consequential():
    """Fail toward reporting. An action nobody classified is not known to be safe."""
    assert analyse(RoleAttribution(proposer="x")).consequential is True


def test_from_mapping_accepts_receipt_shaped_payloads():
    """`source` is what receipts already carry, so it maps to proposer."""
    attribution = from_mapping(
        {
            "source": "cognitive_engine",
            "authority": "authority_gateway",
            "effect_scope": "privileged_mutation",
        }
    )
    assert attribution.proposer == "cognitive_engine"
    assert attribution.authorizer == "authority_gateway"
    assert attribution.criterion is None

    report = analyse(attribution)
    # An independent authorizer IS a witness, so the verdict is honest. What
    # today's receipts cannot say is who set the bar and who judged the
    # evidence, and the report names those rather than quietly scoring them.
    assert report.verdict == "witnessed"
    assert set(report.unattributed) == {Role.CRITERION, Role.INTERPRETER}


def test_the_report_serialises():
    payload = analyse(
        RoleAttribution(proposer="a", authorizer="b", criterion="c", interpreter="a")
    ).to_dict()
    assert payload["schema"] == "aura.role_independence.v1"
    assert payload["verdict"] == "self_graded"
    assert ["proposer", "interpreter"] in payload["collapsed"]


def test_the_module_does_not_score_correctness():
    """The one thing it must not do.

    Judging whether a decision was RIGHT is the failure this module names. A
    scoring method here would reproduce it one level up, with more authority.
    """
    report = analyse(RoleAttribution(proposer="a", interpreter="b"))
    assert not hasattr(report, "score")
    assert not hasattr(report, "quality")
    assert not hasattr(report, "correct")
    assert isinstance(report, IndependenceReport)
