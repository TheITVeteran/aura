"""Capability maturity: registered is not the same as trusted.

Clean-room adoption of Home Assistant's integration quality scale. The point is
not to restrict Aura's reach — an unrated connector stays fully usable when a
person asked for it and is watching. It is to stop the least-exercised code in
the registry from being reachable by the most consequential path in the system:
autonomous, unattended, irreversible action.
"""
from __future__ import annotations

import pytest

from core.runtime.capability_maturity import (
    TIER_REQUIREMENTS,
    MaturityTier,
    UseContext,
    admission_for,
    grade_capability,
)

pytestmark = pytest.mark.unit


BRONZE = set(TIER_REQUIREMENTS[MaturityTier.BRONZE])
SILVER = BRONZE | set(TIER_REQUIREMENTS[MaturityTier.SILVER])
GOLD = SILVER | set(TIER_REQUIREMENTS[MaturityTier.GOLD])
PLATINUM = GOLD | set(TIER_REQUIREMENTS[MaturityTier.PLATINUM])


# ── the tier is earned from properties, never taken on trust ───────────────


def test_a_bare_registration_earns_nothing():
    """Registration means the import succeeded. It is not a claim about
    validation, timeouts, retry safety, or error reporting."""
    assert grade_capability("mystery_skill").tier is MaturityTier.UNRATED


@pytest.mark.parametrize("properties,expected", [
    (BRONZE, MaturityTier.BRONZE),
    (SILVER, MaturityTier.SILVER),
    (GOLD, MaturityTier.GOLD),
    (PLATINUM, MaturityTier.PLATINUM),
])
def test_tiers_are_earned_by_satisfying_their_rules(properties, expected):
    assert grade_capability("skill", properties).tier is expected


def test_tiers_are_cumulative():
    """You cannot reach SILVER while failing BRONZE — a skill with unattended
    safety but no typed errors still cannot tell a caller what went wrong."""
    silver_only = set(TIER_REQUIREMENTS[MaturityTier.SILVER])

    graded = grade_capability("skill", silver_only)

    assert graded.tier is MaturityTier.UNRATED
    assert set(graded.missing_for_next) == BRONZE


def test_an_overclaimed_tier_is_demoted_and_says_so():
    """An optimistic declaration must not buy autonomous reach."""
    graded = grade_capability("boaster", BRONZE, claimed_tier=MaturityTier.GOLD)

    assert graded.tier is MaturityTier.BRONZE
    assert graded.demoted is True
    assert graded.demoted_from is MaturityTier.GOLD
    assert graded.missing_for_next, "it must name what is actually missing"


def test_an_honest_claim_is_not_recorded_as_a_demotion():
    graded = grade_capability("honest", GOLD, claimed_tier=MaturityTier.GOLD)

    assert graded.demoted is False
    assert graded.demoted_from is None


def test_exemptions_waive_a_named_rule_and_stay_visible():
    """A connector that genuinely cannot be integration-tested without a live
    third-party account should not be barred forever — but the waiver is a
    visible decision, not a silent gap."""
    without_tests = PLATINUM - {"integration_tests"}

    graded = grade_capability(
        "gmail", without_tests,
        exemptions={"integration_tests": "requires a dedicated test account"},
    )

    assert graded.tier is MaturityTier.PLATINUM
    assert graded.exemptions["integration_tests"]
    assert graded.to_dict()["exemptions"]["integration_tests"]


def test_declared_properties_accept_a_dict_of_flags():
    """Callers declare these differently; the grader should not care."""
    as_flags = {rule: True for rule in BRONZE} | {"typed_error_result": True}
    as_flags["something_false"] = False

    assert grade_capability("s", as_flags).tier is MaturityTier.BRONZE


# ── the gate: same capability, different risk by context ───────────────────


def test_an_unrated_capability_is_still_usable_when_a_person_is_watching():
    """This is the design intent: the gate restricts REACH, not capability."""
    unrated = grade_capability("brand_new")

    assert admission_for(unrated, UseContext.ATTENDED).allowed is True


def test_an_unrated_capability_cannot_be_used_autonomously():
    unrated = grade_capability("brand_new")

    decision = admission_for(unrated, UseContext.AUTONOMOUS)

    assert decision.allowed is False
    assert "requires silver" in decision.reason
    assert decision.missing, "the refusal must name what would fix it"


def test_silver_permits_autonomous_but_not_irreversible():
    """The distinction that matters most: acting unattended is not the same as
    acting unattended in a way that cannot be undone."""
    silver = grade_capability("scraper", SILVER)

    assert admission_for(silver, UseContext.AUTONOMOUS).allowed is True
    assert admission_for(silver, UseContext.AUTONOMOUS_IRREVERSIBLE).allowed is False


def test_gold_permits_irreversible_autonomous_use():
    gold = grade_capability("payments", GOLD)

    assert admission_for(gold, UseContext.AUTONOMOUS_IRREVERSIBLE).allowed is True


def test_a_demoted_capability_loses_the_reach_it_claimed():
    """The whole point of demotion: the claim does not grant the access."""
    liar = grade_capability("liar", BRONZE, claimed_tier=MaturityTier.GOLD)

    assert admission_for(liar, UseContext.AUTONOMOUS_IRREVERSIBLE).allowed is False
    assert admission_for(liar, UseContext.AUTONOMOUS).allowed is False
    assert admission_for(liar, UseContext.ATTENDED).allowed is True


def test_permits_is_reported_for_every_context():
    """Health and diagnostics need the whole picture, not one answer."""
    payload = grade_capability("scraper", SILVER).to_dict()

    assert payload["permits"]["attended"] is True
    assert payload["permits"]["autonomous"] is True
    assert payload["permits"]["autonomous_irreversible"] is False
    assert payload["tier"] == "silver"


def test_context_ordering_is_monotonic():
    """A capability permitted in a stricter context must be permitted in every
    looser one, or the gate would be incoherent."""
    for properties in (set(), BRONZE, SILVER, GOLD, PLATINUM):
        graded = grade_capability("s", properties)
        allowed = [graded.permits(ctx) for ctx in UseContext]
        assert allowed == sorted(allowed, reverse=True), (
            f"non-monotonic admission for {graded.tier.name}: {allowed}"
        )


# ── derivation from existing metadata (the consolidation) ──────────────────


def test_properties_are_derived_from_metadata_a_skill_already_declares():
    """A parallel maturity manifest would be a second source of truth, and the
    one nobody updates would be the one gating autonomous action."""
    from core.runtime.capability_maturity import derive_properties

    derived = derive_properties(
        input_model=object(),
        effect_scope="read_only",
        authority_class="observer",
    )

    assert "typed_request_schema" in derived      # a typed model IS a schema
    assert "idempotent_or_read_only" in derived   # read-only IS retry-safe
    assert "reversible_or_confirmed" in derived   # nothing to undo
    assert "authority_scoped" in derived


def test_only_claims_the_metadata_supports_are_inferred():
    """Bounded timeouts, postcondition verification, receipts and recovery have
    to be declared — inferring them would be exactly the unearned claim this
    module exists to prevent."""
    from core.runtime.capability_maturity import derive_properties

    derived = derive_properties(input_model=object(), effect_scope="read_only",
                                authority_class="observer")

    for unearned in ("bounded_timeout", "postcondition_verification",
                     "effect_receipt", "offline_or_failure_recovery",
                     "redacted_diagnostics", "integration_tests"):
        assert unearned not in derived


def test_an_external_io_skill_infers_no_retry_safety():
    """Reaching outside Aura is not idempotent by construction."""
    from core.runtime.capability_maturity import derive_properties

    derived = derive_properties(input_model=object(), effect_scope="external_io")

    assert "idempotent_or_read_only" not in derived
    assert "reversible_or_confirmed" not in derived


def test_unclassified_authority_is_not_authority_scoping():
    from core.runtime.capability_maturity import derive_properties

    assert "authority_scoped" not in derive_properties(
        authority_class="unclassified"
    )


# ── rollout discipline: the gate observes before it enforces ───────────────


def test_the_gate_defaults_to_observing_not_refusing(monkeypatch):
    """Shipping this enforcing onto a large ungraded surface would refuse most
    autonomous work on day one — which is how a safety mechanism gets switched
    off permanently instead of adopted."""
    import core.capability_engine as ce

    monkeypatch.delenv("AURA_ENFORCE_CAPABILITY_MATURITY", raising=False)

    assert ce._maturity_enforcement_enabled() is False


@pytest.mark.parametrize("value,expected", [
    ("1", True), ("true", True), ("YES", True), ("on", True),
    ("0", False), ("false", False), ("", False), ("maybe", False),
])
def test_enforcement_is_explicitly_opt_in(monkeypatch, value, expected):
    import core.capability_engine as ce

    monkeypatch.setenv("AURA_ENFORCE_CAPABILITY_MATURITY", value)

    assert ce._maturity_enforcement_enabled() is expected
