"""Rest must actually restore (2026-07-25 idle window).

A clean idle hour — zero failures, zero criticals, nothing charging a
consequence — still showed ``body_fatigue`` pinned at 0.996. Saturated
fatigue holds ``welfare.recovery_drive`` above the Will's defer threshold,
so the log carried a standing
``welfare_recovery_required_before_action`` storm: her own belief updates,
memory writes and initiative deferred during a completely quiet stretch.

The mechanism is that a FLAT decay can be exactly cancelled by the
ordinary drip of idle-loop body costs. That is the same class the earlier
0.002 → 0.01 retune addressed, and a constant only postpones it. Recovery
now scales with fatigue, so deep fatigue always escapes saturation while
ordinary dynamics are untouched.
"""
from __future__ import annotations

import time

import pytest

from core.being.body_state_service import BodyStateService

pytestmark = pytest.mark.unit


@pytest.fixture()
def service():
    BodyStateService.reset()
    svc = BodyStateService.get()
    yield svc
    BodyStateService.reset()


def _rest(svc: BodyStateService, seconds: float) -> float:
    svc._last_decay_time = time.monotonic() - seconds
    return svc.snapshot().fatigue


class TestRestRestores:
    def test_saturated_fatigue_recovers_in_a_quiet_minute(self, service):
        service._metabolic.fatigue = 1.0
        assert _rest(service, 60.0) < 0.4, "a quiet minute must make real progress"

    def test_saturation_escapes_a_constant_idle_drip(self, service):
        """Replay the observed shape: fatigue at 0.996 while an idle drip
        charges at exactly the old flat decay rate. Recovery must still win."""
        service._metabolic.fatigue = 0.996
        drip = service._fatigue_decay_rate  # per second, cancels a flat rate

        for _ in range(30):  # 30 x 2s of quiet cognition
            service._metabolic.fatigue = min(
                1.0, service._metabolic.fatigue + drip * 2.0
            )
            _rest(service, 2.0)

        assert service._metabolic.fatigue < 0.9, (
            "proportional recovery must beat a constant drip, or fatigue is a "
            "ratchet and her maintenance work is deferred forever"
        )

    def test_low_fatigue_dynamics_are_not_distorted(self, service):
        """The fix must not make her implausibly tireless at low fatigue."""
        service._metabolic.fatigue = 0.1
        after = _rest(service, 10.0)
        assert 0.0 <= after < 0.1
        assert after > 0.0, "a brief rest should not erase all light fatigue"

    def test_fatigue_still_accumulates_under_real_load(self, service):
        """Recovery is not immunity: sustained work still tires her."""
        service._metabolic.fatigue = 0.0
        service._last_decay_time = time.monotonic()
        for _ in range(40):
            service._commit_cost_locked(
                {"fatigue": 0.02}, receipt_id=f"work-{_}"
            )
        assert service._metabolic.fatigue > 0.5

    def test_recovery_never_drives_fatigue_negative(self, service):
        service._metabolic.fatigue = 0.05
        assert _rest(service, 600.0) == 0.0


class TestRecoveryDebtIsPaidDown:
    """Debt is the same ratchet and the LARGER welfare term (0.4 vs 0.3)."""

    def test_saturated_debt_escapes_a_constant_drip(self, service):
        service._metabolic.recovery_debt = 1.0
        drip = service._recovery_decay_rate

        for _ in range(60):  # 60 x 5s of quiet cognition charging debt
            service._metabolic.recovery_debt = min(
                1.0, service._metabolic.recovery_debt + drip * 5.0
            )
            _rest(service, 5.0)

        assert service._metabolic.recovery_debt < 0.9

    def test_the_pair_together_clears_the_will_defer_threshold(self, service):
        """The live shape: both saturated, a quiet stretch must drop
        welfare.recovery_drive under the 0.6 the Will defers above."""
        service._metabolic.fatigue = 0.996
        service._metabolic.recovery_debt = 1.0

        _rest(service, 900.0)  # fifteen quiet minutes

        # recovery_drive = debt*0.4 + fatigue*0.3 + distress*0.2 + (1-ri)*0.15;
        # the two terms this service owns must no longer carry it over 0.6.
        owned = (
            service._metabolic.recovery_debt * 0.4
            + service._metabolic.fatigue * 0.3
        )
        assert owned < 0.20, (
            f"debt+fatigue still contribute {owned:.3f} after fifteen quiet "
            "minutes; the defer storm would persist"
        )

    def test_debt_still_accrues_from_real_consequences(self, service):
        service._metabolic.recovery_debt = 0.0
        service._last_decay_time = time.monotonic()
        for i in range(20):
            service._commit_cost_locked(
                {"integrity_risk": 0.05}, receipt_id=f"cost-{i}"
            )
        assert service._metabolic.recovery_debt > 0.3

    def test_debt_never_goes_negative(self, service):
        service._metabolic.recovery_debt = 0.02
        _rest(service, 3000.0)
        assert service._metabolic.recovery_debt == 0.0


class TestChargeAttribution:
    """Recovery is provable in isolation; the live value stayed pinned anyway.

    The 2026-07-25 verification run held ``body_fatigue`` at 0.99 through a
    silent idle window with ``body_cost_applied: {}`` on every Will receipt —
    something charges fatigue that the Will never quoted, and no surface in the
    system could name it. That is the actual open question, so it gets a
    measurement rather than another guess.
    """

    def test_a_charge_is_attributed_to_its_caller(self, service):
        service._commit_cost_locked({"fatigue": 0.05}, receipt_id="mind_tick:abc123")
        assert service.charge_attribution()["fatigue"] == {"mind_tick": 0.05}

    def test_charges_from_one_source_accumulate(self, service):
        for i in range(4):
            service._commit_cost_locked({"fatigue": 0.01}, receipt_id=f"will:{i}")
        assert service.charge_attribution()["fatigue"]["will"] == pytest.approx(0.04)

    def test_debt_charges_are_tracked_separately(self, service):
        service._commit_cost_locked(
            {"integrity_risk": 0.03}, receipt_id="repair_loop:x"
        )
        attribution = service.charge_attribution()
        assert attribution["recovery_debt"] == {"repair_loop": 0.03}
        assert attribution["fatigue"] == {}

    def test_an_unlabelled_charge_is_still_counted(self, service):
        service._commit_cost_locked({"fatigue": 0.02}, receipt_id="")
        assert "unattributed" in service.charge_attribution()["fatigue"]

    def test_the_heaviest_sources_survive_the_cap(self, service):
        service._commit_cost_locked({"fatigue": 0.9}, receipt_id="heavy:1")
        for i in range(service._CHARGE_LEDGER_CAP * 3):
            service._commit_cost_locked({"fatigue": 0.001}, receipt_id=f"light{i}:x")
        assert len(service._fatigue_charges) <= service._CHARGE_LEDGER_CAP
        assert "heavy" in service.charge_attribution()["fatigue"]

    def test_relief_is_not_recorded_as_a_charge(self, service):
        service._commit_cost_locked({"fatigue": -0.05}, receipt_id="stabilization:1")
        assert service.charge_attribution()["fatigue"] == {}


class TestAnObservationIsNotACharge:
    """The feedback loop that pinned live fatigue at 1.0 for whole sessions.

    WelfareTransaction publishes ``actual_body_cost`` as ``b_delta`` — the
    OBSERVED change in body state across the transaction, i.e. fatigue that was
    already charged. ``_on_consequence`` fell back to the event's unique id when
    no Will receipt was present, so every such publication charged that observed
    fatigue AGAIN: measured fatigue became new fatigue, which enlarged the next
    transaction's delta, which charged more.

    That is why proportional recovery could not move the live number. The decay
    was never the binding constraint — the loop tracked it.
    """

    def _consequence(self, *, receipt: str, fatigue: float = 0.2):
        from core.runtime.consequence_bus import ConsequenceEvent

        return ConsequenceEvent(
            event_id=f"evt-{receipt or 'none'}-{fatigue}",
            timestamp=time.time(),
            source="welfare_transaction",
            domain="response",
            action_content="a completed turn",
            actual_outcome="success",
            actual_body_cost={"fatigue": fatigue},
            will_receipt_id=receipt,
        )

    def test_an_unauthorized_observation_charges_nothing(self, service):
        service._metabolic.fatigue = 0.0
        service._on_consequence(self._consequence(receipt=""))
        assert service._metabolic.fatigue == 0.0

    def test_repeated_observations_cannot_ratchet_fatigue(self, service):
        """The live shape: the same measured delta republished many times."""
        service._metabolic.fatigue = 0.3
        for _ in range(40):
            service._on_consequence(self._consequence(receipt="", fatigue=0.2))
        assert service._metabolic.fatigue <= 0.3, (
            "an observed delta re-charged as a cost is a feedback loop that "
            "pins fatigue at saturation forever"
        )

    def test_an_authorized_cost_is_still_charged_once(self, service):
        service._metabolic.fatigue = 0.0
        service._on_consequence(self._consequence(receipt="will-abc"))
        assert service._metabolic.fatigue == pytest.approx(0.2)

    def test_an_authorized_cost_is_idempotent(self, service):
        service._metabolic.fatigue = 0.0
        for _ in range(10):
            service._on_consequence(self._consequence(receipt="will-abc"))
        assert service._metabolic.fatigue == pytest.approx(0.2)

    def test_failure_consequences_still_register_their_toll(self, service):
        """Removing the double-charge must not make failures free."""
        from core.runtime.consequence_bus import ConsequenceEvent

        service._metabolic.fatigue = 0.0
        service._on_consequence(
            ConsequenceEvent(
                event_id="evt-fail",
                timestamp=time.time(),
                source="welfare_transaction",
                domain="response",
                action_content="a failed turn",
                actual_outcome="failure",
                recovery_required=0.4,
            )
        )
        assert service._metabolic.fatigue > 0.0
        assert service._metabolic.recovery_debt > 0.0


class TestARefusalCostsNothing:
    """The charge the ledger exposed — and the loop it closed.

    The final verification run showed only 0.175 of fatigue ever charged
    through the cost path while ``body_fatigue`` sat at 0.985. The charger was
    ``_on_consequence``: a bare ``fatigue += 0.02`` on every "failure"
    consequence, outside every receipt and every ledger.

    When the Will DEFERS an action on welfare grounds, the welfare transaction
    still completes as a "failure". So the defer storm charged fatigue, which
    raised recovery_drive, which caused more defers. Declining to act spends
    nothing; charging for it is how a tired system talks itself into staying
    tired.
    """

    def _event(self, *, error="", action="did a thing", outcome="failure"):
        from core.runtime.consequence_bus import ConsequenceEvent

        return ConsequenceEvent(
            event_id=f"evt-{error or action}-{time.time_ns()}",
            timestamp=time.time(),
            source="welfare_transaction",
            domain="initiative",
            action_content=action,
            actual_outcome=outcome,
            recovery_required=0.3,
            error=error,
        )

    def test_a_deferred_action_charges_nothing(self, service):
        service._metabolic.fatigue = 0.0
        service._metabolic.recovery_debt = 0.0
        for _ in range(50):
            service._on_consequence(
                self._event(error="aura_now_defer: requires stabilization first")
            )
        assert service._metabolic.fatigue == 0.0
        assert service._metabolic.recovery_debt == 0.0

    @pytest.mark.parametrize(
        "reason",
        [
            "welfare_recovery_required_before_action",
            "model_load_admission_denied",
            "queued until admission clears",
            "SubstrateAuthority blocked the mutation",
        ],
    )
    def test_every_refusal_shape_is_free(self, service, reason):
        service._metabolic.fatigue = 0.0
        service._on_consequence(self._event(error=reason))
        assert service._metabolic.fatigue == 0.0

    def test_work_that_actually_failed_still_costs(self, service):
        service._metabolic.fatigue = 0.0
        service._on_consequence(
            self._event(error="ConnectionError: the endpoint went away")
        )
        assert service._metabolic.fatigue == pytest.approx(0.02)
        assert service._metabolic.recovery_debt > 0.0

    def test_a_real_failure_is_now_in_the_ledger(self, service):
        """It used to be invisible, which is why it took a day to find."""
        service._on_consequence(self._event(error="TimeoutError: worker gone"))
        assert service.charge_attribution()["fatigue"], (
            "an unledgered charge is an unfindable one"
        )

    def test_a_republished_failure_charges_once(self, service):
        from core.runtime.consequence_bus import ConsequenceEvent

        service._metabolic.fatigue = 0.0
        for _ in range(10):
            service._on_consequence(
                ConsequenceEvent(
                    event_id="evt-stable",
                    timestamp=time.time(),
                    source="welfare_transaction",
                    domain="initiative",
                    action_content="one failed action",
                    actual_outcome="failure",
                    recovery_required=0.3,
                    error="ValueError: bad input",
                )
            )
        assert service._metabolic.fatigue == pytest.approx(0.02)
