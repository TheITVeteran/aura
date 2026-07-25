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


class TestHomeostaticSetPoint:
    """A fixed decay constant can only be right for one workload.

    Measured live 2026-07-25 with the charge ledger: every charge was a
    legitimate Will-authorised cost, they arrived faster than 0.01/s could
    repay, and fatigue sat at 0.96 with recovery_debt at 0.9999 — 0.69 of the
    welfare recovery drive before distress, on a runtime with nothing wrong.
    Three retunes of the constant preceded this; none of them could have
    worked, because the workload is a variable and the constant is not.

    Recovery now tracks the observed charge rate, so any steady workload
    settles near the set-point and a SURGE above her own recent normal still
    signals. That is what makes the number mean something.
    """

    def _work(self, service, *, per_second: float, seconds: float, step: float = 1.0):
        """Charge at a steady rate for a while, decaying between charges."""
        for _ in range(int(seconds / step)):
            service._commit_cost_locked(
                {"fatigue": per_second * step, "integrity_risk": per_second * step},
                receipt_id=f"will-{time.time_ns()}",
            )
            _rest(service, step)

    def test_a_heavy_steady_workload_settles_not_saturates(self, service):
        """The live rate: ~0.02 per action, several actions a second."""
        self._work(service, per_second=0.05, seconds=600)
        assert service._metabolic.fatigue < 0.75, (
            f"fatigue pinned at {service._metabolic.fatigue:.3f} under a steady "
            "load — a saturated signal cannot report anything"
        )
        assert service._metabolic.recovery_debt < 0.75

    def test_the_welfare_drive_stays_under_the_defer_threshold(self, service):
        """The whole point: her own work must not be deferred while working."""
        self._work(service, per_second=0.05, seconds=600)
        owned = (
            service._metabolic.recovery_debt * 0.4
            + service._metabolic.fatigue * 0.3
        )
        assert owned < 0.5, (
            f"debt+fatigue contribute {owned:.3f}; with any distress at all "
            "that clears the Will's 0.6 defer threshold and the storm returns"
        )

    def test_a_light_workload_stays_low(self, service):
        self._work(service, per_second=0.002, seconds=300)
        assert service._metabolic.fatigue < 0.3

    def test_a_surge_above_her_own_normal_still_registers(self, service):
        """Adaptation must not become numbness."""
        self._work(service, per_second=0.01, seconds=600)   # establish normal
        settled = service._metabolic.fatigue
        for _ in range(30):                                  # sudden burst
            service._commit_cost_locked(
                {"fatigue": 0.05}, receipt_id=f"burst-{time.time_ns()}"
            )
        assert service._metabolic.fatigue > settled, (
            "a burst well above the established rate must still tire her"
        )

    def test_quiet_after_work_still_recovers_fully(self, service):
        self._work(service, per_second=0.05, seconds=300)
        _rest(service, 1800.0)
        assert service._metabolic.fatigue < 0.05
        assert service._metabolic.recovery_debt < 0.05

    def test_the_rate_estimate_forgets(self, service):
        self._work(service, per_second=0.05, seconds=120)
        assert service._fatigue_charge_rate > 0.0
        _rest(service, 3000.0)
        assert service._fatigue_charge_rate < 1e-4, (
            "an estimate that never forgets makes past work permanent"
        )


class TestDebtClearsInAQuietMinute:
    """Debt was the term that kept the defer storm alive after fatigue settled.

    Measured on the live code 2026-07-25: 667 seconds to clear from saturation
    once charges stop. Debt carries welfare weight 0.4 against fatigue's 0.3
    and had a base decay rate ten times slower, so a quiet runtime spent
    eleven minutes deferring its own belief updates and memory writes for work
    it had already finished paying for.
    """

    def test_saturated_debt_clears_in_about_two_quiet_minutes(self, service):
        service._metabolic.recovery_debt = 1.0
        _rest(service, 150.0)
        assert service._metabolic.recovery_debt < 0.05

    def test_the_welfare_terms_fall_under_the_threshold_quickly(self, service):
        service._metabolic.recovery_debt = 1.0
        service._metabolic.fatigue = 1.0
        _rest(service, 90.0)
        owned = (
            service._metabolic.recovery_debt * 0.4
            + service._metabolic.fatigue * 0.3
        )
        assert owned < 0.2, (
            f"after ninety quiet seconds the terms still contribute {owned:.3f}"
        )

    def test_debt_still_accrues_faster_than_it_clears(self, service):
        """Recovery must not outrun genuine integrity risk."""
        service._metabolic.recovery_debt = 0.0
        service._last_decay_time = time.monotonic()
        for i in range(10):
            service._commit_cost_locked(
                {"integrity_risk": 0.05}, receipt_id=f"risk-{i}"
            )
        assert service._metabolic.recovery_debt > 0.4


class TestTheEstimatorTracksTheLoad:
    """An estimator slower than the thing it regulates cannot regulate it.

    Measured 2026-07-25: a live-shaped load (40 charges of 0.013 over 20s =
    0.026/s) drove recovery_debt from 0.91 to 0.996 while the 300-second
    estimator read 0.0017/s — fifteen times low, because it was only 4% of the
    way to steady state. The set-point machinery was correct and silently
    inert, because its input never left the floor.

    With the window matched to the ~40s saturation time, a sustained load from
    full saturation settles at debt 0.32 / fatigue 0.34 — welfare terms 0.23
    against the Will's 0.6 threshold.
    """

    def test_the_rate_estimate_converges_on_the_real_rate(self, service):
        """rate += a/tau with rate *= exp(-dt/tau) settles at exactly n*a."""
        per_charge, interval = 0.02, 0.25
        expected = per_charge / interval          # units per second
        for _ in range(int(240 / interval)):      # well past the time constant
            service._observe_charge("fatigue", per_charge)
            service._decay_charge_rates(interval)
        assert service._fatigue_charge_rate == pytest.approx(expected, rel=0.15), (
            f"estimator reads {service._fatigue_charge_rate:.4f} for a real "
            f"{expected:.4f}/s load; a mis-scaled estimate makes the set-point inert"
        )

    def test_the_window_is_of_the_order_of_the_saturation_time(self, service):
        assert service._CHARGE_RATE_HALF_LIFE_S <= 60.0, (
            "debt saturates in about forty seconds under load; an estimator "
            "slower than that cannot respond in time"
        )

    def test_a_sustained_load_from_saturation_reaches_the_set_point(self, service):
        service._metabolic.recovery_debt = 0.95
        service._metabolic.fatigue = 0.95
        for _ in range(400):                      # 200s at 0.5s intervals
            service._commit_cost_locked(
                {"integrity_risk": 0.013, "fatigue": 0.02},
                receipt_id=f"will-{time.time_ns()}",
            )
            _rest(service, 0.5)
        owned = (
            service._metabolic.recovery_debt * 0.4
            + service._metabolic.fatigue * 0.3
        )
        assert owned < 0.35, (
            f"welfare terms still contribute {owned:.3f} under a steady load; "
            "the defer storm returns above ~0.4"
        )
