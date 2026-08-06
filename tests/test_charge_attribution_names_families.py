"""An attribution ledger keyed by instance cannot attribute anything.

2026-07-30 demo, live neural feed, verbatim:

    Body metabolic saturation: fatigue 0.0620, recovery_debt 0.9684.
    Charges so far by source: {"fatigue": {'will_d6f534a1c077': 0.0294,
    'will_7c62f0b8bb87': 0.0294, 'will_047a99558973': 0.0294, ...

Sixteen entries, all the same value, because they were all the same caller
counted under sixteen names. The reduction handled "family:instance" and
real receipt ids are "family_instance".
"""

from __future__ import annotations

from core.being.body_state_service import BodyStateService


class TestFamilyReduction:
    def test_the_demo_ids_collapse_to_one_family(self) -> None:
        observed = {
            BodyStateService._charge_source(rid)
            for rid in (
                "will_d6f534a1c077",
                "will_7c62f0b8bb87",
                "will_047a99558973",
                "will_8aa4b4bc669f",
            )
        }
        assert observed == {"will"}

    def test_the_colon_form_still_works(self) -> None:
        assert BodyStateService._charge_source("cognition:step-4") == "cognition"

    def test_a_multi_word_family_is_kept_whole(self) -> None:
        assert BodyStateService._charge_source("mind_tick_ab12cd34ef") == "mind_tick"

    def test_a_family_with_no_instance_is_unchanged(self) -> None:
        assert BodyStateService._charge_source("plain_source") == "plain_source"

    def test_a_bare_hex_id_reports_no_family_rather_than_inventing_one(self) -> None:
        for blob in ("830a35775a9321f3", "5580a24ddb02025c"):
            assert BodyStateService._charge_source(blob) == "unattributed"

    def test_empty_input_is_unattributed(self) -> None:
        assert BodyStateService._charge_source("") == "unattributed"
        assert BodyStateService._charge_source(None) == "unattributed"

    def test_the_family_is_bounded(self) -> None:
        assert len(BodyStateService._charge_source("x" * 500)) <= 48


class TestTheLedgerNowAggregates:
    def test_repeated_calls_from_one_family_accumulate(self) -> None:
        """The point of the fix: one caller, one row, a total worth reading."""
        service = BodyStateService.__new__(BodyStateService)
        ledger: dict[str, float] = {}
        service._CHARGE_LEDGER_CAP = 16
        for index in range(20):
            BodyStateService._note_charge(
                service, ledger, f"will_{index:012x}", 0.0294
            )
        assert list(ledger) == ["will"]
        assert ledger["will"] == pytest_approx(20 * 0.0294)


def pytest_approx(value: float, tolerance: float = 1e-9):
    class _Approx:
        def __eq__(self, other: object) -> bool:
            return abs(float(other) - value) <= tolerance  # type: ignore[arg-type]

        def __repr__(self) -> str:  # pragma: no cover - failure output only
            return f"~{value}"

    return _Approx()
