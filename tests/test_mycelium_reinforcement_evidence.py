"""Routing confidence must not rise on "nothing threw".

CP126 2462d3c5: reinforce() trained the routing ledger from a caller boolean
with no execution receipt and no independent outcome check. The live callers
made that concrete — the response lane passed success=True immediately after
a tool call that had merely not raised, so a tool RETURNING a failure still
strengthened the pathway that chose it.
"""
from __future__ import annotations

import pytest

from core.mycelium import HardwiredPathway, _evidence_verifies_outcome
from core.orchestrator.mixins.response_processing import _tool_result_succeeded


def _pathway():
    return HardwiredPathway(pathway_id="p1", pattern="x", skill_name="s")


# --- evidence, not assertion --------------------------------------------


@pytest.mark.parametrize("evidence", [None, True, False, "done", 42, {"text": "hi"}])
def test_a_claim_without_an_outcome_field_verifies_nothing(evidence):
    assert _evidence_verifies_outcome(evidence, True) is False


@pytest.mark.parametrize("key", ["ok", "success", "verified_success"])
def test_an_agreeing_outcome_field_verifies(key):
    assert _evidence_verifies_outcome({key: True}, True) is True
    assert _evidence_verifies_outcome({key: False}, False) is True


def test_evidence_contradicting_the_caller_is_not_evidence_for_them():
    assert _evidence_verifies_outcome({"ok": False}, True) is False
    assert _evidence_verifies_outcome({"ok": True}, False) is False


def test_an_object_with_an_outcome_attribute_verifies():
    class _Result:
        ok = True

    assert _evidence_verifies_outcome(_Result(), True) is True


# --- verified and asserted are counted apart ----------------------------


def test_an_unverified_success_earns_less_confidence():
    asserted, verified = _pathway(), _pathway()

    asserted.reinforce(True, verified=False)
    verified.reinforce(True, verified=True)

    assert verified.confidence > asserted.confidence


def test_a_failure_is_penalised_fully_either_way():
    """Discounting an unverified failure would keep a broken route alive."""
    asserted, verified = _pathway(), _pathway()

    asserted.reinforce(False, verified=False)
    verified.reinforce(False, verified=True)

    assert asserted.confidence == verified.confidence


def test_verified_outcomes_are_counted_separately():
    pathway = _pathway()

    pathway.reinforce(True, verified=True)
    pathway.reinforce(True, verified=False)
    pathway.reinforce(False, verified=True)

    assert pathway.hit_count == 2 and pathway.miss_count == 1
    assert pathway.verified_hits == 1 and pathway.verified_misses == 1
    assert pathway.unverified_reinforcements == 1


def test_no_verified_evidence_is_none_not_zero():
    pathway = _pathway()
    assert pathway.verified_success_rate is None

    pathway.reinforce(True, verified=False)
    assert pathway.verified_success_rate is None  # still nothing checked


def test_the_verified_rate_ignores_asserted_outcomes():
    pathway = _pathway()

    pathway.reinforce(True, verified=True)
    for _ in range(5):
        pathway.reinforce(False, verified=False)

    assert pathway.verified_success_rate == 1.0
    assert pathway.success_rate < 1.0


@pytest.mark.parametrize(
    "sequence,grade",
    [
        ([], "untested"),
        ([(True, False)], "asserted_only"),
        ([(True, True)], "verified"),
        ([(True, True), (True, False)], "mixed"),
    ],
)
def test_the_evidence_grade_reports_what_backs_the_record(sequence, grade):
    pathway = _pathway()
    for success, verified in sequence:
        pathway.reinforce(success, verified=verified)

    assert pathway.evidence_grade == grade


# --- the call site derives success from the result ----------------------


@pytest.mark.parametrize(
    "result,expected",
    [
        ({"ok": True}, True),
        ({"ok": False}, False),
        ({"success": False}, False),
        ({"error": "boom"}, False),
        ({"status": "failed"}, False),
        ({"status": "refused"}, False),
        ({"status": "ok"}, True),
        ({"text": "an answer"}, True),
        ("a bare string", True),
        (None, False),
        (False, False),
    ],
)
def test_a_returned_failure_is_not_a_success(result, expected):
    assert _tool_result_succeeded(result) is expected


def test_the_call_site_no_longer_hardcodes_success():
    import inspect

    from core.orchestrator.mixins import response_processing

    source = inspect.getsource(response_processing)
    assert "mycelium.reinforce(pw.pathway_id, success=True)" not in source
    assert "_tool_result_succeeded(result)" in source


# ── Phenomenal telemetry is not evidence about a route (CP126 d926886e) ──


class _Qualia:
    def __init__(self, q_norm):
        self.q_norm = q_norm


class _Experiencer:
    def __init__(self, arousal):
        self.current_arousal = arousal


@pytest.fixture()
def network():
    from core.mycelium import MycelialNetwork

    MycelialNetwork._instance = None
    MycelialNetwork._initialized = False
    net = MycelialNetwork()
    net.register_pathway(pathway_id="p", pattern=r"go", skill_name="s")
    yield net
    MycelialNetwork._instance = None
    MycelialNetwork._initialized = False


@pytest.fixture()
def resonating(monkeypatch):
    """Install a high-intensity qualia reading the routing lane can see."""
    def _install(q_norm=0.9, arousal=0.8):
        from core.container import ServiceContainer

        services = {
            "qualia_synthesizer": _Qualia(q_norm),
            "phenomenological_experiencer": _Experiencer(arousal),
        }
        monkeypatch.setattr(
            ServiceContainer,
            "get",
            classmethod(lambda cls, name, default=None: services.get(name, default)),
        )
    return _install


def test_qualia_cannot_write_durable_confidence(network, resonating):
    """The block wrote pw.confidence directly, straight past the evidence gate
    the reinforcement path had just applied."""
    resonating()
    before = network.pathways["p"].confidence

    network.reinforce("p", success=True)  # asserted only — not durable

    assert network.pathways["p"].confidence == before


def test_qualia_still_moves_the_session_view(network, resonating):
    resonating()
    before = network.effective_confidence("p")

    network.reinforce("p", success=True)

    assert network.effective_confidence("p") > before


def test_qualia_cannot_exceed_the_declared_confidence_ceiling(network, resonating):
    """The old clamp was min(10.0, ...) against a declared ceiling of 1.0."""
    from core.mycelium import HardwiredPathway

    resonating(q_norm=1.0, arousal=1.0)
    for _ in range(200):
        network.reinforce("p", success=True)

    assert network.pathways["p"].confidence <= HardwiredPathway.MAX_CONFIDENCE
    assert network.effective_confidence("p") <= HardwiredPathway.MAX_CONFIDENCE


def _unweighted_delta(network):
    """The ordinary unverified session step, with no qualia weighting at all.

    The comparison has to be against this rather than against "did not move":
    an unverified success moves the session view on its own, and the question
    here is only whether a bad phenomenal reading added anything on top.
    """
    before = network.effective_confidence("p")
    network.reinforce("p", success=True)
    return network.effective_confidence("p") - before


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -1.0, 42.0, "0.9", None])
def test_an_out_of_range_qualia_reading_is_not_acted_on(network, resonating, bad):
    control = _unweighted_delta(network)  # no qualia service installed yet
    resonating(q_norm=bad)

    assert _unweighted_delta(network) == pytest.approx(control)


def test_unmeasured_arousal_is_not_treated_as_half(network, resonating):
    """The old code defaulted a missing arousal to 0.5 and weighted with it."""
    control = _unweighted_delta(network)
    resonating(q_norm=0.9, arousal=None)

    assert _unweighted_delta(network) == pytest.approx(control)


def test_a_valid_reading_does_add_on_top_of_the_ordinary_step(network, resonating):
    """The guard must not have simply disabled the feature."""
    control = _unweighted_delta(network)
    resonating(q_norm=0.9, arousal=0.8)

    assert _unweighted_delta(network) > control


def test_the_unit_reading_guard_rejects_what_it_should():
    from core.mycelium import _unit_reading

    assert _unit_reading(0.0) == 0.0
    assert _unit_reading(1.0) == 1.0
    assert _unit_reading(0.5) == 0.5
    for bad in (None, True, False, "0.5", float("nan"), float("inf"), -0.1, 1.1):
        assert _unit_reading(bad) is None
