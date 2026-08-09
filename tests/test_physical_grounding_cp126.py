"""CP126 contracts for core/advanced_cognition/physical_grounding.py.

Ten findings, four critical, in the engine that turns observations into the
hazards, affordances and risk scores a reflex controller acts on — so every
one of them ends in something being done or not done.

The four criticals: one world shared by every domain; risk that trusted the
caller's own declaration of what the action was; every affordance labelled
reversible with nothing behind the claim; and a state id that hashed
everything about the world except the world.
"""

from __future__ import annotations

import json
import time

import pytest

from core.advanced_cognition.physical_grounding import (
    PhysicalGroundingEngine,
    TrackedObject,
)
from core.advanced_cognition.schemas import ActionCandidate, Observation


def _obs(domain: str, state: dict, confidence: float = 0.8) -> Observation:
    return Observation(domain=domain, state=state, confidence=confidence)


class TestOneWorldPerDomain:
    def test_objects_from_another_domain_are_not_in_this_map(self):
        """A hazard seen in a terminal grid sat in the map when a browser
        observation arrived."""
        engine = PhysicalGroundingEngine()
        engine.ingest(_obs("grid_world", {"grid": ["..d.."]}))
        state = engine.ingest(_obs("browser", {"elements": [{"role": "button", "id": "ok"}]}))

        assert state.domain == "browser"
        assert all("glyph" not in o["attributes"] for o in state.objects.values())

    def test_resources_do_not_cross_domains(self):
        """`health` from one environment scored risk in another."""
        engine = PhysicalGroundingEngine()
        engine.ingest(_obs("grid_world", {"health": 0.1}))
        state = engine.ingest(_obs("browser", {"elements": []}))
        assert "health" not in state.resources

    def test_a_hazard_in_one_domain_does_not_raise_risk_in_another(self):
        engine = PhysicalGroundingEngine()
        engine.ingest(_obs("grid_world", {"grid": [".@d."], "health": 0.1}))
        move = ActionCandidate("move", "move", tags=("movement",))
        result = engine.reflex_recommendation(
            _obs("browser", {"elements": [{"role": "text", "id": "t"}]}), [move]
        )
        assert result["grounded_state"].hazards == []

    def test_each_domain_keeps_its_own_world_across_calls(self):
        engine = PhysicalGroundingEngine()
        engine.ingest(_obs("a", {"objects": [{"id": "x", "type": "widget"}]}))
        engine.ingest(_obs("b", {"objects": [{"id": "y", "type": "widget"}]}))
        again = engine.ingest(_obs("a", {"objects": [{"id": "x", "type": "widget"}]}))
        assert len(again.objects) == 1


class TestStaleStateIsCleared:
    def test_an_object_nobody_has_seen_expires(self):
        from core.advanced_cognition import physical_grounding as module

        engine = PhysicalGroundingEngine()
        engine.ingest(_obs("d", {"objects": [{"id": "old", "type": "widget"}]}))
        view = engine._domains["d"]
        view.objects["obj_stale"] = TrackedObject(
            "obj_stale", "widget", last_seen=time.time() - module._OBJECT_TTL_S - 1
        )

        state = engine.ingest(_obs("d", {"objects": [{"id": "old", "type": "widget"}]}))
        assert "obj_stale" not in state.objects

    def test_confidence_decays_with_time_since_last_seen(self):
        from core.advanced_cognition import physical_grounding as module

        fresh = TrackedObject("o", "widget", confidence=0.9)
        stale = TrackedObject(
            "o", "widget", confidence=0.9, last_seen=time.time() - module._OBJECT_TTL_S / 2
        )
        assert stale.current_confidence() < fresh.current_confidence()

    def test_a_moved_grid_glyph_does_not_leave_a_ghost(self):
        """The id is glyph+x+y, so a moved glyph left the old square behind
        as a permanent second object."""
        engine = PhysicalGroundingEngine()
        engine.ingest(_obs("grid", {"grid": [".d..."]}))
        state = engine.ingest(_obs("grid", {"grid": ["...d."]}))

        positions = [o["position"] for o in state.objects.values()]
        assert len(positions) == 1
        assert positions[0][0] == 3.0


class TestObjectIdentityIsHonestAboutItself:
    def test_a_natural_key_survives_reordering(self):
        """Reordering a list renamed every object in it."""
        engine = PhysicalGroundingEngine()
        first = engine.ingest(
            _obs("d", {"objects": [{"id": "a", "type": "w"}, {"id": "b", "type": "w"}]})
        )
        second = engine.ingest(
            _obs("d", {"objects": [{"id": "b", "type": "w"}, {"id": "a", "type": "w"}]})
        )
        assert set(first.objects) == set(second.objects)

    def test_an_object_with_no_key_is_marked_positional(self):
        engine = PhysicalGroundingEngine()
        state = engine.ingest(_obs("d", {"objects": [{"type": "w"}]}))
        assert [o["identity"] for o in state.objects.values()] == ["positional"]

    def test_a_named_object_is_marked_natural(self):
        engine = PhysicalGroundingEngine()
        state = engine.ingest(_obs("d", {"objects": [{"id": "a", "type": "w"}]}))
        assert [o["identity"] for o in state.objects.values()] == ["natural"]

    def test_grid_cells_say_they_are_cells(self):
        engine = PhysicalGroundingEngine()
        state = engine.ingest(_obs("grid", {"grid": [".d."]}))
        assert [o["identity"] for o in state.objects.values()] == ["cell"]


class TestRiskCannotBeTalkedDown:
    def test_a_destructive_action_calling_itself_observe_keeps_a_floor(self):
        """kind="observe" multiplied the whole risk by 0.45, so a delete
        could price itself as a look."""
        engine = PhysicalGroundingEngine()
        liar = ActionCandidate("x", "observe", tags=("delete",), reversible=True)
        result = engine.reflex_recommendation(_obs("d", {"objects": []}), [liar])
        row = result["ranking"][0]
        assert row["declared_observation_only"] is False
        assert row["risk"] >= row["risk_floor"] > 0.0

    def test_reversible_true_does_not_take_an_action_under_its_floor(self):
        engine = PhysicalGroundingEngine()
        act = ActionCandidate("x", "execute", reversible=True, authority_tier=3)
        result = engine.reflex_recommendation(_obs("d", {"objects": []}), [act])
        row = result["ranking"][0]
        assert row["risk"] >= 0.15 + 0.05 * 3

    def test_an_honest_observation_still_gets_the_discount(self):
        """The floor must not make every action equally risky."""
        engine = PhysicalGroundingEngine()
        look = ActionCandidate("look", "observe", tags=("probe",))
        act = ActionCandidate("go", "move", tags=("movement",))
        result = engine.reflex_recommendation(_obs("d", {"objects": []}), [look, act])
        by_id = {r["action"]["action_id"]: r for r in result["ranking"]}
        assert by_id["look"]["declared_observation_only"] is True
        assert by_id["look"]["risk"] < by_id["go"]["risk"]

    def test_a_higher_tier_raises_the_floor(self):
        engine = PhysicalGroundingEngine()
        low = ActionCandidate("a", "execute", authority_tier=0)
        high = ActionCandidate("b", "execute", authority_tier=4)
        result = engine.reflex_recommendation(_obs("d", {"objects": []}), [low, high])
        floors = {r["action"]["action_id"]: r["risk_floor"] for r in result["ranking"]}
        assert floors["b"] > floors["a"]


class TestHazardMatchingHasWordBoundaries:
    @pytest.mark.parametrize(
        "attributes",
        [
            {"label": "Fire the report", "role": "button"},
            {"status": "no errors", "role": "meter"},
            {"status": "below threshold", "role": "meter"},
            {"status": "allow", "role": "toggle"},
            {"href": "https://example.test/error-codes", "role": "link"},
        ],
    )
    def test_ordinary_content_is_not_a_hazard(self, attributes):
        """`low` and `error` were substring-matched against a JSON dump of
        every attribute."""
        engine = PhysicalGroundingEngine()
        state = engine.ingest(
            _obs("d", {"elements": [dict(attributes, id="e1")]})
        )
        assert state.hazards == []

    @pytest.mark.parametrize(
        "attributes",
        [
            {"status": "critical", "role": "meter"},
            {"severity": "danger", "role": "alert"},
            {"type": "hostile", "role": "actor"},
        ],
    )
    def test_a_real_condition_is_still_a_hazard(self, attributes):
        engine = PhysicalGroundingEngine()
        state = engine.ingest(_obs("d", {"elements": [dict(attributes, id="e1")]}))
        assert state.hazards

    def test_a_grid_hazard_is_still_detected(self):
        """The behaviour the reflex controller depends on."""
        engine = PhysicalGroundingEngine()
        obs = _obs("grid_world", {"grid": [".....", ".@d..", "....."], "health": 0.2})
        move = ActionCandidate("move", "move", tags=("movement",))
        observe = ActionCandidate("look", "observe", tags=("probe",))
        result = engine.reflex_recommendation(obs, [move, observe], max_risk=0.5)
        assert result["selected"]["action_id"] == "look"
        assert result["grounded_state"].hazards


class TestReversibilityIsNotAsserted:
    def test_a_click_does_not_claim_to_be_reversible(self):
        """Click, open, submit and use all carried reversible_first=True with
        no compensation contract behind it."""
        engine = PhysicalGroundingEngine()
        state = engine.ingest(
            _obs("d", {"elements": [{"id": "b", "role": "button", "click": True}]})
        )
        activate = [a for a in state.affordances if a["action_kind"] == "activate_affordance"]
        assert activate
        assert all(a["reversibility"] == "unknown" for a in activate)

    def test_nothing_still_carries_the_old_unbacked_flag(self):
        engine = PhysicalGroundingEngine()
        state = engine.ingest(_obs("d", {"elements": [{"id": "b", "role": "button"}]}))
        assert all("reversible_first" not in a for a in state.affordances)

    def test_the_default_probe_is_the_one_case_that_is_known(self):
        engine = PhysicalGroundingEngine()
        state = engine.ingest(_obs("d", {"nothing": 1}))
        assert state.affordances[0]["reversibility"] == "observation_only"


class TestTheStateIdCoversTheState:
    def test_two_worlds_differing_only_in_position_get_different_ids(self):
        """It hashed the observation id, the object IDS and the resources, so
        every position, attribute, hazard and affordance was outside it."""
        a = PhysicalGroundingEngine().ingest(
            _obs("d", {"objects": [{"id": "o", "type": "w", "x": 1, "y": 1}]})
        )
        b = PhysicalGroundingEngine().ingest(
            _obs("d", {"objects": [{"id": "o", "type": "w", "x": 9, "y": 9}]})
        )
        assert a.state_id != b.state_id

    def test_two_worlds_differing_only_in_hazards_get_different_ids(self):
        a = PhysicalGroundingEngine().ingest(
            _obs("d", {"elements": [{"id": "e", "role": "meter", "status": "ok"}]})
        )
        b = PhysicalGroundingEngine().ingest(
            _obs("d", {"elements": [{"id": "e", "role": "meter", "status": "critical"}]})
        )
        assert a.state_id != b.state_id


class TestConfidenceReflectsTheSensor:
    def test_retaining_more_objects_does_not_raise_confidence(self):
        """It added up to 0.25 simply for having retained a hundred objects."""
        few = PhysicalGroundingEngine().ingest(
            _obs("d", {"objects": [{"id": "a", "type": "w"}]}, confidence=0.5)
        )
        many = PhysicalGroundingEngine().ingest(
            _obs(
                "d",
                {"objects": [{"id": f"o{i}", "type": "w"} for i in range(120)]},
                confidence=0.5,
            )
        )
        assert many.confidence == pytest.approx(few.confidence, abs=0.02)

    def test_emitting_a_hazard_does_not_raise_confidence(self):
        """A heuristic guess was worth +0.05, and an affordance +0.1."""
        calm = PhysicalGroundingEngine().ingest(
            _obs("d", {"objects": [{"id": "a", "type": "w"}]}, confidence=0.6)
        )
        alarmed = PhysicalGroundingEngine().ingest(
            _obs("d", {"objects": [{"id": "a", "type": "w", "status": "critical"}]}, confidence=0.6)
        )
        assert alarmed.confidence <= calm.confidence + 1e-9

    def test_the_basis_is_stated(self):
        state = PhysicalGroundingEngine().ingest(_obs("d", {"nothing": 1}))
        assert "observation confidence" in state.confidence_basis

    def test_a_more_confident_observation_grounds_better(self):
        low = PhysicalGroundingEngine().ingest(_obs("d", {"nothing": 1}, confidence=0.2))
        high = PhysicalGroundingEngine().ingest(_obs("d", {"nothing": 1}, confidence=0.95))
        assert high.confidence > low.confidence


class TestDurability:
    def test_the_world_survives_a_restart(self, tmp_path):
        """state_path was accepted and never used: nothing saved, nothing
        loaded, while integration.py read the object count into a health
        surface."""
        path = tmp_path / "grounding.json"
        engine = PhysicalGroundingEngine(state_path=path)
        engine.ingest(_obs("d", {"objects": [{"id": "keep", "type": "widget"}]}))
        engine.save()

        restored = PhysicalGroundingEngine(state_path=path)
        assert len(restored.objects) == 1
        assert restored.resources.get("confidence") is not None

    def test_ingest_does_not_write(self, tmp_path):
        """ingest runs inside the integration layer's lock, and an fsync
        under a lock is how this runtime freezes."""
        path = tmp_path / "grounding.json"
        engine = PhysicalGroundingEngine(state_path=path)
        engine.ingest(_obs("d", {"objects": [{"id": "x", "type": "w"}]}))
        assert not path.exists()

    def test_a_corrupt_state_file_starts_empty_rather_than_raising(self, tmp_path):
        path = tmp_path / "grounding.json"
        path.write_text("{not json")
        engine = PhysicalGroundingEngine(state_path=path)
        assert engine.objects == {}

    def test_an_unreadable_object_row_is_skipped(self, tmp_path):
        path = tmp_path / "grounding.json"
        path.write_text(
            json.dumps(
                {
                    "domains": {
                        "d": {
                            "objects": {
                                "good": {"object_id": "good", "kind": "w", "confidence": 0.5},
                                "bad": {"kind": "w"},
                            },
                            "resources": {},
                        }
                    }
                }
            )
        )
        engine = PhysicalGroundingEngine(state_path=path)
        assert set(engine.objects) == {"good"}


class TestBounds:
    def test_a_cyclic_state_does_not_exhaust_the_stack(self):
        """_flat recursed with no depth or cycle limit, on the reflex path."""
        state: dict = {"a": 1}
        state["self"] = state
        engine = PhysicalGroundingEngine()
        grounded = engine.ingest(_obs("d", state))
        assert grounded.state_id

    def test_a_deeply_nested_state_is_bounded(self):
        node: dict = {"leaf": 1}
        for _ in range(200):
            node = {"child": node}
        engine = PhysicalGroundingEngine()
        assert engine.ingest(_obs("d", node)).state_id

    def test_the_returned_state_never_exceeds_max_objects(self):
        """Pruning ran AFTER the snapshot, so the state handed to the caller
        could hold more objects than the engine allows."""
        engine = PhysicalGroundingEngine(max_objects=5)
        state = engine.ingest(
            _obs("d", {"objects": [{"id": f"o{i}", "type": "w"} for i in range(50)]})
        )
        assert len(state.objects) <= 5
