from __future__ import annotations

from types import SimpleNamespace

import pytest

from core.agency.choice_game import SubjectiveChoiceGame, build_subjective_ending_game
from core.agency.initiative_arbiter import InitiativeArbiter, ScoredInitiative
from core.agency.subjective_choice import (
    ChoiceOption,
    SubjectiveChoiceEngine,
)


def test_subjective_choice_can_override_raw_drive_pressure(tmp_path):
    engine = SubjectiveChoiceEngine(state_path=tmp_path / "choices.json", mirror_identity=False)

    receipt = engine.choose(
        [
            ChoiceOption(
                id="should",
                label="Do the efficient maintenance ending",
                drive_score=0.92,
                features={"coherence": 0.35, "calm": 0.2},
            ),
            ChoiceOption(
                id="preferred",
                label="Choose the curious, beautiful, relational ending",
                drive_score=0.52,
                features={"novelty": 1.0, "beauty": 1.0, "connection": 0.9, "play": 0.7},
            ),
        ],
        context="choice_game:ending_style",
    )

    assert receipt.chosen_id == "preferred"
    assert receipt.drive_top_id == "should"
    assert receipt.preference_override is True
    assert "intentionally overrode raw drive top" in receipt.rationale


def test_choice_recall_satisfaction_and_consistency_are_persistent(tmp_path):
    path = tmp_path / "choices.json"
    engine = SubjectiveChoiceEngine(state_path=path, mirror_identity=False)
    options = [
        ChoiceOption(
            id="tidy",
            label="Tidy ending",
            drive_score=0.80,
            features={"coherence": 0.5, "calm": 0.4},
        ),
        ChoiceOption(
            id="wonder",
            label="Wonder ending",
            drive_score=0.62,
            features={"novelty": 0.9, "beauty": 0.9, "connection": 0.7},
        ),
    ]
    receipt = engine.choose(options, context="choice_game:memory")
    appraised = engine.appraise_outcome(
        receipt.choice_id,
        outcome="The ending felt more alive and worth remembering.",
        satisfaction=0.85,
    )

    assert appraised is not None
    assert appraised.happy_with_outcome is True
    recalled = engine.recall_choice(receipt.choice_id)
    assert recalled is not None
    assert recalled.chosen_label == receipt.chosen_label
    assert recalled.satisfaction == pytest.approx(0.85)

    reloaded = SubjectiveChoiceEngine(state_path=path, mirror_identity=False)
    persisted = reloaded.recall_choice(receipt.choice_id)
    assert persisted is not None
    assert persisted.chosen_label == receipt.chosen_label
    report = reloaded.consistency_report(context="choice_game:memory", options=options)
    assert report["consistent_with_prior"] is True


@pytest.mark.asyncio
async def test_initiative_arbiter_uses_subjective_choice_engine_for_valid_options(tmp_path, monkeypatch):
    import core.agency.subjective_choice as sce

    engine = SubjectiveChoiceEngine(state_path=tmp_path / "choices.json", mirror_identity=False)
    monkeypatch.setattr(sce, "get_subjective_choice_engine", lambda: engine)
    arbiter = InitiativeArbiter()

    async def fake_score(initiative, state):
        return ScoredInitiative(
            initiative=initiative,
            scores={
                "urgency": initiative["raw"],
                "novelty": 0.5,
                "identity_relevance": 0.5,
                "tension_resolution": 0.5,
                "expected_value": initiative["raw"],
                "resource_cost": 0.5,
                "social_appropriateness": 0.5,
                "continuity": 0.5,
            },
            final_score=initiative["raw"],
            rationale="",
        )

    monkeypatch.setattr(arbiter, "score_initiative", fake_score)
    state = SimpleNamespace(
        cognition=SimpleNamespace(
            pending_initiatives=[
                {
                    "goal": "Run routine maintenance",
                    "raw": 0.90,
                    "metadata": {"preference_features": {"coherence": 0.3}},
                },
                {
                    "goal": "Explore a novel beautiful idea with Bryan",
                    "raw": 0.52,
                    "metadata": {
                        "preference_features": {
                            "novelty": 1.0,
                            "beauty": 0.9,
                            "connection": 0.9,
                            "autonomy": 0.8,
                        }
                    },
                },
            ],
            working_memory=[],
        )
    )

    selected = await arbiter.arbitrate(state)

    assert selected is not None
    assert selected.initiative["goal"] == "Explore a novel beautiful idea with Bryan"
    assert selected.initiative["metadata"]["subjective_preference_override"] is True
    assert engine.recall_choice(context="initiative_arbiter") is not None


def test_subjective_choice_game_aligns_intention_action_recall_and_satisfaction(tmp_path):
    engine = SubjectiveChoiceEngine(state_path=tmp_path / "choices.json", mirror_identity=False)
    game = SubjectiveChoiceGame(engine)

    report = game.run(
        scenario_id="subjective_ending_preference",
        stages=build_subjective_ending_game(),
    )

    assert report.stated_action_alignment_rate == 1.0
    assert report.recall_alignment_rate == 1.0
    assert report.average_satisfaction > 0.70
    assert report.happy_with_final_outcome is True
    assert any(stage.preference_override for stage in report.stages)
    assert "open beautiful mythos" in report.stages[-1].actual_choice_label
    assert "mean satisfaction" in report.final_commentary

    reloaded = SubjectiveChoiceEngine(state_path=tmp_path / "choices.json", mirror_identity=False)
    last = reloaded.recall_choice(report.stages[-1].choice_id)
    assert last is not None
    assert last.chosen_label == report.stages[-1].actual_choice_label
