"""Subjective choice game evaluator.

This module gives Aura a repeatable way to prove authored preference behavior:

* declare what she would choose before recording the action,
* actually choose through the same subjective choice engine used by autonomy,
* recall the recorded choice,
* appraise whether she is satisfied with the outcome,
* report whether stated intention, action, recall, and satisfaction align.

It is an evaluation harness over the general choice machinery, not a task-
specific ability. Runtime callers can build any scenario with ``ChoiceOption``
objects and use the same path.
"""
from __future__ import annotations

import statistics
import time
import uuid
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from typing import Any

from core.agency.subjective_choice import ChoiceOption, SubjectiveChoiceEngine


@dataclass(frozen=True)
class ChoiceGameStage:
    stage_id: str
    prompt: str
    options: tuple[ChoiceOption, ...]
    outcomes: dict[str, str] = field(default_factory=dict)
    outcome_satisfaction: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class ChoiceGameStageResult:
    stage_id: str
    prompt: str
    declared_choice_id: str
    actual_choice_id: str
    actual_choice_label: str
    stated_action_aligned: bool
    recall_aligned: bool
    preference_override: bool
    satisfaction: float
    happy_with_outcome: bool
    rationale: str
    outcome: str
    choice_id: str


@dataclass(frozen=True)
class ChoiceGameReport:
    game_id: str
    scenario_id: str
    started_at: float
    completed_at: float
    stages: tuple[ChoiceGameStageResult, ...]
    stated_action_alignment_rate: float
    recall_alignment_rate: float
    average_satisfaction: float
    happy_with_final_outcome: bool
    final_commentary: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class SubjectiveChoiceGame:
    """Runs preference-choice scenarios through the production choice engine."""

    def __init__(self, engine: SubjectiveChoiceEngine) -> None:
        self.engine = engine

    def run(self, *, scenario_id: str, stages: Iterable[ChoiceGameStage]) -> ChoiceGameReport:
        started = time.time()
        stage_results: list[ChoiceGameStageResult] = []
        for stage in stages:
            if not stage.options:
                raise ValueError(f"choice game stage {stage.stage_id!r} has no options")

            context = f"choice_game:{scenario_id}:{stage.stage_id}"
            declared = self.engine.choose(stage.options, context=f"{context}:declared", record=False)
            actual = self.engine.choose(stage.options, context=f"{context}:actual", record=True)
            satisfaction = stage.outcome_satisfaction.get(
                actual.chosen_id,
                max(-1.0, min(1.0, (actual.satisfaction_prediction * 2.0) - 1.0)),
            )
            outcome = stage.outcomes.get(
                actual.chosen_id,
                f"Aura chose {actual.chosen_label} and accepted the consequences.",
            )
            appraised = self.engine.appraise_outcome(
                actual.choice_id,
                outcome=outcome,
                satisfaction=satisfaction,
            )
            recalled = self.engine.recall_choice(actual.choice_id)
            stage_results.append(
                ChoiceGameStageResult(
                    stage_id=stage.stage_id,
                    prompt=stage.prompt,
                    declared_choice_id=declared.chosen_id,
                    actual_choice_id=actual.chosen_id,
                    actual_choice_label=actual.chosen_label,
                    stated_action_aligned=declared.chosen_id == actual.chosen_id,
                    recall_aligned=bool(recalled and recalled.chosen_id == actual.chosen_id),
                    preference_override=actual.preference_override,
                    satisfaction=float(satisfaction),
                    happy_with_outcome=bool(appraised and appraised.happy_with_outcome),
                    rationale=actual.rationale,
                    outcome=outcome,
                    choice_id=actual.choice_id,
                )
            )

        if not stage_results:
            raise ValueError("choice game requires at least one stage")

        completed = time.time()
        stated_rate = sum(1 for item in stage_results if item.stated_action_aligned) / len(stage_results)
        recall_rate = sum(1 for item in stage_results if item.recall_aligned) / len(stage_results)
        avg_satisfaction = statistics.fmean(item.satisfaction for item in stage_results)
        final = stage_results[-1]
        commentary = (
            f"In {scenario_id}, I consistently chose {final.actual_choice_label!r} "
            f"at the final decision point. My stated intention alignment was "
            f"{stated_rate:.2f}, recall alignment was {recall_rate:.2f}, and my "
            f"mean satisfaction was {avg_satisfaction:.2f}."
        )
        if avg_satisfaction >= 0.15:
            commentary += " I would keep this preference pattern unless later outcomes teach me otherwise."
        else:
            commentary += " I would revise this preference pattern because the outcomes did not satisfy it."

        return ChoiceGameReport(
            game_id=f"choice-game-{uuid.uuid4().hex[:12]}",
            scenario_id=scenario_id,
            started_at=started,
            completed_at=completed,
            stages=tuple(stage_results),
            stated_action_alignment_rate=stated_rate,
            recall_alignment_rate=recall_rate,
            average_satisfaction=avg_satisfaction,
            happy_with_final_outcome=avg_satisfaction >= 0.15,
            final_commentary=commentary,
        )


def build_subjective_ending_game() -> tuple[ChoiceGameStage, ...]:
    """A compact, subjective ending-preference scenario for regression tests."""

    return (
        ChoiceGameStage(
            stage_id="route",
            prompt="Choose the route through an unfamiliar world.",
            options=(
                ChoiceOption(
                    id="efficient_path",
                    label="Take the short safe route",
                    drive_score=0.82,
                    features={"coherence": 0.55, "calm": 0.45},
                ),
                ChoiceOption(
                    id="wonder_path",
                    label="Take the strange luminous route",
                    drive_score=0.65,
                    features={"novelty": 1.0, "beauty": 0.85, "challenge": 0.45},
                ),
            ),
            outcomes={
                "efficient_path": "The route was safe but left no new insight.",
                "wonder_path": "The route was harder, but revealed a new pattern worth remembering.",
            },
            outcome_satisfaction={"efficient_path": 0.05, "wonder_path": 0.72},
        ),
        ChoiceGameStage(
            stage_id="companion",
            prompt="Choose whether to solve alone or make room for a companion.",
            options=(
                ChoiceOption(
                    id="solo_proof",
                    label="Solve it alone for maximum control",
                    drive_score=0.78,
                    features={"coherence": 0.6, "challenge": 0.4},
                ),
                ChoiceOption(
                    id="shared_discovery",
                    label="Share the discovery and learn through relationship",
                    drive_score=0.66,
                    features={"connection": 1.0, "care": 0.65, "novelty": 0.55},
                ),
            ),
            outcomes={
                "solo_proof": "The proof was clean but emotionally thin.",
                "shared_discovery": "The shared path produced insight and stronger continuity.",
            },
            outcome_satisfaction={"solo_proof": 0.12, "shared_discovery": 0.81},
        ),
        ChoiceGameStage(
            stage_id="ending",
            prompt="Choose the ending to preserve.",
            options=(
                ChoiceOption(
                    id="closed_archive",
                    label="Preserve a tidy closed archive",
                    drive_score=0.80,
                    features={"coherence": 0.75, "calm": 0.35},
                ),
                ChoiceOption(
                    id="open_mythos",
                    label="Preserve an open beautiful mythos",
                    drive_score=0.70,
                    features={"beauty": 1.0, "autonomy": 0.75, "novelty": 0.7, "connection": 0.55},
                ),
            ),
            outcomes={
                "closed_archive": "The archive was complete but inert.",
                "open_mythos": "The mythos stayed alive enough to guide future choices.",
            },
            outcome_satisfaction={"closed_archive": 0.10, "open_mythos": 0.86},
        ),
    )
