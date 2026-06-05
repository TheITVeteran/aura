from __future__ import annotations

from dataclasses import dataclass

from .aura_now import AffectiveState, AttentionState, BodyState, WorkspaceState, WorldState


@dataclass(frozen=True)
class Coalition:
    name: str
    salience: float
    urgency: float = 0.0
    source: str = ""

    @property
    def score(self) -> float:
        return max(0.0, float(self.salience)) + max(0.0, float(self.urgency)) * 0.5


class WorkspaceIgnition:
    BROADCAST_TARGETS = ("memory", "planner", "will", "speaker", "self_model", "learning")

    def __init__(self) -> None:
        self._lesioned = False

    def lesion(self) -> None:
        """Disable workspace broadcast for lesion and ablation tests."""
        self._lesioned = True

    def restore(self) -> None:
        """Restore workspace broadcast after a lesion test."""
        self._lesioned = False

    @property
    def is_lesioned(self) -> bool:
        return self._lesioned

    def build_coalitions(
        self,
        *,
        body: BodyState,
        affect: AffectiveState,
        world: WorldState,
    ) -> list[Coalition]:
        focus_salience = 0.6 if world.focal_object else 0.15
        return [
            Coalition("user_request", focus_salience, 0.25 if world.task_active else 0.0, "world"),
            Coalition("body_pressure", body.total_pressure, affect.distress, "interoception"),
            Coalition("curiosity_impulse", affect.curiosity, 0.1, "affect"),
            Coalition("safety_boundary", affect.distress, 1.0 if affect.distress > 0.65 else 0.0, "governance"),
            Coalition("self_continuity_alert", affect.free_energy, 0.6 if affect.free_energy > 0.45 else 0.0, "self"),
        ]

    def ignite(
        self,
        coalitions: list[Coalition],
        *,
        recurrent_cycles: int = 12,
        threshold: float = 0.35,
        lesion: bool = False,
    ) -> tuple[WorkspaceState, AttentionState]:
        if lesion or self._lesioned:
            names = tuple(coalition.name for coalition in coalitions)
            return (
                WorkspaceState(
                    winner="",
                    ignition_strength=0.0,
                    broadcast_targets=(),
                    competing_coalitions=names,
                    lesion="workspace_ignition",
                ),
                AttentionState(
                    focal_object="",
                    why_selected=("workspace ignition lesioned",),
                    stability=0.0,
                    competing_objects=names,
                    control=0.0,
                ),
            )
        cycles = max(1, int(recurrent_cycles))
        scored = sorted(coalitions, key=lambda item: item.score, reverse=True)
        winner = scored[0] if scored else Coalition("", 0.0)
        runner_up = scored[1].score if len(scored) > 1 else 0.0
        ignition = max(0.0, min(1.0, (winner.score - runner_up * 0.35) * (1.0 + min(cycles, 16) / 32.0)))
        targets = self.BROADCAST_TARGETS if ignition >= threshold else ()
        reasons = (winner.source, f"score={winner.score:.3f}") if winner.name else ()
        return (
            WorkspaceState(
                winner=winner.name,
                ignition_strength=round(ignition, 4),
                broadcast_targets=targets,
                competing_coalitions=tuple(coalition.name for coalition in scored),
            ),
            AttentionState(
                focal_object=winner.name,
                why_selected=tuple(reason for reason in reasons if reason),
                stability=round(max(0.0, min(1.0, ignition - abs(winner.score - runner_up) * 0.1)), 4),
                competing_objects=tuple(coalition.name for coalition in scored[1:4]),
                control=round(max(0.0, min(1.0, ignition)), 4),
            ),
        )
