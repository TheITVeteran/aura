"""Sample where learning is possible (CP237).

GRPO's failure mode is precise: if every completion in a group earns the
same grade, the advantages are all zero and the step teaches nothing. On
hard tasks the base model never solves, EVERY group is all-wrong, and the
run burns compute producing a tidy loss curve over no signal. That is the
"tasks the model never gets right" problem, and skipping degenerate groups
after sampling them is treating the symptom -- the compute was already
spent.

Anima Rationis line 920 names the cure: a minimax curriculum. Sample the
domain where the model is WEAKEST but not hopeless -- where reward variance
still exists -- and escalate as it improves. Concretely:

    P(cell) proportional to learnability(cell)
    learnability is highest where pass rate is mid (max reward variance)
    and zero where pass rate is 0 (nothing to reinforce) or 1 (nothing to
    improve)

Two mechanisms make a hopeless cell learnable rather than merely skipped:

* **Escalation.** A saturated cell (pass rate -> 1) stops being sampled, and
  the next harder cell -- previously all-wrong because the model could not
  yet reach it -- now sits in the learnable band. Difficulty rises with
  competence instead of being fixed.
* **Retrieval as a variance source.** For knowledge-gated tasks, a cell the
  model cannot solve from memory becomes solvable-sometimes once retrieval
  is enabled (CP236). The curriculum can turn retrieval on for a hopeless
  knowledge cell rather than abandon it, which is the direct tie between
  this sampler and the integrated evaluation.

The honesty property: the sampler REPORTS which cells are learnable,
saturated, and hopeless, so a run that has run out of reachable frontier
says so rather than continuing to sample dead cells.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any

ADAPTIVE_CURRICULUM_SCHEMA = "aura.adaptive_curriculum.v1"

# A cell is learnable when its pass rate leaves room for reward variance:
# not so low that groups are all-wrong, not so high they are all-correct.
LEARNABLE_LOW = 0.05
LEARNABLE_HIGH = 0.95


@dataclass
class CellStats:
    """Running reward statistics for one (family, difficulty) cell."""

    family: str
    difficulty: int
    trials: int = 0
    reward_sum: float = 0.0
    degenerate: int = 0

    @property
    def pass_rate(self) -> float:
        return self.reward_sum / self.trials if self.trials else 0.0

    def observe(self, mean_reward: float, *, degenerate: bool) -> None:
        if not 0.0 <= float(mean_reward) <= 1.0:
            raise ValueError("mean_reward must be a rate in [0, 1]")
        self.trials += 1
        self.reward_sum += float(mean_reward)
        if degenerate:
            self.degenerate += 1

    def learnability(self, *, exploration_prior: float = 0.3) -> float:
        """How much learning signal this cell is expected to yield.

        Unexplored cells get an optimistic prior so the sampler tries them
        before writing them off. Explored cells score by reward variance --
        peaked at pass rate 0.5, zero at the extremes -- weighted toward the
        weaker side (minimax: prefer where the model is worse but not lost).
        """
        if self.trials < 2:
            return float(exploration_prior)
        rate = self.pass_rate
        if rate <= LEARNABLE_LOW or rate >= LEARNABLE_HIGH:
            return 0.0
        # Bernoulli variance rate*(1-rate), tilted toward low pass rates so
        # the weakest reachable frontier is preferred.
        variance = rate * (1.0 - rate)
        minimax_tilt = (1.0 - rate) ** 0.5
        return float(variance * minimax_tilt)

    def state(self) -> dict[str, Any]:
        return {
            "family": self.family,
            "difficulty": self.difficulty,
            "trials": self.trials,
            "reward_sum": self.reward_sum,
            "degenerate": self.degenerate,
        }


@dataclass
class AdaptiveCurriculum:
    """Minimax curriculum over (family, difficulty) cells."""

    cells: dict[tuple[str, int], CellStats] = field(default_factory=dict)
    exploration_prior: float = 0.3

    @classmethod
    def over(
        cls, families: list[str], difficulties: list[int]
    ) -> AdaptiveCurriculum:
        if not families or not difficulties:
            raise ValueError("curriculum needs families and difficulties")
        curriculum = cls()
        for family in families:
            for difficulty in difficulties:
                curriculum.cells[(family, difficulty)] = CellStats(
                    family=family, difficulty=difficulty
                )
        return curriculum

    def observe(
        self, family: str, difficulty: int, mean_reward: float, *, degenerate: bool
    ) -> None:
        key = (family, difficulty)
        if key not in self.cells:
            self.cells[key] = CellStats(family=family, difficulty=difficulty)
        self.cells[key].observe(mean_reward, degenerate=degenerate)

    def sample(self, rng: random.Random) -> tuple[str, int]:
        """Pick a cell weighted by learnability.

        When every cell is hopeless (all learnability 0), falls back to the
        LEAST-explored cell rather than a dead one -- optimism under total
        uncertainty, not surrender.
        """
        weights = [
            (key, cell.learnability(exploration_prior=self.exploration_prior))
            for key, cell in self.cells.items()
        ]
        total = sum(w for _key, w in weights)
        if total <= 0.0:
            least_explored = min(self.cells.values(), key=lambda c: c.trials)
            return (least_explored.family, least_explored.difficulty)
        target = rng.random() * total
        cumulative = 0.0
        for key, weight in weights:
            cumulative += weight
            if target <= cumulative:
                return key
        return weights[-1][0]

    def report(self) -> dict[str, Any]:
        learnable, saturated, hopeless, unexplored = [], [], [], []
        for cell in self.cells.values():
            label = f"{cell.family}@{cell.difficulty}"
            if cell.trials < 2:
                unexplored.append(label)
            elif cell.learnability() > 0.0:
                learnable.append(label)
            elif cell.pass_rate >= LEARNABLE_HIGH:
                saturated.append(label)
            else:
                hopeless.append(label)
        return {
            "schema": ADAPTIVE_CURRICULUM_SCHEMA,
            "cells": len(self.cells),
            "learnable": sorted(learnable),
            "saturated": sorted(saturated),
            "hopeless": sorted(hopeless),
            "unexplored": sorted(unexplored),
            # A run with no learnable cells left has exhausted its reachable
            # frontier; continuing to sample would be theatre.
            "has_reachable_frontier": bool(learnable or unexplored),
            "pass_rates": {
                f"{c.family}@{c.difficulty}": round(c.pass_rate, 4)
                for c in self.cells.values()
                if c.trials >= 2
            },
        }

    def state(self) -> dict[str, Any]:
        """Serializable, so a resumed run keeps its learned difficulty map."""
        return {
            "schema": ADAPTIVE_CURRICULUM_SCHEMA,
            "exploration_prior": self.exploration_prior,
            "cells": [cell.state() for cell in self.cells.values()],
        }

    @classmethod
    def from_state(cls, state: dict[str, Any]) -> AdaptiveCurriculum:
        if not isinstance(state, dict) or set(state) != {
            "schema",
            "exploration_prior",
            "cells",
        }:
            raise ValueError("curriculum state shape differs")
        if state.get("schema") != ADAPTIVE_CURRICULUM_SCHEMA:
            raise ValueError("curriculum state schema differs")
        exploration_prior = state.get("exploration_prior")
        if (
            isinstance(exploration_prior, bool)
            or not isinstance(exploration_prior, (int, float))
            or not math.isfinite(float(exploration_prior))
            or not 0.0 <= float(exploration_prior) <= 1.0
        ):
            raise ValueError("curriculum exploration prior is invalid")
        entries = state.get("cells")
        if not isinstance(entries, list) or not entries:
            raise ValueError("curriculum cells are invalid")
        curriculum = cls(exploration_prior=float(exploration_prior))
        for entry in entries:
            if not isinstance(entry, dict) or set(entry) != {
                "family",
                "difficulty",
                "trials",
                "reward_sum",
                "degenerate",
            }:
                raise ValueError("curriculum cell state shape differs")
            family = entry.get("family")
            difficulty = entry.get("difficulty")
            trials = entry.get("trials")
            reward_sum = entry.get("reward_sum")
            degenerate = entry.get("degenerate")
            if not isinstance(family, str) or not family:
                raise ValueError("curriculum cell family is invalid")
            if type(difficulty) is not int or difficulty < 1:
                raise ValueError("curriculum cell difficulty is invalid")
            if type(trials) is not int or trials < 0:
                raise ValueError("curriculum cell trials are invalid")
            if type(degenerate) is not int or not 0 <= degenerate <= trials:
                raise ValueError("curriculum cell degenerate count is invalid")
            if (
                isinstance(reward_sum, bool)
                or not isinstance(reward_sum, (int, float))
                or not math.isfinite(float(reward_sum))
                or not 0.0 <= float(reward_sum) <= float(trials)
            ):
                raise ValueError("curriculum cell reward sum is invalid")
            cell = CellStats(
                family=family,
                difficulty=difficulty,
                trials=trials,
                reward_sum=float(reward_sum),
                degenerate=degenerate,
            )
            key = (cell.family, cell.difficulty)
            if key in curriculum.cells:
                raise ValueError("curriculum cell is duplicated")
            curriculum.cells[key] = cell
        return curriculum


def warm_start_pass_rates(
    families: list[str],
    difficulties: list[int],
    measure: Any,
    *,
    samples_per_cell: int = 4,
) -> AdaptiveCurriculum:
    """Measure base pass rates BEFORE training, so step one is not wasted.

    ``measure(family, difficulty) -> reward in [0, 1] | None`` runs a quick
    base rollout. ``None`` means the cell was not measured (for example, a
    wall-clock deadline); it remains unexplored rather than receiving a
    fabricated mid-band pass rate. Without this, the first many steps are
    spent discovering which cells are all-wrong; with it, the sampler starts
    on the measured learnable band while unknown cells stay explicit.
    """
    curriculum = AdaptiveCurriculum.over(families, difficulties)
    for family in families:
        for difficulty in difficulties:
            for _ in range(max(2, samples_per_cell)):
                measured = measure(family, difficulty)
                if measured is None:
                    break
                reward = float(measured)
                curriculum.observe(
                    family, difficulty, reward,
                    degenerate=reward <= 0.0 or reward >= 1.0,
                )
    return curriculum


__all__ = [
    "ADAPTIVE_CURRICULUM_SCHEMA",
    "LEARNABLE_HIGH",
    "LEARNABLE_LOW",
    "AdaptiveCurriculum",
    "CellStats",
    "warm_start_pass_rates",
]
