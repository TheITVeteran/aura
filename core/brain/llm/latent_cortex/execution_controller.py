"""Learned per-problem execution controller — evidence-gated, never vibes.

The schedule library already promotes validated layer programs; the
allocation already scales with stakes/uncertainty. What neither does is
LEARN which execution configuration suits which kind of problem. This
controller closes that loop as a conservative contextual bandit:

    context bucket  = (domain, facet signature, stakes band, uncertainty band)
    arm             = a bounded, validated tweak over the base allocation
                      (deeper recurrence / wider branches / probe-guided
                      bytecode / lean fast weights)
    reward          = the episode's VERIFIED outcome (task-verifier best
                      score), never convergence prettiness

Selection is Wilson-bounded and evidence-gated: an arm may override the
base allocation only when its pessimistic (lower-bound) verified success
rate beats the base arm's optimistic (upper-bound) rate on ≥ MIN_TRIALS
graded episodes in that context — the same conservatism the Verifier
Foundry applies to verifiers. Until then the controller only OBSERVES
(base allocation runs, outcomes are recorded) and explores at most one
arm per EXPLORE_EVERY episodes, budget permitting. Every decision is
receipted with the evidence that justified it.

State persists under data/latent_cortex/controller/ through the governed
write gateway; a corrupt ledger degrades to observe-only, never crashes.
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger("Aura.LatentCortex.ExecutionController")

EXECUTION_CONTROLLER_SCHEMA = "aura.latent_execution_controller.v1"

MIN_TRIALS = 12
EXPLORE_EVERY = 4
_MAX_LEDGER_ROWS = 5000
_Z95 = 1.959963984540054

# Bounded arm menu: every arm is a small, validated delta over the base
# allocation. Arms may only tighten or reshape — never exceed the absolute
# caps the service already enforces.
ARMS: dict[str, dict[str, Any]] = {
    "base": {},
    "deeper_recurrence": {"max_steps_delta": 4, "n_branches_cap": 2},
    "wider_branches": {"n_branches_delta": 1, "max_steps_delta": -2},
    "probe_guided_bytecode": {"bytecode_probes": True},
    "lean_fast_weights": {"fast_weights_max_layers_cap": 2},
}

_WORD_RE = re.compile(r"[^\W\d_]{4,}", re.UNICODE)


def _wilson(successes: int, n: int, *, upper: bool) -> float:
    """Wilson bound for binary, independently graded outcomes.

    A mean verifier score is not a binomial success count. Keeping that score
    as descriptive telemetry is useful, but feeding its sum into this formula
    creates fictitious fractional trials and invalid confidence intervals.
    """

    if n <= 0:
        return 1.0 if upper else 0.0
    if successes < 0 or successes > n:
        raise ValueError("Wilson successes must be an integer in [0, n]")
    p_hat = successes / n
    z2 = _Z95 * _Z95
    denominator = 1.0 + z2 / n
    center = p_hat + z2 / (2 * n)
    margin = _Z95 * math.sqrt(
        (p_hat * (1.0 - p_hat) + z2 / (4 * n)) / n
    )
    bound = (center + margin) / denominator if upper else (center - margin) / denominator
    return max(0.0, min(1.0, bound))


def context_bucket(
    objective: str, domain: str, stakes: float, uncertainty: float
) -> str:
    """Coarse, deterministic context key: generalizes, never memorizes."""
    try:
        from core.brain.llm.latent_cortex.output_quality import request_facets

        facets = ",".join(sorted(request_facets(str(objective or ""))))
    except (ImportError, AttributeError, RuntimeError, TypeError, ValueError):
        facets = ""
    words = len(_WORD_RE.findall(str(objective or "")))
    length_band = "short" if words < 24 else ("medium" if words < 96 else "long")
    stakes_band = "high" if stakes >= 0.7 else ("mid" if stakes >= 0.4 else "low")
    uncertainty_band = (
        "high" if uncertainty >= 0.7 else ("mid" if uncertainty >= 0.4 else "low")
    )
    return "|".join(
        [
            str(domain or "general")[:24],
            facets or "none",
            length_band,
            f"s:{stakes_band}",
            f"u:{uncertainty_band}",
        ]
    )


class ExecutionController:
    """Persistent bandit over execution arms, keyed by context bucket."""

    def __init__(self, root: Path | str | None = None) -> None:
        if root is None:
            try:
                from core.config import DATA_DIR

                root = Path(DATA_DIR) / "latent_cortex" / "controller"
            except (ImportError, AttributeError, RuntimeError, TypeError):
                root = Path("data/latent_cortex/controller")
        self.root = Path(root)
        self.ledger_path = self.root / "outcomes.jsonl"
        self._cells: dict[tuple[str, str], dict[str, float]] = {}
        self._episodes_seen = 0
        self._restore_errors = 0
        self._restore()

    # ── State ────────────────────────────────────────────────────────────
    def _restore(self) -> None:
        try:
            if not self.ledger_path.exists():
                return
            with open(self.ledger_path, encoding="utf-8") as handle:
                for line in handle:
                    try:
                        row = json.loads(line)
                        self._fold(row)
                    except (json.JSONDecodeError, TypeError, ValueError, KeyError):
                        self._restore_errors += 1
        except OSError as exc:
            self._restore_errors += 1
            logger.warning("Controller ledger unreadable — observe-only: %s", exc)

    def _fold(self, row: dict[str, Any]) -> None:
        if row.get("checked") is not True:
            raise ValueError("controller outcome is not independently checked")
        if not isinstance(row.get("success"), bool):
            raise ValueError("controller outcome success must be boolean")
        bucket = row.get("bucket")
        arm = row.get("arm")
        score = row.get("verified_score")
        if not isinstance(bucket, str) or not bucket or len(bucket) > 160:
            raise ValueError("controller outcome bucket is invalid")
        if not isinstance(arm, str) or arm not in ARMS:
            raise ValueError("controller outcome arm is invalid")
        if (
            not isinstance(score, (int, float))
            or isinstance(score, bool)
            or not math.isfinite(float(score))
            or not 0.0 <= float(score) <= 1.0
        ):
            raise ValueError("controller verified score is invalid")
        cell = self._cells.setdefault(
            (bucket, arm),
            {"n": 0, "verified_sum": 0.0, "successes": 0},
        )
        cell["n"] += 1
        cell["verified_sum"] += float(score)
        cell["successes"] += int(bool(row.get("success")))
        self._episodes_seen += 1

    def _append(self, row: dict[str, Any]) -> bool:
        try:
            from core.governance_context import local_internal_governed_scope
            from core.runtime.file_write_gateway import get_file_write_gateway

            line = json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
            with local_internal_governed_scope(
                "latent_execution_controller", domain="state_mutation"
            ):
                get_file_write_gateway().append_text(
                    self.ledger_path, line, source="latent_execution_controller"
                )
            return True
        except (ImportError, OSError, RuntimeError, TypeError, ValueError) as exc:
            logger.debug("Controller outcome not persisted: %s", exc)
            return False

    # ── Decisions ────────────────────────────────────────────────────────
    def choose(
        self,
        *,
        objective: str,
        domain: str,
        stakes: float,
        uncertainty: float,
    ) -> dict[str, Any]:
        """Pick an arm for this episode, with the evidence receipted.

        Exploitation requires separation: arm.lb > base.ub with both arms
        having ≥ MIN_TRIALS in this bucket. Exploration runs one non-base
        arm every EXPLORE_EVERY episodes (round-robin over the least-tried
        arms) so evidence accumulates without destabilizing the live path.
        """
        bucket = context_bucket(objective, domain, stakes, uncertainty)
        decision: dict[str, Any] = {
            "schema": EXECUTION_CONTROLLER_SCHEMA,
            "bucket": bucket,
            "arm": "base",
            "mode": "observe",
            "evidence": {},
        }
        base_cell = self._cells.get((bucket, "base"))
        base_n = int(base_cell["n"]) if base_cell else 0
        base_ub = (
            _wilson(int(base_cell["successes"]), base_n, upper=True)
            if base_cell and base_n
            else 1.0
        )
        best_arm, best_lb = "", 0.0
        for arm in ARMS:
            if arm == "base":
                continue
            cell = self._cells.get((bucket, arm))
            if not cell or cell["n"] < MIN_TRIALS or base_n < MIN_TRIALS:
                continue
            lb = _wilson(
                int(cell["successes"]), int(cell["n"]), upper=False
            )
            if lb > base_ub and lb > best_lb:
                best_arm, best_lb = arm, lb
        if best_arm:
            decision.update(
                {
                    "arm": best_arm,
                    "mode": "exploit",
                    "evidence": {
                        "arm_lb": round(best_lb, 4),
                        "base_ub": round(base_ub, 4),
                        "arm_n": int(self._cells[(bucket, best_arm)]["n"]),
                        "base_n": base_n,
                    },
                }
            )
            return decision
        self._episodes_seen += 1
        if self._episodes_seen % EXPLORE_EVERY == 0:
            candidates = sorted(
                (arm for arm in ARMS if arm != "base"),
                key=lambda arm: self._cells.get((bucket, arm), {"n": 0})["n"],
            )
            decision.update({"arm": candidates[0], "mode": "explore"})
        return decision

    def apply_arm(
        self,
        arm: str,
        config: dict[str, Any],
        *,
        recurrent_region: tuple[int, int] | None = None,
    ) -> dict[str, Any]:
        """Overlay one arm's bounded deltas onto an allocation config.

        ``recurrent_region`` is the (prelude_end, coda_start) of the target
        model; the probe-guided bytecode arm needs it to emit a valid
        program and quietly degenerates to base when it is unknown.
        """
        spec = ARMS.get(arm) or {}
        adjusted = dict(config)
        if "max_steps_delta" in spec:
            adjusted["max_steps"] = max(
                2, min(16, int(config.get("max_steps", 4)) + spec["max_steps_delta"])
            )
        if "n_branches_delta" in spec:
            adjusted["n_branches"] = max(
                1, min(4, int(config.get("n_branches", 1)) + spec["n_branches_delta"])
            )
        if "n_branches_cap" in spec:
            adjusted["n_branches"] = min(
                int(adjusted.get("n_branches", 1)), spec["n_branches_cap"]
            )
        if "fast_weights_max_layers_cap" in spec:
            adjusted["fast_weights_max_layers"] = min(
                int(config.get("fast_weights_max_layers", 4)),
                spec["fast_weights_max_layers_cap"],
            )
        if spec.get("bytecode_probes") and recurrent_region is not None:
            start, end = int(recurrent_region[0]), int(recurrent_region[1])
            repeats = max(2, int(adjusted.get("max_steps", 4)))
            first = max(1, repeats // 2)
            second = max(1, repeats - first)
            adjusted["schedule"] = {
                "name": "controller_probe_guided_v1",
                "ops": [
                    {"start": start, "end": end, "repeats": first},
                    {"kind": "savepoint"},
                    {"kind": "verify_probe", "revert_on_drop": True},
                    {"kind": "exchange"},
                    {"start": start, "end": end, "repeats": second},
                    {"kind": "verify_probe", "revert_on_drop": True},
                ],
            }
        return adjusted

    def record_outcome(
        self,
        *,
        bucket: str,
        arm: str,
        verified_score: float,
        success: bool,
        checked: bool,
        wall_clock_s: float = 0.0,
    ) -> bool:
        """Fold one independently checked episode outcome and persist it."""
        if checked is not True or not isinstance(success, bool):
            return False
        if not isinstance(bucket, str) or not bucket or len(bucket) > 160:
            return False
        if not isinstance(arm, str) or arm not in ARMS:
            return False
        if not isinstance(verified_score, (int, float)) or isinstance(
            verified_score, bool
        ):
            return False
        score = float(verified_score)
        if not math.isfinite(score) or not 0.0 <= score <= 1.0:
            return False
        try:
            elapsed = float(wall_clock_s)
        except (TypeError, ValueError, OverflowError):
            return False
        if not math.isfinite(elapsed) or elapsed < 0.0:
            return False
        row = {
            "bucket": bucket,
            "arm": arm,
            "verified_score": round(score, 6),
            "success": success,
            "checked": True,
            "wall_clock_s": round(elapsed, 3),
            "at": time.time(),
        }
        if not self._append(row):
            return False
        self._fold(row)
        return True

    def status(self) -> dict[str, Any]:
        return {
            "schema": EXECUTION_CONTROLLER_SCHEMA,
            "buckets": len({bucket for bucket, _ in self._cells}),
            "cells": [
                {
                    "bucket": bucket,
                    "arm": arm,
                    "n": int(cell["n"]),
                    "mean_verified": round(
                        cell["verified_sum"] / max(1, int(cell["n"])), 4
                    ),
                    "success_rate": round(
                        int(cell["successes"]) / max(1, int(cell["n"])), 4
                    ),
                }
                for (bucket, arm), cell in sorted(self._cells.items())
            ][:200],
            "episodes_seen": self._episodes_seen,
            "restore_errors": self._restore_errors,
        }


def controller_enabled() -> bool:
    """Kill switch: AURA_EXECUTION_CONTROLLER=0 disables learn + apply."""
    from core.runtime.flags import FlagKind, declare

    return bool(
        declare(
            "AURA_EXECUTION_CONTROLLER",
            kind=FlagKind.BOOL,
            default=True,
            description="Learned per-problem execution controller (learn + apply)",
            owner="core.brain.llm.latent_cortex.execution_controller",
        ).value()
    )


_instance: ExecutionController | None = None


def get_execution_controller() -> ExecutionController:
    global _instance
    if _instance is None:
        _instance = ExecutionController()
    return _instance


__all__ = [
    "ARMS",
    "EXECUTION_CONTROLLER_SCHEMA",
    "ExecutionController",
    "context_bucket",
    "controller_enabled",
    "get_execution_controller",
]
