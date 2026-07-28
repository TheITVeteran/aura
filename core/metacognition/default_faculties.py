"""The faculties Aura actually has, and the probes that can currently see them.

This is where the model stops being a mechanism and starts being a picture of
a specific mind. Every probe here reads a real runtime surface, and every
faculty without one says so rather than being quietly omitted — an undeclared
faculty is invisible, and invisible is worse than known-unmeasured.

The honest state today: concurrency safety and operational integrity are
measurable from live counters; memory is measurable only as far as "is dense
retrieval even available"; temporal reasoning has no probe at all. That last
one is not a gap in this file, it is a true statement about Aura, and the
model is built to surface it as a target rather than hide it.
"""

from __future__ import annotations

import threading
from typing import Any

from core.metacognition.faculty_model import (
    Faculty,
    FacultyRegistry,
    ImprovementMetric,
    get_faculty_registry,
)

_declared_lock = threading.RLock()
_declared = False


# ── probes ────────────────────────────────────────────────────────────────
# Each returns a number, or None when it genuinely cannot measure. None is a
# real answer; a fabricated default would be the exact failure this campaign
# exists to remove.


def _loop_blocking_holds() -> float | None:
    """How often a lock froze the event loop. Attention that cannot be paid.

    A blocking hold on the loop is attention allocation failing in its most
    literal form: for that window the runtime could attend to nothing.
    """
    try:
        from core.runtime.lockdep import lockdep_report

        report = lockdep_report()
    except (ImportError, RuntimeError, AttributeError, TypeError, ValueError):
        return None
    splats = report.get("splats")
    if not isinstance(splats, (list, tuple)):
        return None
    return float(
        sum(1 for s in splats if "loop_blocking" in str(getattr(s, "kind", s) or ""))
    )


def _open_degradations() -> float | None:
    """Recorded degradations — capability the runtime knows it has lost."""
    try:
        from core.runtime.errors import get_degradation_tracker

        # count() is per-subsystem; the process-wide total lives in status().
        status = get_degradation_tracker().status()
    except (ImportError, RuntimeError, AttributeError, TypeError, ValueError):
        return None
    if not isinstance(status, dict):
        return None
    total = status.get("total_degradations")
    try:
        return float(total)
    except (TypeError, ValueError):
        return None


def _dense_retrieval_available() -> float | None:
    """Whether recall runs on dense embeddings or has fallen back to lexical.

    A coarse probe, and deliberately labelled as one: it answers "is the good
    path available", not "how good is recall". Recall@k against a fixed probe
    set is the metric this should become.
    """
    try:
        from core.memory import rag

        if getattr(rag, "_EMBED_ENGINE_FAILED", False):
            return 0.0
        return 1.0 if getattr(rag, "_EMBED_ENGINE", None) is not None else 0.0
    except (ImportError, RuntimeError, AttributeError, TypeError, ValueError):
        return None


def _unmeasured(reason: str):
    """A probe for a faculty nothing can currently see.

    Declaring the faculty with an honest non-probe is the point: it makes the
    blind spot countable and gives the self-model something to want.
    """

    def _probe() -> float | None:
        return None

    _probe.__doc__ = reason
    return _probe


# ── declarations ──────────────────────────────────────────────────────────


def declare_default_faculties(registry: FacultyRegistry | None = None) -> FacultyRegistry:
    """Declare Aura's faculties into ``registry`` (idempotent)."""
    target = registry if registry is not None else get_faculty_registry()

    target.declare(
        Faculty(
            faculty_id="memory",
            description="Storing and recalling what happened and what is true.",
            owner="core.memory",
            gates=("attention_allocation", "temporal_reasoning", "reasoning"),
            metrics=(
                ImprovementMetric(
                    metric_id="dense_retrieval_available",
                    unit="",
                    direction="higher_is_better",
                    probe=_dense_retrieval_available,
                    floor=0.0,
                    target=1.0,
                    ceiling=1.0,
                    description=(
                        "Availability of dense retrieval, not recall quality. "
                        "Should become recall@k against a fixed probe set."
                    ),
                ),
            ),
        )
    )

    target.declare(
        Faculty(
            faculty_id="attention_allocation",
            description="Where cognitive effort goes, and whether the loop can spend it.",
            owner="core.runtime",
            gates=("temporal_reasoning", "reasoning"),
            metrics=(
                ImprovementMetric(
                    metric_id="loop_blocking_holds",
                    unit=" holds",
                    direction="lower_is_better",
                    probe=_loop_blocking_holds,
                    floor=20.0,
                    target=0.0,
                    ceiling=0.0,
                    description="Windows in which the event loop could attend to nothing.",
                ),
            ),
        )
    )

    target.declare(
        Faculty(
            faculty_id="operational_integrity",
            description="Capability the runtime knows it has lost.",
            owner="core.runtime",
            gates=("memory", "attention_allocation", "reasoning"),
            metrics=(
                ImprovementMetric(
                    metric_id="open_degradations",
                    unit=" degradations",
                    direction="lower_is_better",
                    probe=_open_degradations,
                    floor=50.0,
                    target=0.0,
                    ceiling=0.0,
                ),
            ),
        )
    )

    # Declared WITHOUT a working probe on purpose. These are the faculties the
    # user named that Aura genuinely cannot see yet; leaving them out would
    # make the self-model look complete when it is not.
    target.declare(
        Faculty(
            faculty_id="temporal_reasoning",
            description="Ordering events, estimating durations, reasoning about when.",
            owner="core.cognition",
            metrics=(
                ImprovementMetric(
                    metric_id="event_order_accuracy",
                    unit="",
                    direction="higher_is_better",
                    probe=_unmeasured("no temporal benchmark is wired into the runtime"),
                    floor=0.0,
                    target=0.9,
                    ceiling=1.0,
                ),
            ),
        )
    )

    target.declare(
        Faculty(
            faculty_id="reasoning",
            description="Multi-step inference and its calibration.",
            owner="core.cognition",
            metrics=(
                ImprovementMetric(
                    metric_id="verifier_pass_rate",
                    unit="",
                    direction="higher_is_better",
                    probe=_unmeasured(
                        "no live verifier stream is aggregated into a rate yet"
                    ),
                    floor=0.0,
                    target=0.85,
                    ceiling=1.0,
                ),
            ),
        )
    )
    return target


def ensure_default_faculties() -> FacultyRegistry:
    """Declare the defaults once, on first use of the self-model."""
    global _declared
    registry = get_faculty_registry()
    with _declared_lock:
        if _declared:
            return registry
        _declared = True
    return declare_default_faculties(registry)


def reset_default_faculties_for_test() -> None:
    global _declared
    with _declared_lock:
        _declared = False


__all__ = [
    "declare_default_faculties",
    "ensure_default_faculties",
    "reset_default_faculties_for_test",
]
