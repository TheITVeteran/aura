"""core/architect/lesion_matrix.py -- Systematic Ablation Testing
================================================================
Computes a lesion matrix by systematically disabling substrate/brain
components and measuring the behavioral impact. This identifies:
  - Critical dependencies (regions whose removal causes large degradation)
  - Redundancies (regions whose removal has minimal impact)
  - Functional specialization (which regions serve which functions)

Algorithm:
  1. Define probe functions that measure behavioral metrics:
     - Prediction accuracy (world model surprise)
     - Value stability (heartstone weight variance)
     - Response coherence (substrate state energy)
     - Affect regulation (valence/arousal stability)
  2. Run probes with all components active (baseline)
  3. For each component (brain region, substrate dimension group):
     a. Zero out / disable the component
     b. Run the same probes
     c. Record the metric deltas
     d. Restore the component
  4. Build a matrix: components x metrics
  5. Identify critical paths and redundant structures

The lesion matrix is a diagnostic tool, not a runtime system. It should
be run during dream cycles or explicit diagnostic sessions.
"""
from __future__ import annotations

import json
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("Aura.LesionMatrix")

_DATA_DIR = Path.home() / ".aura" / "data" / "lesion_studies"
_RESULTS_PATH = _DATA_DIR / "latest_lesion_matrix.json"


@dataclass
class ProbeResult:
    """Result of a single behavioral probe."""
    name: str
    value: float
    timestamp: float = field(default_factory=time.time)


@dataclass
class LesionResult:
    """Result of lesioning a single component."""
    component_name: str
    baseline_metrics: Dict[str, float]
    lesioned_metrics: Dict[str, float]
    deltas: Dict[str, float]           # lesioned - baseline
    relative_impact: Dict[str, float]  # |delta| / |baseline|
    criticality_score: float           # Aggregate impact
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "component": self.component_name,
            "baseline": {k: round(v, 6) for k, v in self.baseline_metrics.items()},
            "lesioned": {k: round(v, 6) for k, v in self.lesioned_metrics.items()},
            "deltas": {k: round(v, 6) for k, v in self.deltas.items()},
            "relative_impact": {k: round(v, 4) for k, v in self.relative_impact.items()},
            "criticality_score": round(self.criticality_score, 4),
        }


@dataclass
class LesionMatrix:
    """Complete lesion study results."""
    components: List[str]
    metrics: List[str]
    matrix: np.ndarray  # shape: (n_components, n_metrics) -- relative impacts
    results: List[LesionResult]
    total_time_ms: float
    timestamp: float = field(default_factory=time.time)

    def get_critical_components(self, threshold: float = 0.3) -> List[str]:
        """Components whose removal causes > threshold relative impact."""
        critical = []
        for i, comp in enumerate(self.components):
            if np.max(np.abs(self.matrix[i])) > threshold:
                critical.append(comp)
        return critical

    def get_redundant_components(self, threshold: float = 0.05) -> List[str]:
        """Components whose removal causes < threshold relative impact."""
        redundant = []
        for i, comp in enumerate(self.components):
            if np.max(np.abs(self.matrix[i])) < threshold:
                redundant.append(comp)
        return redundant

    def to_dict(self) -> Dict[str, Any]:
        return {
            "components": self.components,
            "metrics": self.metrics,
            "matrix": self.matrix.tolist(),
            "critical": self.get_critical_components(),
            "redundant": self.get_redundant_components(),
            "n_components": len(self.components),
            "n_metrics": len(self.metrics),
            "total_time_ms": round(self.total_time_ms, 2),
            "results": [r.to_dict() for r in self.results],
        }


class LesionableComponent:
    """A component that can be temporarily disabled for lesion testing."""

    def __init__(self, name: str, get_state: Callable[[], np.ndarray],
                 set_state: Callable[[np.ndarray], None],
                 zero_fn: Callable[[], None]) -> None:
        self.name = name
        self._get_state = get_state
        self._set_state = set_state
        self._zero_fn = zero_fn
        self._saved_state: Optional[np.ndarray] = None

    def save_and_lesion(self) -> None:
        """Save current state and zero out the component."""
        self._saved_state = self._get_state().copy()
        self._zero_fn()

    def restore(self) -> None:
        """Restore the saved state."""
        if self._saved_state is not None:
            self._set_state(self._saved_state)
            self._saved_state = None


class LesionStudy:
    """Conducts systematic ablation studies on neural components.

    Usage:
        study = LesionStudy()

        # Register components
        study.register_component(LesionableComponent(
            name="sensory_region",
            get_state=lambda: brain._regions["sensory"].state,
            set_state=lambda s: setattr(brain._regions["sensory"], "state", s),
            zero_fn=lambda: brain._regions["sensory"].state.fill(0),
        ))

        # Register probes
        study.register_probe("prediction_error",
            lambda: world_model.get_mean_surprise())

        # Run study
        matrix = study.run()
        print(matrix.get_critical_components())
    """

    def __init__(self, n_probe_steps: int = 20, seed: int = 42) -> None:
        self._components: Dict[str, LesionableComponent] = {}
        self._probes: Dict[str, Callable[[], float]] = {}
        self._step_fn: Optional[Callable[[], None]] = None
        self._n_probe_steps = n_probe_steps
        self._rng = np.random.default_rng(seed)
        self._last_matrix: Optional[LesionMatrix] = None
        self._history: Deque[Dict[str, Any]] = deque(maxlen=20)

        _DATA_DIR.mkdir(parents=True, exist_ok=True)

    def register_component(self, component: LesionableComponent) -> None:
        """Register a component for lesion testing."""
        self._components[component.name] = component

    def register_probe(self, name: str, fn: Callable[[], float]) -> None:
        """Register a behavioral probe function."""
        self._probes[name] = fn

    def set_step_function(self, fn: Callable[[], None]) -> None:
        """Set the function that advances the system by one step."""
        self._step_fn = fn

    def _run_probes(self) -> Dict[str, float]:
        """Run all registered probes and return their values."""
        results = {}
        for name, fn in self._probes.items():
            try:
                results[name] = float(fn())
            except (TypeError, ValueError, ArithmeticError) as exc:
                logger.debug("Probe '%s' failed: %s", name, exc)
                results[name] = 0.0
        return results

    def _run_steps_and_probe(self) -> Dict[str, float]:
        """Run N steps, then probe. Returns averaged probe results."""
        accumulated: Dict[str, List[float]] = {n: [] for n in self._probes}

        for _ in range(self._n_probe_steps):
            if self._step_fn:
                self._step_fn()
            values = self._run_probes()
            for name, val in values.items():
                accumulated[name].append(val)

        return {
            name: float(np.mean(vals)) if vals else 0.0
            for name, vals in accumulated.items()
        }

    def run(self) -> LesionMatrix:
        """Execute the full lesion study.

        Returns:
            LesionMatrix with component x metric impact scores.
        """
        t_start = time.monotonic()

        if not self._components:
            raise ValueError("No components registered for lesion study")
        if not self._probes:
            raise ValueError("No probes registered for lesion study")

        comp_names = sorted(self._components.keys())
        metric_names = sorted(self._probes.keys())

        logger.info("Starting lesion study: %d components, %d probes",
                     len(comp_names), len(metric_names))

        # 1. Baseline measurement
        baseline = self._run_steps_and_probe()
        logger.info("Baseline: %s", {k: round(v, 4) for k, v in baseline.items()})

        # 2. Lesion each component
        results: List[LesionResult] = []
        matrix = np.zeros((len(comp_names), len(metric_names)), dtype=np.float64)

        for c_idx, comp_name in enumerate(comp_names):
            component = self._components[comp_name]

            # Save and lesion
            component.save_and_lesion()

            # Measure with lesion
            lesioned = self._run_steps_and_probe()

            # Restore
            component.restore()

            # Run a few recovery steps
            for _ in range(5):
                if self._step_fn:
                    self._step_fn()

            # Compute deltas and relative impact
            deltas = {}
            relative = {}
            for m_idx, metric in enumerate(metric_names):
                base_val = baseline.get(metric, 0.0)
                les_val = lesioned.get(metric, 0.0)
                delta = les_val - base_val
                deltas[metric] = delta
                rel = abs(delta) / max(abs(base_val), 1e-8)
                relative[metric] = rel
                matrix[c_idx, m_idx] = rel

            criticality = float(np.mean(list(relative.values())))

            result = LesionResult(
                component_name=comp_name,
                baseline_metrics=baseline.copy(),
                lesioned_metrics=lesioned,
                deltas=deltas,
                relative_impact=relative,
                criticality_score=criticality,
            )
            results.append(result)

            logger.info(
                "Lesion '%s': criticality=%.4f, deltas=%s",
                comp_name, criticality,
                {k: round(v, 4) for k, v in deltas.items()},
            )

        total_ms = (time.monotonic() - t_start) * 1000

        lm = LesionMatrix(
            components=comp_names,
            metrics=metric_names,
            matrix=matrix,
            results=results,
            total_time_ms=total_ms,
        )

        self._last_matrix = lm
        self._history.append(lm.to_dict())
        self._save(lm)

        logger.info(
            "Lesion study complete: %d components, critical=%s, redundant=%s (%.1fms)",
            len(comp_names), lm.get_critical_components(),
            lm.get_redundant_components(), total_ms,
        )

        return lm

    def _save(self, matrix: LesionMatrix) -> None:
        try:
            with open(_RESULTS_PATH, "w") as f:
                json.dump(matrix.to_dict(), f, indent=2, default=str)
        except (OSError, IOError, TypeError, ValueError) as exc:
            logger.debug("Lesion results save failed: %s", exc)

    def get_status(self) -> Dict[str, Any]:
        return {
            "n_components": len(self._components),
            "n_probes": len(self._probes),
            "components": list(self._components.keys()),
            "probes": list(self._probes.keys()),
            "has_results": self._last_matrix is not None,
            "n_studies": len(self._history),
        }
