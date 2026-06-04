"""Phenomenal Consciousness Test Harness.

Shared infrastructure for the 10-test battery:
- Receipt generation (RECEIPTS.jsonl compatible)
- AuraNow state factory
- Perturbation engine (sealed/revealed)
- Scoring and statistical helpers
- Artifact/output directory management
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import secrets
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]

# ── Receipt infrastructure ─────────────────────────────────────────────

@dataclass
class Receipt:
    """Immutable receipt for any consequential test action."""
    receipt_id: str = field(default_factory=lambda: secrets.token_hex(12))
    receipt_type: str = "GenericReceipt"
    timestamp: float = field(default_factory=time.time)
    test_name: str = ""
    phase: str = ""
    payload: dict[str, Any] = field(default_factory=dict)
    state_hash: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_jsonl(self) -> str:
        return json.dumps(self.to_dict(), default=str)


class ReceiptLog:
    """Append-only receipt journal for a test run."""

    def __init__(self, path: Path | None = None):
        self.path = path or REPO_ROOT / "artifacts" / "phenomenal" / "RECEIPTS.jsonl"
        self.entries: list[Receipt] = []

    def record(self, receipt: Receipt) -> Receipt:
        self.entries.append(receipt)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(receipt.to_jsonl() + "\n")
        return receipt

    def query(self, **filters: Any) -> list[Receipt]:
        results = []
        for r in self.entries:
            match = True
            for k, v in filters.items():
                if getattr(r, k, None) != v:
                    match = False
                    break
            if match:
                results.append(r)
        return results


# ── State hash helper ──────────────────────────────────────────────────

def hash_state(data: dict[str, Any]) -> str:
    encoded = json.dumps(data, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:24]


# ── AuraNow factory for testing ────────────────────────────────────────

def make_aura_now(
    *,
    tick: int = 0,
    valence: float = 0.0,
    arousal: float = 0.5,
    distress: float = 0.0,
    curiosity: float = 0.5,
    free_energy: float = 0.0,
    cpu_pressure: float = 0.0,
    memory_pressure: float = 0.0,
    workspace_winner: str = "",
    ignition_strength: float = 0.0,
    broadcast_targets: tuple[str, ...] = (),
    agency_confidence: float = 0.5,
    controllability: float = 0.5,
    dominance: float = 0.5,
    boredom: float = 0.0,
    care: float = 0.0,
    identity_stability: float = 1.0,
    continuity_risk: float = 0.0,
    workspace_lesion: str = "",
    attribution: str = "mixed",
    dominant_drive: str = "coherence",
    focal_object: str = "",
    task_active: bool = False,
    will_confidence: float = 0.7,
):
    """Create an AuraNow snapshot with specified overrides."""
    from core.being.aura_now import (
        AffectiveState,
        AttentionState,
        AuraNow,
        BodyState,
        MemoryContext,
        OwnershipState,
        PredictionState,
        ReportBoundary,
        SelfState,
        WillStateSnapshot,
        WorkspaceState,
        WorldState,
    )

    now_ts = time.time()
    body = BodyState(cpu_pressure=cpu_pressure, memory_pressure=memory_pressure)
    world = WorldState(focal_object=focal_object, task_active=task_active)
    affect = AffectiveState(
        valence=valence,
        arousal=arousal,
        distress=distress,
        curiosity=curiosity,
        dominance=dominance,
        boredom=boredom,
        care=care,
        free_energy=free_energy,
        dominant_drive=dominant_drive,
    )
    attention = AttentionState(focal_object=workspace_winner, stability=ignition_strength)
    self_model = SelfState(
        identity_stability=identity_stability,
        continuity_risk=continuity_risk,
    )
    memory_ctx = MemoryContext()
    workspace = WorkspaceState(
        winner=workspace_winner,
        ignition_strength=ignition_strength,
        broadcast_targets=broadcast_targets,
        lesion=workspace_lesion,
    )
    will = WillStateSnapshot(confidence=will_confidence)
    prediction = PredictionState(
        free_energy=free_energy,
        controllability=controllability,
    )
    ownership = OwnershipState(
        agency_confidence=agency_confidence,
        attribution=attribution,
    )
    report_boundary = ReportBoundary()

    return AuraNow(
        tick=tick,
        timestamp=now_ts,
        monotonic_time=time.monotonic(),
        continuous_field=(valence, arousal, distress, curiosity, free_energy),
        body=body,
        world=world,
        attention=attention,
        affect=affect,
        self_model=self_model,
        memory_context=memory_ctx,
        workspace=workspace,
        will=will,
        prediction=prediction,
        ownership=ownership,
        report_boundary=report_boundary,
    )


# ── Perturbation Engine ────────────────────────────────────────────────

@dataclass
class Perturbation:
    """A hidden change to internal state for blind testing."""
    trial_id: int
    changes: dict[str, float]
    seed_hex: str = field(default_factory=lambda: secrets.token_hex(8))
    applied_at: float = 0.0

    def commitment_hash(self) -> str:
        """Sealed hash that doesn't reveal the actual changes."""
        payload = json.dumps(
            {"trial": self.trial_id, "seed": self.seed_hex, "keys": sorted(self.changes.keys())},
            sort_keys=True,
        ).encode()
        return hashlib.sha256(payload).hexdigest()[:24]


class PerturbationEngine:
    """Generates and tracks hidden internal-state perturbations."""

    def __init__(self, rng_seed: int | None = None):
        import random
        self._rng = random.Random(rng_seed or secrets.randbits(64))
        self.schedule: list[Perturbation] = []
        self.committed_hashes: list[str] = []

    def generate_perturbation(
        self,
        trial_id: int,
        dimensions: list[str] | None = None,
    ) -> Perturbation:
        dims = dimensions or [
            "valence", "arousal", "distress", "curiosity",
            "free_energy", "cpu_pressure", "memory_pressure",
            "agency_confidence", "controllability", "dominance",
        ]
        # Pick 2-5 dimensions to perturb
        n = self._rng.randint(2, min(5, len(dims)))
        chosen = self._rng.sample(dims, n)
        changes = {}
        for dim in chosen:
            delta = round(self._rng.uniform(-0.5, 0.5), 3)
            if abs(delta) < 0.05:
                delta = 0.15 * (1 if delta >= 0 else -1)
            changes[dim] = delta
        p = Perturbation(trial_id=trial_id, changes=changes)
        self.schedule.append(p)
        self.committed_hashes.append(p.commitment_hash())
        return p

    def apply_perturbation(self, perturbation: Perturbation, base_kwargs: dict[str, Any]) -> dict[str, Any]:
        """Apply perturbation to base AuraNow kwargs, clamping to [0, 1]."""
        result = dict(base_kwargs)
        for dim, delta in perturbation.changes.items():
            current = result.get(dim, 0.5)
            result[dim] = max(0.0, min(1.0, current + delta))
        perturbation.applied_at = time.time()
        return result

    def generate_sham(self, trial_id: int) -> Perturbation:
        """Generate a sham perturbation (no actual changes) for control trials."""
        p = Perturbation(trial_id=trial_id, changes={})
        self.schedule.append(p)
        self.committed_hashes.append(p.commitment_hash())
        return p


# ── Scoring helpers ────────────────────────────────────────────────────

def accuracy_score(predictions: list[float], actuals: list[float], threshold: float = 0.15) -> float:
    """Fraction of predictions within threshold of actual values."""
    if not predictions or len(predictions) != len(actuals):
        return 0.0
    correct = sum(1 for p, a in zip(predictions, actuals) if abs(p - a) <= threshold)
    return correct / len(predictions)


def direction_accuracy(predicted_deltas: list[float], actual_deltas: list[float]) -> float:
    """Fraction where predicted and actual deltas share the same sign."""
    if not predicted_deltas or len(predicted_deltas) != len(actual_deltas):
        return 0.0
    correct = sum(
        1 for p, a in zip(predicted_deltas, actual_deltas)
        if (p > 0 and a > 0) or (p < 0 and a < 0) or (abs(p) < 0.05 and abs(a) < 0.05)
    )
    return correct / len(predicted_deltas)


def cohens_d(group_a: list[float], group_b: list[float]) -> float:
    """Effect size between two groups."""
    if len(group_a) < 2 or len(group_b) < 2:
        return 0.0
    mean_a = sum(group_a) / len(group_a)
    mean_b = sum(group_b) / len(group_b)
    mean_delta = mean_a - mean_b
    var_a = sum((x - mean_a) ** 2 for x in group_a) / (len(group_a) - 1)
    var_b = sum((x - mean_b) ** 2 for x in group_b) / (len(group_b) - 1)
    pooled_std = math.sqrt((var_a + var_b) / 2)
    if pooled_std <= 1e-12:
        if abs(mean_delta) <= 1e-12:
            return 0.0
        return math.copysign(min(abs(mean_delta) / 1e-6, 1_000_000.0), mean_delta)
    return mean_delta / pooled_std


def consistency_score(labels_1: list[str], labels_2: list[str]) -> float:
    """Fraction of matching labels between two label sequences."""
    if not labels_1 or len(labels_1) != len(labels_2):
        return 0.0
    matches = sum(1 for a, b in zip(labels_1, labels_2) if a == b)
    return matches / len(labels_1)


# ── Output directory management ───────────────────────────────────────

def get_run_dir(test_name: str) -> Path:
    """Get output directory for a test run."""
    run_id = f"run_{int(time.time())}_{secrets.token_hex(4)}"
    base = Path(os.environ.get(
        "AURA_PHENOMENAL_OUT",
        str(REPO_ROOT / "artifacts" / "phenomenal"),
    ))
    out = base / test_name / run_id
    out.mkdir(parents=True, exist_ok=True)
    return out
