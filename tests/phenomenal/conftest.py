"""Shared fixtures for the Phenomenal Consciousness Test Battery."""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from tests.phenomenal.harness import (
    PerturbationEngine,
    ReceiptLog,
    make_aura_now,
)


@pytest.fixture
def receipt_log(tmp_path):
    """Append-only receipt log per test."""
    configured_path = os.environ.get("AURA_PHENOMENAL_RECEIPTS")
    return ReceiptLog(Path(configured_path) if configured_path else tmp_path / "RECEIPTS.jsonl")


@pytest.fixture
def perturbation_engine():
    """Deterministic perturbation engine with fixed seed."""
    return PerturbationEngine(rng_seed=42)


@pytest.fixture
def run_dir(tmp_path):
    """Temp directory for test artifacts."""
    d = tmp_path / "phenomenal_run"
    d.mkdir(parents=True, exist_ok=True)
    return d


@pytest.fixture
def baseline_state():
    """Default AuraNow baseline."""
    return make_aura_now(
        tick=0,
        valence=0.0,
        arousal=0.5,
        distress=0.0,
        curiosity=0.5,
        free_energy=0.0,
        agency_confidence=0.5,
        controllability=0.5,
    )
