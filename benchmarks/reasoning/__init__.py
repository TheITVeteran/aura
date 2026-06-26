"""Reasoning benchmark harness — keep the amplifier honest.

Every reasoning upgrade must show its work: pass rate, verifier-catch rate, false
confidence, hallucination catch, and latency on a fixed battery of known-answer
cases. Without this you fool yourself; with it you can actually climb.

Run deterministically (canned candidates exercise the verifiers/amplifier) or
against the live model with ``--live``.
"""
from __future__ import annotations

from .harness import BenchmarkResult, ReasoningBenchmark, run_benchmark
from .suites import ReasoningCase, default_suite

__all__ = [
    "BenchmarkResult",
    "ReasoningBenchmark",
    "run_benchmark",
    "ReasoningCase",
    "default_suite",
]
