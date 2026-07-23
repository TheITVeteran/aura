"""CP126: memory-pressure guards cannot be lifted by self-assertion.

The critical-pressure refusal is the last guard before the model process can
push macOS into swap or jetsam, so what is allowed to waive it matters.
"""
from __future__ import annotations

import inspect

from core.brain.llm import mlx_client


def _generate_source() -> str:
    return inspect.getsource(mlx_client.MLXLocalClient.generate)


class TestBenchmarkLabelCannotWaiveSafety:
    def test_safety_guards_use_the_explicit_flag_only(self):
        """Label inference may set scheduling, never the safety exemption.

        benchmark_request is inferred from any free-form purpose containing
        "_baseline"; that inference must not reach a guard that refuses
        generation under critical memory pressure.
        """
        source = _generate_source()
        assert "benchmark_request_explicit = bool(kwargs.get(\"benchmark_request\", False))" in source

        # The guards were refactored so both the override branch and the
        # refusal branch key off one `override_applies` predicate. Assert that
        # PREDICATE carries the explicit-flag binding (and that both branches
        # use it) rather than pinning the old inline formatting — the property
        # under test is "inference cannot waive the safety exemption", not the
        # line layout that expressed it in one earlier revision.
        assert (
            "override_applies = (\n"
            "                memory_snapshot.refuse_heavy_local_generation\n"
            "                and self._is_primary_or_deep_lane()\n"
            "                and not benchmark_request_explicit\n"
            "            )"
        ) in source, "critical-pressure override predicate is not bound to the explicit flag"
        for marker in (
            "if override_applies and critical_override:",
            "if override_applies and not critical_override:",
            "if self._is_primary_or_deep_lane() and not benchmark_request_explicit:",
        ):
            assert marker in source, f"pressure guard not bound to the explicit flag: {marker!r}"

    def test_label_inference_still_classifies_scheduling(self):
        source = _generate_source()
        # The benign scheduling behaviour is retained.
        assert "benchmark_request = benchmark_request_explicit or (" in source
        assert "if benchmark_request:\n            request_is_background = False" in source


class TestUnobservablePressureFailsClosed:
    def test_blind_probe_refuses_heavy_generation(self):
        source = _generate_source()
        # A probe that cannot answer is not evidence of headroom.
        assert "memory_pressure_unobservable_refused_generation" in source
        assert "refused heavy local generation because memory pressure could not be observed" in source

    def test_blind_probe_catches_more_than_os_errors(self):
        source = _generate_source()
        assert "except (OSError, AttributeError, RuntimeError, TypeError, ValueError) as exc:" in source


class TestCriticalOverrideIsLoud:
    def test_override_records_a_critical_receipt(self):
        source = _generate_source()
        assert "memory_pressure_generation_override" in source
        # It must be as loud as the refusal it replaces.
        assert "logger.critical(" in source
