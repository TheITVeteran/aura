"""The latent bridge's backward arrow, which could not have worked anywhere.

``AffectiveSteering`` carries substrate state INTO the residual stream. The
LatentBridge is supposed to carry the model's own representations back out.
It was written, tested in isolation, and had no production caller — and
calling it as written would not have helped, for two independent reasons in
``SubstrateInjectionThread._loop``:

1. It resolved the substrate with ``ServiceContainer.get("conscious_substrate")``.
   The readout hooks run in the MLX worker subprocess; the substrate is
   registered in the main runtime. That lookup is None there, always.
2. It injected via ``asyncio.get_running_loop()`` from a plain daemon thread,
   which raises unconditionally. Even in the main process, with the substrate
   present, the injection sat inside a ``try`` that could only take the
   ``except``.

So the coupling was one-way and read as two-way. These tests cover the
transport that closes it and the wiring at both ends.
"""
from __future__ import annotations

import ast
import inspect
import multiprocessing as mp
from pathlib import Path

import pytest

from core.consciousness.latent_readout_channel import (
    MAX_DELTA,
    SLOTS,
    create_channel,
    drain,
    publish_count,
    publish_deltas,
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def channel():
    return create_channel(mp.get_context("spawn"))


class TestTheTransport:
    def test_a_fresh_reader_injects_nothing(self, channel):
        """A baseline read, or the first drain dumps all history as one hit."""
        publish_deltas(channel, {0: 0.3, 1: 0.2})
        deltas, snapshot = drain(channel, None)
        assert deltas == {}
        assert snapshot[0] == pytest.approx(0.3)

    def test_deltas_accumulate_across_worker_cycles(self, channel):
        _, seen = drain(channel, None)
        publish_deltas(channel, {0: 0.10, 1: -0.04})
        publish_deltas(channel, {0: 0.05, 4: 0.20})
        deltas, _ = drain(channel, seen)
        assert deltas[0] == pytest.approx(0.15)
        assert deltas[1] == pytest.approx(-0.04)
        assert deltas[4] == pytest.approx(0.20)

    def test_an_idle_worker_produces_no_injection(self, channel):
        _, seen = drain(channel, None)
        publish_deltas(channel, {0: 0.1})
        _, seen = drain(channel, seen)
        deltas, _ = drain(channel, seen)
        assert deltas == {}

    def test_runaway_feedback_saturates_rather_than_escalating(self, channel):
        _, seen = drain(channel, None)
        publish_deltas(channel, {0: 9000.0})
        deltas, _ = drain(channel, seen)
        assert deltas[0] == pytest.approx(MAX_DELTA)

    def test_a_missed_read_is_recovered_not_dropped(self, channel):
        """Cumulative totals: a reader that skips a cycle still gets the sum."""
        _, seen = drain(channel, None)
        for _ in range(5):
            publish_deltas(channel, {2: 0.02})
        deltas, _ = drain(channel, seen)
        assert deltas[2] == pytest.approx(0.10)

    def test_out_of_range_indices_are_ignored(self, channel):
        _, seen = drain(channel, None)
        assert publish_deltas(channel, {SLOTS + 5: 1.0, 3: 0.1})
        deltas, _ = drain(channel, seen)
        assert deltas == {3: pytest.approx(0.1)}

    def test_nan_never_poisons_a_slot(self, channel):
        _, seen = drain(channel, None)
        publish_deltas(channel, {1: float("nan"), 2: 0.05})
        deltas, _ = drain(channel, seen)
        assert 1 not in deltas
        assert deltas[2] == pytest.approx(0.05)

    def test_a_missing_channel_is_silent_rather_than_fatal(self):
        assert publish_deltas(None, {0: 1.0}) is False
        assert drain(None, None) == ({}, [0.0] * SLOTS)
        assert publish_count(None) == 0


def _executable_source(cls) -> str:
    """The class's code with every docstring removed.

    Both corrections below QUOTE the bug they fixed, so a plain substring
    search over the source finds the explanation and calls it the defect.
    Strip the prose and search what actually runs.
    """
    tree = ast.parse(inspect.getsource(cls).lstrip())
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            body = node.body
            if (
                body
                and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)
            ):
                node.body = body[1:] or [ast.Pass()]
    return ast.unparse(ast.fix_missing_locations(tree))


class TestTheOldBugsAreGone:
    def test_the_publisher_does_not_resolve_the_substrate_in_the_worker(self):
        from core.consciousness import latent_bridge

        code = _executable_source(latent_bridge.SubstrateInjectionThread)
        assert "conscious_substrate" not in code, (
            "the readout thread is looking the substrate up in its own "
            "process again; in the MLX worker that is always None"
        )

    def test_the_publisher_does_not_need_a_running_loop(self):
        from core.consciousness import latent_bridge

        code = _executable_source(latent_bridge.SubstrateInjectionThread)
        assert "get_running_loop" not in code, (
            "asyncio.get_running_loop() from a plain daemon thread raises "
            "unconditionally — that is how this injected nothing for years"
        )

    def test_the_publisher_refuses_to_run_without_a_transport(self):
        """Collecting deltas and dropping them is the failure being fixed."""
        from core.consciousness.latent_bridge import SubstrateInjectionThread

        thread = SubstrateInjectionThread([], channel=None)
        thread.start()
        assert thread._running is False
        assert thread.get_diagnostics()["has_channel"] is False


class TestBothEndsAreWired:
    def test_the_parent_allocates_the_channel_before_the_fork(self):
        source = (ROOT / "core" / "brain" / "llm" / "mlx_client.py").read_text(
            encoding="utf-8"
        )
        assert "_create_latent_channel(self._mp_context)" in source

    def test_the_channel_is_passed_to_the_worker(self):
        source = (ROOT / "core" / "brain" / "llm" / "mlx_client.py").read_text(
            encoding="utf-8"
        )
        tree = ast.parse(source)
        passed = any(
            isinstance(node, ast.Attribute) and node.attr == "_latent_readout_mem"
            for node in ast.walk(tree)
        )
        assert passed

    def test_the_worker_attaches_the_bridge(self):
        source = (ROOT / "core" / "brain" / "llm" / "mlx_worker.py").read_text(
            encoding="utf-8"
        )
        assert "attach_latent_bridge(model, channel=latent_readout_mem)" in source
        assert "bridge.start_substrate_sync(channel=latent_readout_mem)" in source

    def test_a_failed_bridge_does_not_take_inference_down(self):
        """Unsteered inference is a governance failure. A missing backward
        arrow is a lost feedback loop, and answering beats not answering."""
        source = (ROOT / "core" / "brain" / "llm" / "mlx_worker.py").read_text(
            encoding="utf-8"
        )
        attach_block = source.split("attach_latent_bridge(model")[1][:1400]
        assert "forward-only" in attach_block
        assert "raise RuntimeError" not in attach_block

    def test_the_parent_injects_on_the_generation_path(self):
        from core.brain.llm.mlx_client import MLXLocalClient

        assert hasattr(MLXLocalClient, "_drain_latent_readouts")
        source = inspect.getsource(MLXLocalClient.generate)
        assert "_drain_latent_readouts" in source

    def test_the_injection_is_awaited_not_fired_and_forgotten(self):
        from core.brain.llm.mlx_client import MLXLocalClient

        source = inspect.getsource(MLXLocalClient.generate)
        assert "await self._drain_latent_readouts()" in source, (
            "an un-awaited injection can land in the middle of the next turn"
        )
