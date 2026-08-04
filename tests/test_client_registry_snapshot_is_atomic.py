"""Copying the MLX client registry must not race registration.

Live 2026-08-03 22:13, repeatedly, holding the runtime DEGRADED:

    Skynet: Subsystem 'orchestrator' health probe failed (1/2):
    important:unified_runtime_pressure:recent_inference_gate_critical:
    dictionary changed size during iteration

The health probe was honest — it was reporting a real recorded degradation.
The defect was that five observers copied the shared registry with
``dict(_CLIENTS)`` or ``list(_CLIENTS.values())``, and both ITERATE. A client
registered or torn down mid-copy raises. inference_gate is on the fail-closed
list, so the RuntimeError was escalated to CRITICAL.

The lock and an atomic snapshot already existed (CP126 bec28d76); the
observers were reaching past them.
"""
from __future__ import annotations

import threading

import pytest

from core.brain.llm import mlx_client


@pytest.fixture(autouse=True)
def _clean_registry():
    with mlx_client._CLIENTS_LOCK:
        saved = dict(mlx_client._CLIENTS)
    yield
    with mlx_client._CLIENTS_LOCK:
        mlx_client._CLIENTS.clear()
        mlx_client._CLIENTS.update(saved)


class TestTheSnapshotSurvivesConcurrentRegistration:
    def test_copying_under_churn_never_raises(self):
        stop = threading.Event()
        errors: list[BaseException] = []

        def churn() -> None:
            index = 0
            while not stop.is_set():
                with mlx_client._CLIENTS_LOCK:
                    mlx_client._CLIENTS[f"/model/{index}"] = object()
                    if index > 40:
                        mlx_client._CLIENTS.pop(f"/model/{index - 40}", None)
                index += 1

        def read() -> None:
            try:
                for _ in range(4000):
                    dict(mlx_client.clients_snapshot())
            except BaseException as exc:  # noqa: BLE001 — the point of the test
                errors.append(exc)

        writer = threading.Thread(target=churn, daemon=True)
        writer.start()
        try:
            reader = threading.Thread(target=read, daemon=True)
            reader.start()
            reader.join(60)
            assert not reader.is_alive()
        finally:
            stop.set()
            writer.join(5)

        assert not errors, f"snapshot raced registration: {errors[:1]}"

    def test_it_returns_the_membership(self):
        with mlx_client._CLIENTS_LOCK:
            mlx_client._CLIENTS.clear()
            mlx_client._CLIENTS["/a"] = object()
            mlx_client._CLIENTS["/b"] = object()
        assert {path for path, _client in mlx_client.clients_snapshot()} == {"/a", "/b"}


class TestNoObserverCopiesTheRegistryDirectly:
    """A ratchet: the next observer must use the snapshot too."""

    OBSERVERS = (
        "core/brain/inference_gate.py",
        "core/brain/llm/batch_candidates.py",
        "core/runtime/runtime_pressure.py",
        "core/resilience/memory_governor.py",
    )

    @pytest.mark.parametrize("path", OBSERVERS)
    def test_no_iterating_copy_of_the_shared_registry(self, path):
        from pathlib import Path

        source = Path(path).read_text(encoding="utf-8")
        for forbidden in ("dict(_CLIENTS)", "list(_CLIENTS.values())", "_CLIENTS.items()"):
            offending = [
                line
                for line in source.split("\n")
                if forbidden in line and not line.lstrip().startswith("#")
            ]
            assert not offending, f"{path} copies the live registry: {offending}"


class TestDeepSnapshotSurvivesLiveMutation:
    """The evidence path walks provider metadata other threads are writing.

    _deep_snapshot_inner comprehended over ``value.items()`` with no copy. A
    Python-level comprehension CAN be interrupted between elements, unlike the
    C-level ``dict(d)``, so a live metadata dict growing mid-walk raised inside
    an evidence path — in a fail-closed subsystem, which escalated it to
    CRITICAL and held the runtime DEGRADED.
    """

    @staticmethod
    def _churn(shared: dict, stop: threading.Event) -> None:
        index = 3000
        while not stop.is_set():
            shared[f"k{index}"] = {"n": index}
            shared.pop(f"k{index - 2000}", None)
            index += 1

    def test_snapshotting_a_dict_under_churn_never_raises(self):
        from core.brain.inference_gate import _deep_snapshot

        shared = {f"k{i}": {"n": i} for i in range(3000)}
        stop = threading.Event()
        errors: list[str] = []

        def snap() -> None:
            try:
                for _ in range(200):
                    _deep_snapshot(shared)
            except BaseException as exc:  # noqa: BLE001 — the point of the test
                errors.append(f"{type(exc).__name__}: {exc}")

        writer = threading.Thread(target=self._churn, args=(shared, stop), daemon=True)
        writer.start()
        try:
            reader = threading.Thread(target=snap, daemon=True)
            reader.start()
            reader.join(90)
            assert not reader.is_alive()
        finally:
            stop.set()
            writer.join(5)

        assert not errors, f"the evidence path raced live metadata: {errors[:1]}"

    def test_the_walk_copies_before_iterating(self):
        import inspect

        from core.brain import inference_gate

        body = inspect.getsource(inference_gate._deep_snapshot_inner)
        assert "list(value.items())" in body, "iterating a live dict is the defect"
        assert "for key, item in value.items()" not in body

    def test_it_still_copies_containers_and_keeps_leaves(self):
        from core.brain.inference_gate import _deep_snapshot

        source = {"a": {"b": [1, 2]}, "n": 5}
        copied = _deep_snapshot(source)
        assert copied == source
        assert copied["a"] is not source["a"]
        assert copied["a"]["b"] is not source["a"]["b"]
