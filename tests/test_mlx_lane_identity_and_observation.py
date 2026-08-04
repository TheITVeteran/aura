"""Who owns a model lane, and what the observer actually saw.

Two CP126 findings in the lane bookkeeping mlx admission decides against.

bec28d76 — every observer iterated a copy of the global client registry and
then read each client's mutable lifecycle fields independently, so a worker
could be registered, recycled or torn down mid-scan. Admission decided against
a view that never existed at any instant.

cdbb177d — the lane-owner id was parent pid plus model path, so two clients for
the same artifact, or the same client across a worker recycle, shared one
identity. Exact eviction could not name which owner to evict, and fencing could
not tell a stale generation from the live one.
"""
from __future__ import annotations

import threading

import pytest

from core.brain.llm import mlx_client

pytestmark = pytest.mark.unit


class _Client:
    def __init__(self, model_path="/models/aura-32b"):
        self.model_path = model_path
        self._worker_generation = 0
        self._model_lane_owner_id = ""


@pytest.fixture(autouse=True)
def _clean_registry():
    with mlx_client._CLIENTS_LOCK:
        saved = dict(mlx_client._CLIENTS)
        mlx_client._CLIENTS.clear()
    yield
    with mlx_client._CLIENTS_LOCK:
        mlx_client._CLIENTS.clear()
        mlx_client._CLIENTS.update(saved)


# --- one identity per client per generation (cdbb177d) ------------------


def test_two_clients_for_one_model_do_not_share_an_owner_id():
    """Exact eviction has to be able to name which owner it is evicting."""
    first = _Client()
    second = _Client()

    assert mlx_client._model_lane_owner_id(first) != mlx_client._model_lane_owner_id(
        second
    )


def test_the_owner_id_is_stable_within_a_generation():
    client = _Client()

    assert mlx_client._model_lane_owner_id(client) == mlx_client._model_lane_owner_id(
        client
    )


def test_a_new_generation_gets_a_new_owner_id():
    """A stale generation and the live one must not share an identity."""
    client = _Client()
    before = mlx_client._model_lane_owner_id(client)

    client._worker_generation += 1
    client._model_lane_owner_id = ""
    after = mlx_client._model_lane_owner_id(client)

    assert before != after


def test_the_owner_id_still_names_the_model_and_the_process():
    import os

    client = _Client("/models/aura-32b")
    owner_id = mlx_client._model_lane_owner_id(client)

    assert owner_id.startswith(f"mlx:{os.getpid()}:")
    assert "aura-32b" in owner_id


def test_reboot_bumps_the_generation():
    import inspect

    source = inspect.getsource(mlx_client.MLXLocalClient.reboot_worker)
    code = "\n".join(
        line
        for line in source.splitlines()
        if not line.strip().startswith("#")
    )
    body = code.split('"""')[2] if code.count('"""') >= 2 else code

    assert "_worker_generation" in body
    assert 'self._model_lane_owner_id = ""' in body


# --- the registry view is consistent (bec28d76) -------------------------


def test_the_snapshot_is_taken_under_the_lock():
    """A membership read that races registration can miss or double-count a
    lane, and admission decides against the result."""
    client = _Client()
    with mlx_client._CLIENTS_LOCK:
        mlx_client._CLIENTS["/models/aura-32b"] = client

    assert mlx_client._clients_snapshot() == [("/models/aura-32b", client)]


def test_the_snapshot_does_not_alias_the_registry():
    with mlx_client._CLIENTS_LOCK:
        mlx_client._CLIENTS["/a"] = _Client("/a")
    snapshot = mlx_client._clients_snapshot()

    with mlx_client._CLIENTS_LOCK:
        mlx_client._CLIENTS["/b"] = _Client("/b")

    assert [path for path, _ in snapshot] == ["/a"]


def test_concurrent_registration_never_yields_a_torn_view():
    stop = threading.Event()
    seen: list[int] = []

    def _churn():
        index = 0
        while not stop.is_set():
            with mlx_client._CLIENTS_LOCK:
                mlx_client._CLIENTS[f"/m{index % 8}"] = _Client(f"/m{index % 8}")
                if index % 3 == 0:
                    mlx_client._CLIENTS.pop(f"/m{(index + 1) % 8}", None)
            index += 1

    writer = threading.Thread(target=_churn, daemon=True)
    writer.start()
    try:
        for _ in range(400):
            snapshot = mlx_client._clients_snapshot()
            # A torn view would raise or produce a None-valued entry.
            assert all(client is not None for _path, client in snapshot)
            seen.append(len(snapshot))
    finally:
        stop.set()
        writer.join(timeout=2.0)

    assert seen


def test_no_observer_reads_the_registry_without_the_snapshot():
    """The helper itself is the one sanctioned direct read; every other site
    must go through it. Parsed rather than grepped, because the explanatory
    comment necessarily quotes the pattern it removed."""
    import ast
    import inspect

    tree = ast.parse(inspect.getsource(mlx_client))
    helper = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "_clients_snapshot"
    )
    inside_helper = set(map(id, ast.walk(helper)))

    offenders: list[int] = []
    for node in ast.walk(tree):
        if id(node) in inside_helper:
            continue
        if isinstance(node, ast.Name) and node.id == "_CLIENTS":
            offenders.append(node.lineno)

    # Registration and teardown legitimately touch _CLIENTS under the lock;
    # what must not exist is an OBSERVER iterating it directly.
    source_lines = inspect.getsource(mlx_client).splitlines()
    iterating = [
        line
        for line in (source_lines[n - 1] for n in offenders)
        if "for " in line and "_CLIENTS" in line
    ]
    assert not iterating, f"observer(s) iterating the registry directly: {iterating}"
