"""A gate whose failure equals its approval is not a gate.

The CP126 inventory carries 93 open fail-open findings in `core/`, 66 of them
critical, and they are one defect wearing different clothes: the check could
not be evaluated, so the action proceeded. The absence of a check reported as
a passed check — the shape this codebase keeps rediscovering.

The four fixed here are the ones where the thing that proceeded was Aura
acting on the world without being asked:

  * `proactive_communication` caught the failure and set
    `autonomous_admitted = True` — the admission gate becoming unreachable
    was *literally spelled* as the gate approving.
  * `proactive_presence` recorded the degradation and fell through on both
    the pre-generation and pre-emission probes, so an unsolicited message
    could reach the owner's screen with the idle window, memory ceiling,
    failure-pressure ceiling and user-anchor rule all silently not applying.
  * `proactive_agency` recorded the exception and continued into planning
    and pursuit.
  * `artifact_builder` caught every exception from the governed write —
    including a governance REFUSAL — and then performed the same write
    directly through `atomic_write_text`. Not a fallback: a bypass to the
    identical target, with a log line.

Each test drives the failure and asserts the action did NOT happen.
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


# ────────────────────────────── proactive agency stops on a raising gate


@pytest.mark.asyncio
async def test_a_raising_background_gate_stops_the_pursuit():
    from core.agency.proactive_agency import ProactiveAgency

    planned: list[str] = []

    def _explode() -> bool:
        raise RuntimeError("gate unavailable")

    agency = ProactiveAgency(
        pursuit=None,
        planner=lambda: planned.append("planned"),
        background_allowed=_explode,
    )

    result = await agency.pursue_goal("do something unprompted")

    assert result is None
    assert planned == [], "a pursuit ran because its gate raised"


@pytest.mark.asyncio
async def test_a_refusing_background_gate_still_stops_the_pursuit():
    """The ordinary path must keep working after the fail-closed change."""
    from core.agency.proactive_agency import ProactiveAgency

    planned: list[str] = []
    agency = ProactiveAgency(
        pursuit=None,
        planner=lambda: planned.append("planned"),
        background_allowed=lambda: False,
    )

    assert await agency.pursue_goal("goal") is None
    assert planned == []


# ─────────────────────────── the artifact write does not bypass a refusal


def test_a_refused_governed_write_does_not_fall_back_to_a_direct_write(
    tmp_path, monkeypatch
):
    """The bypass, driven.

    A gateway that refuses must not be answered by writing the same bytes to
    the same path through a different door.
    """
    from core.actuators import artifact_builder

    class _RefusingGateway:
        def write_text(self, *a, **k):
            raise RuntimeError("governance refused this write")

    monkeypatch.setattr(
        "core.runtime.file_write_gateway.get_file_write_gateway",
        lambda: _RefusingGateway(),
    )

    target = tmp_path / "artifact.txt"
    ok = artifact_builder._write(target, "contents")

    assert ok is False, "a refused write reported success"
    assert not target.exists(), (
        "the refused write happened anyway through the atomic fallback"
    )


def test_a_missing_gateway_still_allows_the_offline_fallback(tmp_path, monkeypatch):
    """The fallback exists for a real reason and must survive.

    This module is used outside a runtime, where the gateway genuinely is not
    importable. Closing that path would trade a bypass for a broken tool.
    """
    import builtins

    from core.actuators import artifact_builder

    real_import = builtins.__import__

    def _no_gateway(name, *args, **kwargs):
        if name in {"core.governance_context", "core.runtime.file_write_gateway"}:
            raise ImportError("not in a runtime")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _no_gateway)

    target = tmp_path / "offline.txt"
    ok = artifact_builder._write(target, "contents")

    monkeypatch.setattr(builtins, "__import__", real_import)
    assert ok is True
    assert target.read_text() == "contents"


# ──────────────────────────────────── the structural guard on the class


#: An ASSIGNMENT granting permission. Deliberately not matching keyword
#: arguments: `DispatchOutcome(approved=True, dispatched=False)` reports that
#: the Will approved and dispatch then failed, which is an accurate record
#: rather than a grant, and flagging it would train the reader to ignore this.
_FAIL_OPEN_ASSIGN = re.compile(
    r"^\s*\w*(admitted|allowed|permitted|approved|authori[sz]ed)\s*=\s*True\s*(#.*)?$",
    re.IGNORECASE | re.MULTILINE,
)


def _handlers_of(path: Path) -> list[tuple[int, ast.ExceptHandler]]:
    try:
        tree = ast.parse(path.read_text("utf-8", errors="ignore"))
    except SyntaxError:
        return []
    return [
        (node.lineno, node)
        for node in ast.walk(tree)
        if isinstance(node, ast.ExceptHandler)
    ]


def test_no_exception_handler_grants_permission():
    """`except ...: allowed = True` is the defect in its purest form.

    Nothing else in the codebase should spell "the check failed, so proceed"
    quite this literally, and this is the cheapest place to catch the next
    one.
    """
    offenders: list[str] = []
    for path in (ROOT / "core").rglob("*.py"):
        if "__pycache__" in str(path):
            continue
        source = path.read_text("utf-8", errors="ignore")
        if not _FAIL_OPEN_ASSIGN.search(source):
            continue
        lines = source.splitlines()
        for lineno, handler in _handlers_of(path):
            end = getattr(handler, "end_lineno", lineno) or lineno
            body = "\n".join(lines[lineno - 1 : end])
            if _FAIL_OPEN_ASSIGN.search(body):
                offenders.append(f"{path.relative_to(ROOT)}:{lineno}")

    assert offenders == [], (
        "these exception handlers grant permission when a check fails, so the "
        f"gate being unavailable has the same effect as the gate approving: {offenders}"
    )


@pytest.mark.parametrize(
    "rel,marker",
    [
        ("core/autonomy/proactive_communication.py", "autonomous_admitted = False"),
        ("core/agency/proactive_agency.py", "return None"),
    ],
)
def test_the_fixed_sites_still_fail_closed(rel: str, marker: str):
    """Pin the specific fixes, so a refactor cannot quietly restore them."""
    source = (ROOT / rel).read_text("utf-8")

    assert marker in source, f"{rel} no longer fails closed"


def test_the_presence_probes_return_rather_than_continue():
    """Both probes must stop the emission, not annotate it."""
    source = (ROOT / "core" / "autonomy" / "proactive_presence.py").read_text("utf-8")

    failures: list[str] = []
    for marker in (
        "Background policy check failed",
        "Visible emission policy check failed",
    ):
        index = source.index(marker)
        following = source[index : index + 400]
        if not re.search(r"\n\s+return\b", following):
            failures.append(marker)

    assert failures == [], (
        f"these policy probes record the failure and continue: {failures}"
    )
