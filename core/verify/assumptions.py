"""What the checks assume — the axis the evidence taxonomy does not cover.

``core/organism/model_validation.py`` already answers *what does this test
establish* (MEASURED_LIVE, MEASURED_SYNTHETIC, UNMEASURED, RETRACTED). That is
the strength of the evidence. It is silent on an orthogonal question that has
sunk more real systems: *what does this check take for granted without
checking it*.

seL4 is the reference practice here. It proves functional correctness of the
kernel and then publishes, prominently and permanently, the list of things the
proof assumes and does not establish — compiler correctness, the hand-written
assembly, the hardware model, the boot code, DMA behaviour. The list is not an
apology. It is what makes the proof a scientific object instead of a slogan: a
reader can see the exact edge where the guarantee stops.

A green gate here means "this held, GIVEN the assumptions". Today that "given"
is unwritten, so a passing suite reads as unconditional and the conditions live
only in whoever wrote it. This registry writes them down.

Three statuses, and the distinction that matters
------------------------------------------------
:attr:`AssumptionStatus.DISCHARGED` means something in this repository actually
checks it, and the checker must be **named and must exist** — an assumption
pointing at a test that was renamed away is worse than an undischarged one,
because it reports as covered. :func:`verify_dischargers` is the gate for
exactly that failure.

:attr:`AssumptionStatus.UNDISCHARGED` means it could be checked here and is
not, and it must say what checking it would take. This is honest debt.

:attr:`AssumptionStatus.OUTSIDE_THE_SYSTEM` is seL4's category: things no
amount of work inside this process can establish, because the subject sits
underneath it. Hardware, kernel, clock, filesystem durability. These never
become DISCHARGED and are not debt — but they are load-bearing, and a reader
is entitled to know they are there.

The failure this prevents
-------------------------
This codebase's recurring defect is "the absence of a check reported as a
passed check". An assumption registry attacks it at the root: the conditions a
proof rests on stop being tacit, and the one condition that always goes stale —
"a named checker exists" — becomes itself machine-checked.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

__all__ = [
    "AssumptionStatus",
    "Assumption",
    "AssumptionRegistry",
    "get_assumption_registry",
    "assume",
    "assumptions_for",
    "verify_dischargers",
    "assumption_report",
]

_ROOT = Path(__file__).resolve().parent.parent.parent

#: ``path::name`` — a pytest-style node id.
_NODE_ID = re.compile(r"^(?P<path>[^:]+\.py)::(?P<name>[A-Za-z_][A-Za-z0-9_]*)$")
#: ``make <target>``
_MAKE_TARGET = re.compile(r"^make\s+(?P<target>[A-Za-z0-9_-]+)$")


class AssumptionStatus(StrEnum):
    """Whether this repository does, could, or never can check an assumption."""

    #: Something here checks it. ``discharged_by`` must name that thing, and
    #: the thing must exist.
    DISCHARGED = "discharged"
    #: Checkable in principle, unchecked in fact. ``note`` says what it'd take.
    UNDISCHARGED = "undischarged"
    #: Not establishable from inside this process at all — the subject is
    #: underneath us. seL4's compiler/hardware category.
    OUTSIDE_THE_SYSTEM = "outside_the_system"


@dataclass(frozen=True)
class Assumption:
    """One thing taken for granted, and the state of taking it for granted."""

    id: str
    statement: str
    #: What stops being true if this assumption is false. The reason to care.
    breaks: str
    status: AssumptionStatus
    owner: str
    #: Subsystem this belongs to, matching invariant scopes where possible.
    scope: str = "system"
    #: Required when DISCHARGED: ``tests/x.py::test_y`` or ``make <target>``.
    discharged_by: str = ""
    #: Required when not DISCHARGED: what checking it would take, or why
    #: nothing here ever could.
    note: str = ""

    def __post_init__(self) -> None:
        if self.status is AssumptionStatus.DISCHARGED:
            if not self.discharged_by.strip():
                raise ValueError(
                    f"assumption {self.id!r} claims to be discharged but names no "
                    "checker; an unnamed discharger cannot be verified and so is "
                    "indistinguishable from none"
                )
        elif not self.note.strip():
            raise ValueError(
                f"assumption {self.id!r} is {self.status.value} and says nothing about "
                "what is missing; an unexplained assumption reads as a discharged one"
            )

    def to_dict(self) -> dict[str, str]:
        return {
            "id": self.id,
            "statement": self.statement,
            "breaks": self.breaks,
            "status": str(self.status),
            "owner": self.owner,
            "scope": self.scope,
            "discharged_by": self.discharged_by,
            "note": self.note,
        }


@dataclass
class AssumptionRegistry:
    """Every assumption the system's guarantees rest on."""

    _by_id: dict[str, Assumption] = field(default_factory=dict)

    def register(self, assumption: Assumption) -> Assumption:
        existing = self._by_id.get(assumption.id)
        if existing is not None and existing != assumption:
            raise ValueError(
                f"assumption id {assumption.id!r} is already registered with a "
                "different body; ids are a contract and must not be reused"
            )
        self._by_id[assumption.id] = assumption
        return assumption

    def all(self) -> list[Assumption]:
        return sorted(self._by_id.values(), key=lambda a: (a.scope, a.id))

    def by_scope(self, scope: str) -> list[Assumption]:
        return [a for a in self.all() if a.scope == scope]

    def by_status(self, status: AssumptionStatus) -> list[Assumption]:
        return [a for a in self.all() if a.status is status]

    def scopes(self) -> list[str]:
        return sorted({a.scope for a in self._by_id.values()})

    def __len__(self) -> int:
        return len(self._by_id)

    def __iter__(self) -> Iterator[Assumption]:
        return iter(self.all())


_REGISTRY = AssumptionRegistry()


def get_assumption_registry() -> AssumptionRegistry:
    return _REGISTRY


def assume(
    assumption_id: str,
    *,
    statement: str,
    breaks: str,
    status: AssumptionStatus,
    owner: str,
    scope: str = "system",
    discharged_by: str = "",
    note: str = "",
) -> Assumption:
    """Declare an assumption next to whatever relies on it."""
    return _REGISTRY.register(
        Assumption(
            id=assumption_id,
            statement=statement,
            breaks=breaks,
            status=status,
            owner=owner,
            scope=scope,
            discharged_by=discharged_by,
            note=note,
        )
    )


def assumptions_for(*scopes: str) -> list[Assumption]:
    """Every assumption in the given scopes, for reporting alongside a result."""
    if not scopes:
        return _REGISTRY.all()
    wanted = set(scopes)
    return [a for a in _REGISTRY.all() if a.scope in wanted]


def _discharger_exists(reference: str, root: Path) -> tuple[bool, str]:
    """Does the named checker actually exist in the tree?

    The one check that matters most, because it is the one that rots. A test
    gets renamed, the assumption still points at the old node id, and the
    registry keeps reporting the assumption as covered forever.
    """
    reference = reference.strip()

    make_match = _MAKE_TARGET.match(reference)
    if make_match:
        makefile = root / "Makefile"
        if not makefile.is_file():
            return False, "no Makefile at the repository root"
        target = make_match.group("target")
        pattern = re.compile(rf"^{re.escape(target)}\s*:", re.MULTILINE)
        if pattern.search(makefile.read_text(encoding="utf-8")):
            return True, ""
        return False, f"Makefile has no target {target!r}"

    node_match = _NODE_ID.match(reference)
    if node_match:
        path = root / node_match.group("path")
        if not path.is_file():
            return False, f"no such file: {node_match.group('path')}"
        name = node_match.group("name")
        source = path.read_text(encoding="utf-8")
        if re.search(rf"^\s*(async\s+)?def\s+{re.escape(name)}\s*\(", source, re.MULTILINE):
            return True, ""
        if re.search(rf"^\s*class\s+{re.escape(name)}\b", source, re.MULTILINE):
            return True, ""
        return False, f"{node_match.group('path')} defines no {name!r}"

    if reference.endswith(".py"):
        # A whole test module is a legitimate checker — several gates here are
        # a file of cases rather than one named function.
        if (root / reference).is_file():
            return True, ""
        return False, f"no such file: {reference}"

    return False, (
        f"unrecognised discharger form {reference!r}; expected "
        "'path/to/test.py', 'path/to/test.py::name' or 'make <target>'"
    )


@dataclass(frozen=True)
class DischargeFailure:
    """A discharged assumption whose named checker could not be found."""

    assumption_id: str
    discharged_by: str
    reason: str

    def __str__(self) -> str:
        return (
            f"{self.assumption_id}: claims discharge by {self.discharged_by!r} "
            f"but {self.reason}"
        )


def verify_dischargers(
    assumptions: Iterable[Assumption] | None = None,
    *,
    root: Path | None = None,
) -> list[DischargeFailure]:
    """Every DISCHARGED assumption must name a checker that exists.

    This is the anti-fiction gate. An assumption pointing at a deleted test is
    strictly worse than an undischarged one: it occupies the slot that would
    otherwise show up as debt.
    """
    base = root or _ROOT
    failures: list[DischargeFailure] = []
    for assumption in assumptions if assumptions is not None else _REGISTRY.all():
        if assumption.status is not AssumptionStatus.DISCHARGED:
            continue
        ok, reason = _discharger_exists(assumption.discharged_by, base)
        if not ok:
            failures.append(
                DischargeFailure(
                    assumption_id=assumption.id,
                    discharged_by=assumption.discharged_by,
                    reason=reason,
                )
            )
    return failures


def assumption_report() -> dict[str, object]:
    """The whole ledger, for ``runtime_health_report`` and the CLI gate."""
    counts = {
        status.value: len(_REGISTRY.by_status(status)) for status in AssumptionStatus
    }
    return {
        "total": len(_REGISTRY),
        "counts": counts,
        "scopes": _REGISTRY.scopes(),
        "assumptions": [a.to_dict() for a in _REGISTRY.all()],
        "discharge_failures": [str(f) for f in verify_dischargers()],
    }
