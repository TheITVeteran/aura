"""core/verify/runtime_invariants.py — the standing invariants.

These are the structural facts the runtime assumes everywhere and checks
nowhere. Each one has been true by convention; a convention that nothing
enforces is a convention that a refactor silently retires.

Grouped by scope so `-verify-each` can re-check only what a mutation could
have broken. Importing this module registers them; :mod:`core.runtime.foundations`
does that once at boot.
"""

from __future__ import annotations

from collections.abc import Iterator

from core.verify.invariants import Severity, Violation, invariant

_OWNER = "core/verify/runtime_invariants.py"


# ══════════════════════════════════════════════════════════════════════
# Service container — the spine. Everything else resolves through it.
# ══════════════════════════════════════════════════════════════════════

def _container_state() -> tuple[dict[str, object], dict[str, str]]:
    from core.container import ServiceContainer

    services = dict(getattr(ServiceContainer, "_services", {}) or {})
    aliases = dict(getattr(ServiceContainer, "_aliases", {}) or {})
    return services, aliases


@invariant(
    "container.alias_resolves",
    scope="container",
    owner=_OWNER,
    description="every registered alias resolves to a registered service",
)
def _alias_resolves() -> Iterator[Violation]:
    services, aliases = _container_state()
    for alias, target in aliases.items():
        seen: set[str] = set()
        current = target
        while current in aliases and current not in seen:
            seen.add(current)
            current = aliases[current]
        if current not in services:
            yield Violation(
                subject=alias,
                message=(
                    f"alias {alias!r} resolves to {current!r}, which is not a "
                    "registered service — every lookup through it raises or "
                    "silently returns the default"
                ),
                remedy=f"register {current!r}, or drop the alias",
            )


@invariant(
    "container.alias_terminates",
    scope="container",
    owner=_OWNER,
    description="alias chains are acyclic",
)
def _alias_terminates() -> Iterator[Violation]:
    _services, aliases = _container_state()
    for alias in aliases:
        seen: set[str] = {alias}
        current = aliases[alias]
        while current in aliases:
            if current in seen:
                yield Violation(
                    subject=alias,
                    message=(
                        f"alias chain from {alias!r} cycles at {current!r}; "
                        "resolution never terminates"
                    ),
                    remedy="break the cycle — one of these must point at a real service",
                )
                break
            seen.add(current)
            current = aliases[current]


@invariant(
    "container.declared_dependencies_exist",
    scope="container",
    owner=_OWNER,
    description="every declared service dependency names something registered",
)
def _dependencies_exist() -> Iterator[Violation]:
    services, aliases = _container_state()
    known = set(services) | set(aliases)
    for name, descriptor in services.items():
        for dependency in list(getattr(descriptor, "dependencies", ()) or ()):
            if str(dependency) not in known:
                yield Violation(
                    subject=f"{name} -> {dependency}",
                    message=(
                        f"service {name!r} declares a dependency on {dependency!r}, "
                        "which is not registered; construction will fail the first "
                        "time this service is actually needed"
                    ),
                    remedy=f"register {dependency!r} before {name!r}, or drop the declaration",
                )


@invariant(
    "container.dependency_graph_acyclic",
    scope="container",
    owner=_OWNER,
    description="declared service dependencies form a DAG",
)
def _dependency_graph_acyclic() -> Iterator[Violation]:
    services, aliases = _container_state()

    def resolve(name: str) -> str:
        seen: set[str] = set()
        while name in aliases and name not in seen:
            seen.add(name)
            name = aliases[name]
        return name

    graph = {
        name: [resolve(str(d)) for d in (getattr(desc, "dependencies", ()) or ())]
        for name, desc in services.items()
    }

    WHITE, GREY, BLACK = 0, 1, 2
    colour = dict.fromkeys(graph, WHITE)
    reported: set[frozenset[str]] = set()

    def walk(node: str, path: list[str]) -> Iterator[Violation]:
        colour[node] = GREY
        for nxt in graph.get(node, ()):
            if nxt not in colour:
                continue
            if colour[nxt] == GREY:
                cycle = path[path.index(nxt):] + [nxt] if nxt in path else [node, nxt]
                key = frozenset(cycle)
                if key not in reported:
                    reported.add(key)
                    yield Violation(
                        subject=" -> ".join(cycle),
                        message=(
                            "service construction dependencies form a cycle; "
                            "resolving any member deadlocks or recurses"
                        ),
                        remedy="break the cycle with a lazy accessor on one edge",
                    )
            elif colour[nxt] == WHITE:
                yield from walk(nxt, path + [nxt])
        colour[node] = BLACK

    for node in list(graph):
        if colour.get(node) == WHITE:
            yield from walk(node, [node])


@invariant(
    "container.no_self_dependency",
    scope="container",
    owner=_OWNER,
    description="no service declares itself as a dependency",
)
def _no_self_dependency() -> Iterator[Violation]:
    services, _aliases = _container_state()
    for name, descriptor in services.items():
        if name in {str(d) for d in (getattr(descriptor, "dependencies", ()) or ())}:
            yield Violation(
                subject=name,
                message=f"service {name!r} declares itself as a dependency",
                remedy="remove the self-reference",
            )


# ══════════════════════════════════════════════════════════════════════
# Locking
# ══════════════════════════════════════════════════════════════════════

@invariant(
    "locks.no_open_splats",
    scope="locks",
    owner=_OWNER,
    description="lockdep has reported no order violations",
)
def _no_open_splats() -> Iterator[Violation]:
    from core.runtime.lockdep import lockdep_report

    report = lockdep_report()
    for splat in report["splats"]:
        yield Violation(
            subject=splat["acquiring"],
            message=splat["message"],
            remedy="fix the ordering; a latent deadlock is still a deadlock",
        )


@invariant(
    "locks.declared_ranks_match_observed_order",
    scope="locks",
    owner=_OWNER,
    description="no observed acquisition edge contradicts a declared rank",
)
def _ranks_match_observed() -> Iterator[Violation]:
    from core.runtime.lockdep import LockRank, lockdep_report

    report = lockdep_report()
    ranks = {name: LockRank[value] for name, value in report["declared_ranks"].items()}
    for before, afters in report["order_edges"].items():
        before_rank = ranks.get(before)
        if before_rank is None or before_rank is LockRank.UNRANKED:
            continue
        for after in afters:
            after_rank = ranks.get(after)
            if after_rank is None or after_rank is LockRank.UNRANKED:
                continue
            if after_rank <= before_rank and after != before:
                yield Violation(
                    subject=f"{before} -> {after}",
                    message=(
                        f"{before!r} (rank {before_rank.name}) has been observed "
                        f"holding while {after!r} (rank {after_rank.name}) is taken, "
                        "which inverts the declared order"
                    ),
                    remedy="re-rank one of them, or reverse the acquisition",
                )


# ══════════════════════════════════════════════════════════════════════
# Memory policy
# ══════════════════════════════════════════════════════════════════════

@invariant(
    "oom.spine_is_immune",
    scope="memory",
    owner=_OWNER,
    description="load-bearing organs are never OOM shed candidates",
)
def _spine_is_immune() -> Iterator[Violation]:
    from core.runtime.foundations import IMMUNE_SERVICES
    from core.runtime.oom_policy import OOM_SCORE_ADJ_MIN, get_oom_policy

    table = {row["organ"]: row for row in get_oom_policy().scoring_table()}
    for name in IMMUNE_SERVICES:
        row = table.get(name)
        if row is None:
            continue  # not registered in this process; nothing to protect
        if row["oom_score_adj"] > OOM_SCORE_ADJ_MIN or row["sheddable"]:
            yield Violation(
                subject=name,
                message=(
                    f"{name!r} is load-bearing but is a shed candidate "
                    f"(oom_score_adj={row['oom_score_adj']}, sheddable={row['sheddable']}) "
                    "— memory pressure could take the runtime's spine"
                ),
                remedy=f"register {name!r} with oom_score_adj=OOM_SCORE_ADJ_MIN and no shed hook",
            )


@invariant(
    "oom.ladder_has_rungs",
    scope="memory",
    severity=Severity.WARNING,
    owner=_OWNER,
    description="at least one organ can actually be shed under pressure",
)
def _ladder_has_rungs() -> Iterator[Violation]:
    from core.runtime.oom_policy import get_oom_policy

    report = get_oom_policy().report()
    if report["registered_organs"] and report["sheddable_organs"] == 0:
        yield Violation(
            subject="oom_policy",
            message=(
                "no organ exposes a shed hook, so the OOM ladder has no rungs: "
                "the only available response to memory pressure is a restart"
            ),
            remedy="give at least one cache-holding organ a shed_memory() method",
        )


# ══════════════════════════════════════════════════════════════════════
# Pressure accounting
# ══════════════════════════════════════════════════════════════════════

@invariant(
    "psi.capacity_declared",
    scope="pressure",
    severity=Severity.WARNING,
    owner=_OWNER,
    description="every observed resource has a declared worker capacity",
)
def _psi_capacity_declared() -> Iterator[Violation]:
    from core.runtime.pressure_stall import psi_report

    for name, entry in psi_report().items():
        if entry["capacity"] == 1 and entry["peak_stalled"] > 1:
            yield Violation(
                subject=name,
                message=(
                    f"resource {name!r} has default capacity 1 but has had "
                    f"{entry['peak_stalled']} concurrent waiters, so `full` pressure "
                    "reads as saturated whenever anything waits at all"
                ),
                remedy=f"declare_capacity({name!r}, <real worker count>) at activation",
            )


# ══════════════════════════════════════════════════════════════════════
# Integrity
# ══════════════════════════════════════════════════════════════════════

@invariant(
    "integrity.untainted",
    scope="integrity",
    severity=Severity.WARNING,
    owner=_OWNER,
    description="no credibility-affecting taint is set",
)
def _untainted() -> Iterator[Violation]:
    from core.runtime.taint import taint_report

    report = taint_report()
    for entry in report["flags"]:
        if entry["flag"] in report["credibility_affecting"]:
            yield Violation(
                subject=entry["flag"],
                message=(
                    f"{entry['meaning']} ({entry['count']}×, first: "
                    f"{entry['first_reason']}) — any green verdict since then is "
                    "reported over a runtime that already broke an assumption"
                ),
                remedy="investigate the first occurrence; taint clears only on restart",
            )


# ══════════════════════════════════════════════════════════════════════
# Flags
# ══════════════════════════════════════════════════════════════════════

@invariant(
    "sanitizers.clean",
    scope="integrity",
    severity=Severity.WARNING,
    owner=_OWNER,
    description="no sanitizer has reported a finding",
)
def _sanitizers_clean() -> Iterator[Violation]:
    from core.runtime.sanitizers import sanitizer_report

    for finding in sanitizer_report()["findings"]:
        yield Violation(
            subject=f"{finding['sanitizer']}:{finding['context']}",
            message=f"{finding['message']} ({finding['occurrences']}×)",
            remedy="fix the lifetime, the non-finite source, or the affinity",
        )


@invariant(
    "flags.documented_and_owned",
    scope="flags",
    severity=Severity.WARNING,
    owner=_OWNER,
    description="every declared flag names an owner and says what it does",
)
def _flags_documented() -> Iterator[Violation]:
    from core.runtime.flags import declared_flags

    for name, spec in declared_flags().items():
        missing = [
            field
            for field in ("owner", "description")
            if not str(getattr(spec, field, "") or "").strip()
        ]
        if missing:
            yield Violation(
                subject=name,
                message=f"flag {name!r} is missing {' and '.join(missing)}",
                remedy="a knob nobody owns is a knob nobody can retire",
            )


# ══════════════════════════════════════════════════════════════════════
# Orchestration
# ══════════════════════════════════════════════════════════════════════

@invariant(
    "admission.validators_do_not_mutate",
    scope="orchestration",
    owner=_OWNER,
    description="no validating admission hook has been caught mutating its input",
)
def _validators_do_not_mutate() -> Iterator[Violation]:
    from core.runtime.sanitizers import sanitizer_report

    for finding in sanitizer_report()["findings"]:
        if finding["sanitizer"] == "admission":
            yield Violation(
                subject=finding["signature"].split(":", 1)[-1],
                message=finding["message"],
                remedy="move the edit into a mutating hook, which runs before validation",
            )


@invariant(
    "quota.guaranteed_specs_are_coherent",
    scope="orchestration",
    owner=_OWNER,
    description="a Guaranteed organ's requests equal its limits on every resource",
)
def _guaranteed_specs_coherent() -> Iterator[Violation]:
    from core.runtime.quota import QosClass, get_quota_registry

    for name, spec in get_quota_registry().specs().items():
        if spec.qos_class is not QosClass.GUARANTEED:
            continue
        for kind, limit in spec.limits.items():
            requested = spec.requests.get(kind)
            if requested is None or abs(requested - limit) > 1e-9:
                yield Violation(
                    subject=f"{name}.{kind}",
                    message=(
                        f"{name!r} is classed Guaranteed but requests "
                        f"{requested} against a limit of {limit}"
                    ),
                    remedy="set the request equal to the limit, or accept Burstable",
                )


@invariant(
    "eviction.guaranteed_organs_are_protected",
    scope="orchestration",
    owner=_OWNER,
    description="no Guaranteed organ appears in the eviction order",
)
def _guaranteed_protected() -> Iterator[Violation]:
    from core.runtime.eviction import eviction_report
    from core.runtime.quota import QosClass, get_quota_registry

    order = set(eviction_report()["eviction_order"])
    registry = get_quota_registry()
    for name in order:
        if registry.qos_class(name) is QosClass.GUARANTEED:
            yield Violation(
                subject=name,
                message=(
                    f"{name!r} is Guaranteed but is in the eviction order; the "
                    "guarantee it was given does not hold"
                ),
                remedy="exclude Guaranteed organs from eviction_order()",
            )


@invariant(
    "eviction.thresholds_are_ordered",
    scope="orchestration",
    owner=_OWNER,
    description="each signal's hard threshold is stricter than its soft one",
)
def _thresholds_ordered() -> Iterator[Violation]:
    from core.runtime.eviction import Comparison, eviction_report

    by_signal: dict[str, list[dict]] = {}
    for entry in eviction_report()["thresholds"]:
        by_signal.setdefault(entry["signal"], []).append(entry)
    for signal, entries in by_signal.items():
        hard = [e for e in entries if e["hard"]]
        soft = [e for e in entries if not e["hard"]]
        if not hard or not soft:
            continue
        for h in hard:
            for s in soft:
                if h["comparison"] != s["comparison"]:
                    continue
                stricter = (
                    h["value"] < s["value"]
                    if h["comparison"] == str(Comparison.BELOW)
                    else h["value"] > s["value"]
                )
                if not stricter:
                    yield Violation(
                        subject=signal,
                        message=(
                            f"hard threshold {h['value']} is not stricter than the "
                            f"soft threshold {s['value']} on {signal}; the hard one "
                            "fires first and the grace period never applies"
                        ),
                        remedy="make the hard threshold stricter, or drop the soft one",
                    )


@invariant(
    "reconcile.queues_are_draining",
    scope="orchestration",
    severity=Severity.WARNING,
    owner=_OWNER,
    description="no controller queue is deep and backing off at the same time",
)
def _queues_draining() -> Iterator[Violation]:
    from core.runtime.reconcile import reconcile_report

    for entry in reconcile_report()["controllers"]:
        queue = entry["queue"]
        if queue["depth"] > 32 and queue["backing_off"]:
            yield Violation(
                subject=entry["name"],
                message=(
                    f"controller {entry['name']!r} has {queue['depth']} queued keys "
                    f"while {len(queue['backing_off'])} are backing off; it is not "
                    "converging"
                ),
                remedy="look at last_error; a reconciler that always fails never drains",
            )


@invariant(
    "lease.no_live_duplicate_holder",
    scope="orchestration",
    owner=_OWNER,
    description="no lease we want is held by another live process on this host",
)
def _no_duplicate_holder() -> Iterator[Violation]:
    from core.runtime.lease import lease_report

    for entry in lease_report()["leases"]:
        record = entry.get("record")
        if not record or entry["is_leader"]:
            continue
        holder = record["identity"]
        ours = entry["identity"]
        if holder["host"] == ours["host"] and holder["pid"] != ours["pid"]:
            yield Violation(
                subject=entry["name"],
                message=(
                    f"lease {entry['name']!r} is held by pid {holder['pid']} on this "
                    f"host while pid {ours['pid']} also wants it — two runtimes are "
                    "contending for the same exclusive work"
                ),
                remedy="stop the other runtime; duplicate runtimes double memory",
            )


def register_runtime_invariants() -> int:
    """Import-time registration is the real work; this returns the count."""
    from core.verify.invariants import get_registry

    return len(get_registry().specs())


__all__ = ["register_runtime_invariants"]
