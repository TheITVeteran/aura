"""Runtime markers for intentional evaluation ablations.

Production health monitors should fail closed on missing critical services.
Evaluation harnesses deliberately lesion services to prove architectural
dependencies. This module lets those harnesses mark intentional lesions so
watchdogs do not silently repair them and invalidate the ablation.
"""

from __future__ import annotations

import contextlib
import os
import re
from collections.abc import Iterator

_ABLATION_SERVICES_ENV = "AURA_ACTIVE_ABLATION_SERVICES"

_ABLATION_ALIASES: dict[str, set[str]] = {
    "unified_will": {
        "authority",
        "authority_gateway",
        "authoritygate",
        "authoritygateway",
        "executive_authority",
        "unified_will",
        "unified_will_authority",
        "unifiedwill",
        "will",
        "will_authority",
    },
    "affective_steering_engine": {
        "affect_steering",
        "affective_steering",
        "affective_steering_engine",
        "steering_engine",
    },
    "affect_engine": {
        "affect_engine",
        "affect_engine_v2",
        "affect_facade",
        "affectengine",
    },
    "native_system2": {
        "native_system2",
        "proof_answer_solver",
        "structured_proof_solver",
        "system_2",
        "system2",
        "system2_search",
        "system2_symbolic_reasoner",
    },
}


def _normalize_service_name(name: object) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(name or "").strip().lower()).strip("_")


def _expanded_service_names(name: object) -> set[str]:
    normalized = _normalize_service_name(name)
    if not normalized:
        return set()
    expanded = {normalized}
    for canonical, aliases in _ABLATION_ALIASES.items():
        if normalized == canonical or normalized in aliases:
            expanded.add(canonical)
            expanded.update(aliases)
    return expanded


def active_ablation_services() -> set[str]:
    raw = os.environ.get(_ABLATION_SERVICES_ENV, "")
    active: set[str] = set()
    for part in raw.split(","):
        active.update(_expanded_service_names(part))
    return active


def service_intentionally_lesioned(name: object) -> bool:
    return bool(_expanded_service_names(name) & active_ablation_services())


@contextlib.contextmanager
def mark_services_lesioned(names: list[str]) -> Iterator[None]:
    previous = os.environ.get(_ABLATION_SERVICES_ENV)
    merged = active_ablation_services() | {
        normalized
        for name in names
        if (normalized := _normalize_service_name(name))
    }
    if merged:
        os.environ[_ABLATION_SERVICES_ENV] = ",".join(sorted(merged))
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(_ABLATION_SERVICES_ENV, None)
        else:
            os.environ[_ABLATION_SERVICES_ENV] = previous
