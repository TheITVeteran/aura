"""Import every module that declares a phase contract, so the registry is whole.

`register_contract` runs at import time. That is the right place for it — the
declaration belongs beside the phase it describes — but it makes the registry
depend on who imported what. Two contracts written this session registered
only when something happened to pull their module in, and asking
`contract_coverage_report()` from a context that had not imported them
reported those phases as UNCONTRACTED. A coverage number that changes with
the caller's import graph is worse than no number: it reads as a measurement.

This module is the explicit answer. `ensure_contracts_loaded()` imports the
modules that declare contracts, and `contract_coverage_report()` calls it
first, so the count is the same from anywhere.

The list is checked, not trusted: `tests/test_cognitive_contracts.py` scans
the tree for `register_contract(` and fails if a module declaring a contract
is missing here — otherwise the next contract written would be invisible in
exactly the same way, and the coverage number would quietly understate.
"""

from __future__ import annotations

import importlib
import logging

logger = logging.getLogger(__name__)

#: Modules containing a `register_contract(...)` call. Kept explicit rather
#: than discovered by scanning at runtime: importing whatever looks like a
#: phase would drag optional subsystems into processes that do not want them.
CONTRACT_MODULES: tuple[str, ...] = (
    "core.phases.affect_update",
    "core.phases.cognitive_integration_phase",
    "core.phases.consciousness_phase",
    "core.phases.identity_reflection",
    "core.phases.initiative_generation",
    "core.phases.memory_consolidation",
    "core.phases.motivation_update",
    "core.phases.phi_consciousness",
    "core.phases.proprioceptive_loop",
    "core.phases.social_context_phase",
    "core.phases.unity_binding",
)

_LOADED = False


def ensure_contracts_loaded() -> tuple[str, ...]:
    """Import the contract-declaring modules. Returns those that failed.

    Failures are returned rather than raised: a coverage report that dies
    because one optional phase will not import is less useful than one that
    reports the shortfall, and the shortfall is exactly what a reader needs
    to know.
    """
    global _LOADED
    failed: list[str] = []
    for module_name in CONTRACT_MODULES:
        try:
            importlib.import_module(module_name)
        except Exception as exc:  # noqa: BLE001 — an unimportable phase is data
            logger.warning("contract module %s did not import: %s", module_name, exc)
            failed.append(module_name)
    _LOADED = True
    return tuple(failed)


__all__ = ["CONTRACT_MODULES", "ensure_contracts_loaded"]
