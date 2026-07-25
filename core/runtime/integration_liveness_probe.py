"""Isolated child entrypoint for optional-integration import probes.

This module may import accelerator libraries, but it never loads model
weights. Keeping that contract in a dedicated entrypoint lets model-lane
admission distinguish a bounded import probe from a model-owning process.
"""
from __future__ import annotations

import importlib
import json
import sys
from collections.abc import Sequence

NO_PREFLIGHT = "__aura_no_preflight__"
_PROBE_FAILURES = (
    AssertionError,
    AttributeError,
    ImportError,
    KeyError,
    NameError,
    OSError,
    RuntimeError,
    SyntaxError,
    TypeError,
    ValueError,
)


def _emit(state: str, detail: str = "") -> None:
    print(json.dumps({"state": state, "detail": detail[:400]}))


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) < 2:
        _emit("broken", "probe contract requires module and preflight arguments")
        return 2

    module = args[0]
    preflight = "" if args[1] == NO_PREFLIGHT else args[1]
    required = [attribute for attribute in args[2:] if attribute]

    if preflight:
        try:
            preflight_module, separator, preflight_attribute = preflight.partition(":")
            if not separator or not preflight_module or not preflight_attribute:
                raise ValueError("preflight must use module:function syntax")
            getattr(
                importlib.import_module(preflight_module),
                preflight_attribute,
            )()
        except _PROBE_FAILURES as exc:
            _emit(
                "broken",
                f"preflight {preflight} failed: {type(exc).__name__}: {exc}",
            )
            return 0

    top_level = module.split(".")[0]
    try:
        imported = importlib.import_module(module)
    except _PROBE_FAILURES as exc:
        absent = (
            isinstance(exc, ModuleNotFoundError)
            and getattr(exc, "name", None) == top_level
        )
        _emit(
            "absent" if absent else "broken",
            "not installed" if absent else f"{type(exc).__name__}: {exc}",
        )
        return 0

    missing = [attribute for attribute in required if not hasattr(imported, attribute)]
    if missing:
        _emit("broken", "imported but missing attributes: " + ", ".join(missing))
        return 0

    _emit("live")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
