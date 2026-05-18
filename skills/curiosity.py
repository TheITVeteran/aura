"""Legacy compatibility wrapper for the canonical core curiosity skill."""

from importlib import import_module as _import_module

_module = _import_module("core.skills.curiosity")
__all__ = getattr(_module, "__all__", [name for name in dir(_module) if not name.startswith("_")])
globals().update({name: getattr(_module, name) for name in __all__})
