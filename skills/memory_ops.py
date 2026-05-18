"""Legacy compatibility wrapper for the canonical core memory-ops skill."""
_module = __import__("core.skills.memory_ops", fromlist=["*"])
globals().update({name: getattr(_module, name) for name in dir(_module) if not name.startswith("_")})
