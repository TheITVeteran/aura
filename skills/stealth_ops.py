"""Legacy compatibility wrapper for the canonical core StealthOps skill."""

_module = __import__("core.skills.stealth_ops", fromlist=["*"])
globals().update({name: getattr(_module, name) for name in dir(_module) if not name.startswith("_")})
