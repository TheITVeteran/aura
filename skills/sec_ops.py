"""Legacy compatibility wrapper for the canonical core SecOps skill."""

_module = __import__("core.skills.sec_ops", fromlist=["*"])
globals().update({name: getattr(_module, name) for name in dir(_module) if not name.startswith("_")})
