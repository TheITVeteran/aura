"""Legacy compatibility wrapper for the canonical core environment skill."""
_module = __import__("core.skills.environment_info", fromlist=["*"])
globals().update({name: getattr(_module, name) for name in dir(_module) if not name.startswith("_")})
