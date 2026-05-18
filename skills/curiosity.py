"""Legacy compatibility wrapper for the canonical core curiosity skill."""
_module = __import__("core.skills.curiosity", fromlist=["*"])
globals().update({name: getattr(_module, name) for name in dir(_module) if not name.startswith("_")})
