"""Legacy compatibility wrapper for the canonical core self-improvement skill."""
_module = __import__("core.skills.self_improvement", fromlist=["*"])
globals().update({name: getattr(_module, name) for name in dir(_module) if not name.startswith("_")})
