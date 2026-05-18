"""Legacy compatibility wrapper for the canonical core sandbox skill."""
_module = __import__("core.skills.internal_sandbox", fromlist=["*"])
globals().update({name: getattr(_module, name) for name in dir(_module) if not name.startswith("_")})
