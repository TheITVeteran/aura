"""Legacy compatibility wrapper for the canonical core personality skill."""
_module = __import__("core.skills.personality_skill", fromlist=["*"])
globals().update({name: getattr(_module, name) for name in dir(_module) if not name.startswith("_")})
