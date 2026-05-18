"""Legacy compatibility wrapper for the canonical core social-lurker skill."""
_module = __import__("core.skills.social_lurker", fromlist=["*"])
globals().update({name: getattr(_module, name) for name in dir(_module) if not name.startswith("_")})
