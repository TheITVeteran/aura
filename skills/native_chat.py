"""Legacy compatibility wrapper for the canonical core native-chat skill."""
_module = __import__("core.skills.native_chat", fromlist=["*"])
globals().update({name: getattr(_module, name) for name in dir(_module) if not name.startswith("_")})
