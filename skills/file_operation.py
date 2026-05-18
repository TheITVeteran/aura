"""Legacy compatibility wrapper for the canonical core file-operation skill."""
_module = __import__("core.skills.file_operation", fromlist=["*"])
globals().update({name: getattr(_module, name) for name in dir(_module) if not name.startswith("_")})
