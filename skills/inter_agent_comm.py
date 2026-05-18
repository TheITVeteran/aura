"""Legacy compatibility wrapper for the canonical core inter-agent skill."""
_module = __import__("core.skills.inter_agent_comm", fromlist=["*"])
globals().update({name: getattr(_module, name) for name in dir(_module) if not name.startswith("_")})
