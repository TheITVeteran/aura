import importlib
import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger("aura.safe_import")


@dataclass(frozen=True)
class MissingOptionalDependency:
    name: str
    error: str
    __missing__: bool = True

    @property
    def __name__(self) -> str:
        return self.name

    def __bool__(self) -> bool:
        return False

    def __getattr__(self, attribute: str) -> Any:
        message = (
            f"Optional dependency '{self.name}' is unavailable; attempted to access "
            f"attribute '{attribute}'. Original import error: {self.error}"
        )
        logger.error("missing optional dependency used as module: %s", message)
        raise ModuleNotFoundError(message)


def _missing_requested_module(requested: str, missing: str | None) -> bool:
    if not missing:
        return False
    return missing == requested or requested.startswith(f"{missing}.")


def safe_import(name: str, optional: bool = False) -> Any:
    """
    Try to import `name`. Returns module object if found,
    otherwise returns a fail-closed missing-dependency sentinel for absent
    optional modules.
    """
    try:
        mod = importlib.import_module(name)
        return mod
    except ModuleNotFoundError as e:
        if optional and _missing_requested_module(name, e.name):
            logger.warning("safe_import: optional module '%s' is unavailable: %s", name, e)
            return MissingOptionalDependency(name=name, error=str(e))
        logger.warning("safe_import: import failed for '%s': %s", name, e)
        if optional:
            raise
        raise ImportError(f"Critical dependency '{name}' is missing and not optional.") from e
    except ImportError as e:
        logger.warning("safe_import: import failed for '%s': %s", name, e)
        if optional:
            raise
        raise ImportError(f"Critical dependency '{name}' is missing and not optional.") from e

async def async_safe_import(name: str, optional: bool = False) -> Any:
    """Async wrapper for safe_import to prevent event loop blocking."""
    import asyncio
    # Use run_in_executor to avoid blocking the event loop during heavy imports
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, safe_import, name, optional)
def is_missing(module: Any) -> bool:
    """Check if a module returned by safe_import is actually missing."""
    return isinstance(module, MissingOptionalDependency) or bool(getattr(module, "__missing__", False))
