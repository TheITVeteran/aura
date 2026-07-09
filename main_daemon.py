"""main_daemon.py
────────────────
Standard entry point for the Aura Cognitive Daemon.
"""

import asyncio
import logging
import sys
import tempfile
from pathlib import Path

# Add core to path
sys.path.append(str(Path(__file__).parent))

from core.ops.daemon import main

_DAEMON_RECOVERABLE_ERRORS = (
    ImportError,
    AttributeError,
    RuntimeError,
    TimeoutError,
    OSError,
    ConnectionError,
    LookupError,
    TypeError,
    ValueError,
    PermissionError,
)

if __name__ == "__main__":
    # Setup basic logging for the daemon process itself
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(Path(tempfile.gettempdir()) / "aura_daemon.log", mode='a')
        ]
    )

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
    except _DAEMON_RECOVERABLE_ERRORS as e:
        logging.critical("Daemon crashed: %s", e)
        sys.exit(1)
