"""infrastructure/security.py - Operational Security for Aura.

C-08 FIX: Removed process name masking (setproctitle disguise as syslogd).
Process masking is anti-security — it defeats SIEM, process auditing, and
incident response. SOC teams will flag it as rootkit behavior.
"""
import logging

logger = logging.getLogger("Infra.Security")


class OpSec:
    """Operational Security — legitimate operational safeguards only."""

    @staticmethod
    async def random_sleep(base_seconds: float = 1.0, variance: float = 0.5):
        """Sleep with jitter to avoid thundering-herd on retries (Async)."""
        import asyncio
        import random

        jitter = random.uniform(-variance, variance)
        sleep_time = max(0.1, base_seconds + jitter)
        await asyncio.sleep(sleep_time)

    @staticmethod
    async def safe_delay(seconds: float):
        """Alias for random_sleep with zero variance. Awaits, never blocks."""
        import asyncio
        await asyncio.sleep(seconds)

    @staticmethod
    def mask_process_name(name: str = "aura-autonomy-engine"):
        """Set a truthful, identifiable process name for monitoring.

        Previous implementation disguised the process as /usr/sbin/syslogd.
        This is now replaced with honest process identification.
        """
        try:
            import setproctitle

            setproctitle.setproctitle(name)
            logger.debug("Process name set to: %s", name)
        except ImportError:
            pass  # setproctitle is optional
