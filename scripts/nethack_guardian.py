import logging
import os
import signal
import sys
import threading
import time
from pathlib import Path

# Configuration
PROJECT_ROOT = Path(os.environ.get("AURA_PROJECT_ROOT", Path(__file__).resolve().parents[1])).expanduser().resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.runtime.subprocess_gateway import get_subprocess_gateway

LOG_FILE = Path(os.environ.get("AURA_NETHACK_LOG", PROJECT_ROOT / "simulate_out_v7.txt")).expanduser()
CHECK_INTERVAL = 60  # Check every minute
STALL_TIMEOUT = 1800 # 30 minutes
AURA_MAIN = "aura_main.py"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("nethack_guardian.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("NetHackGuardian")

def get_aura_pids():
    result = get_subprocess_gateway().run(
        ["pgrep", "-f", AURA_MAIN],
        cwd=PROJECT_ROOT,
        timeout=10,
        read_only=True,
        source="nethack_guardian:pgrep",
    )
    if result.returncode == 1:
        return []
    if result.returncode != 0:
        logger.warning("pgrep failed while checking Aura process state: %s", result.stderr.strip())
        return []
    pids = []
    for token in result.stdout.split():
        try:
            pids.append(int(token))
        except ValueError:
            logger.warning("Ignoring non-integer pgrep token: %s", token)
    return pids

def kill_aura():
    pids = get_aura_pids()
    if not pids:
        logger.info("No Aura processes found to kill.")
        return
    
    logger.warning(f"Stall detected! Killing Aura processes: {pids}")
    for pid in pids:
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            logger.debug("Aura process %s exited before kill.", pid)
        except PermissionError:
            logger.error("Permission denied while killing Aura process %s.", pid)
    logger.info("Aura processes killed. Watchdog should restart them shortly.")

def monitor(max_cycles: int | None = None, stop_event: threading.Event | None = None):
    logger.info(f"Starting NetHack Guardian. Monitoring {LOG_FILE} for stalls...")
    last_mtime = 0
    if LOG_FILE.exists():
        last_mtime = LOG_FILE.stat().st_mtime
    
    last_change_time = time.time()
    cycles = 0

    while stop_event is None or not stop_event.is_set():
        if max_cycles is not None and cycles >= max_cycles:
            logger.info("Stopping NetHack Guardian after %s bounded cycle(s).", max_cycles)
            break
        try:
            if LOG_FILE.exists():
                current_mtime = LOG_FILE.stat().st_mtime
                if current_mtime > last_mtime:
                    logger.debug("Log file changed.")
                    last_mtime = current_mtime
                    last_change_time = time.time()
                else:
                    idle_time = time.time() - last_change_time
                    if idle_time > STALL_TIMEOUT:
                        logger.error(f"STALL DETECTED: Log file has not changed for {idle_time/60:.1f} minutes.")
                        kill_aura()
                        # Reset timer to avoid immediate re-kill
                        last_change_time = time.time()
            else:
                logger.warning(f"Log file {LOG_FILE} not found. Waiting...")
            
        except OSError as exc:
            logger.error("Guardian filesystem error: %s", exc)

        cycles += 1
        if max_cycles is not None and cycles >= max_cycles:
            logger.info("Stopping NetHack Guardian after %s bounded cycle(s).", max_cycles)
            break
        if stop_event is not None:
            stop_event.wait(CHECK_INTERVAL)
        else:
            time.sleep(CHECK_INTERVAL)

if __name__ == "__main__":
    monitor()
