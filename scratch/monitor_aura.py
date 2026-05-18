import os
import re
import time
from pathlib import Path

AURA_HOME = Path(os.environ.get("AURA_HOME", Path.home() / ".aura")).expanduser()
PROJECT_ROOT = Path(os.environ.get("AURA_SOURCE_DIR", Path(__file__).resolve().parents[1])).expanduser().resolve()
LOG_PATH = AURA_HOME / "logs" / "desktop-launch.log"
REPORT_PATH = PROJECT_ROOT / "scratch" / "guardian_report.log"

PATTERNS = {
    "DEADLOCK": r"DEADLOCK|Deadlock",
    "ATTRIBUTE_ERROR": r"AttributeError",
    "SERVICE_ERROR": r"ServiceNotFoundError|Service '.*' not found",
    "STALL": r"Event loop blocked for (\d+\.\d+)s",
    "REPETITION": r"Token repetition detected|generative loop",
    "MOTOR_FAILURE": r"Motor cortex watchdog spuriously cancelled",
    "HEALTH_PULSE": r"UNIFIED HEALTH PULSE",
}

def monitor():
    print(f"Aura Guardian started. Monitoring {LOG_PATH}")
    last_pos = LOG_PATH.stat().st_size if LOG_PATH.exists() else 0
    
    while True:
        if not LOG_PATH.exists():
            time.sleep(10)
            continue
            
        current_size = LOG_PATH.stat().st_size
        if current_size < last_pos: # Log rotated
            last_pos = 0
            
        if current_size > last_pos:
            with open(LOG_PATH) as f:
                f.seek(last_pos)
                new_content = f.read()
                last_pos = f.tell()
                
                findings = []
                for key, pattern in PATTERNS.items():
                    matches = re.findall(pattern, new_content)
                    if matches:
                        findings.append(f"[{key}] Found {len(matches)} occurrences.")
                
                if findings:
                    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
                    with open(REPORT_PATH, "a") as report:
                        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                        report.write(f"\n--- {timestamp} ---\n")
                        report.write("\n".join(findings) + "\n")
        
        time.sleep(300) # Check every 5 minutes

if __name__ == "__main__":
    monitor()
