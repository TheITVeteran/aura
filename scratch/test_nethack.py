import sys
import time
from pathlib import Path

# Add current dir to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.adapters.nethack_adapter import NetHackAdapter  # noqa: E402

adapter = NetHackAdapter()
print("Starting adapter...")
adapter.start()
print("Adapter started.")
time.sleep(2)
screen = adapter.get_screen_text()
print("Screen Captured:")
print(screen)
print(f"Is alive: {adapter.is_alive()}")
adapter.close()
