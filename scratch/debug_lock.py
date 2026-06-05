
import os
import time

import pexpect

nethack_path = "/opt/homebrew/bin/nethack"
env = os.environ.copy()
env["TERM"] = "xterm-256color"

child = pexpect.spawn(f"{nethack_path} -u Aura -D", env=env, encoding='utf-8')
time.sleep(1)
try:
    out = child.read_nonblocking(size=1000, timeout=1.0)
    print(f"Output: {repr(out)}")
except (OSError, RuntimeError, pexpect.EOF, pexpect.TIMEOUT):
    print("Timeout/EOF.")
child.terminate()
