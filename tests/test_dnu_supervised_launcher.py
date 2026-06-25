"""#50 phase 2 application: the supervised DNU launcher restarts a wedged battery.

Uses a tiny battery process (writes the liveness beacon then exits cleanly or
hangs) to verify the launcher composes the wedge-recovery supervisor correctly:
a clean battery run is NOT restarted; a wedged (frozen-beacon) battery is
detected and restarted out-of-process. (The real battery is the cert-critical
file and is left untouched; this proves the wiring without a heavy GPU run.)
"""

import sys

import pytest

from core.runtime.subprocess_gateway import get_subprocess_gateway
import tools.agi.run_dnu_supervised as launcher

_BATTERY_HELPER = (
    "import json,os,sys,time\n"
    "beacon,mode=sys.argv[1],sys.argv[2]\n"
    "start=time.time()\n"
    "while time.time()-start<1.2:\n"
    "    now=time.time()\n"
    "    open(beacon,'w').write(json.dumps({'pid':os.getpid(),'last_loop_run':now,'written_at':now,'loop_state':'alive'}))\n"
    "    time.sleep(0.2)\n"
    "if mode=='hang':\n"
    "    time.sleep(10000)\n"
    "sys.exit(0)\n"
)


@pytest.fixture
def battery_helper(tmp_path):
    p = tmp_path / "_battery_helper.py"
    p.write_text(_BATTERY_HELPER, encoding="utf-8")
    return p


def test_clean_battery_run_is_not_restarted(tmp_path, battery_helper):
    beacon = tmp_path / "hb.json"
    rc = launcher.main([
        "--heartbeat", str(beacon),
        "--battery-cmd", f"{sys.executable} {battery_helper} {beacon} exit",
        "--grace", "0.4", "--interval", "0.3", "--stale-ceiling", "2.0", "--max-restarts", "2",
    ])
    assert rc == 0  # completed cleanly, no wedge


def test_wedged_battery_is_detected_and_restarted(tmp_path, battery_helper, capsys):
    beacon = tmp_path / "hb.json"
    # The helper hangs after beating → the launcher's supervisor must detect the
    # wedge and restart it; with a 1-restart budget the run ends restart-exceeded.
    rc = launcher.main([
        "--heartbeat", str(beacon),
        "--battery-cmd", f"{sys.executable} {battery_helper} {beacon} hang",
        "--grace", "0.8", "--interval", "0.4", "--stale-ceiling", "1.5", "--max-restarts", "1",
    ])
    out = capsys.readouterr().out
    assert "supervised restart 1" in out, "a frozen-beacon battery must trigger a supervised restart"
    assert rc != 0  # restart budget exhausted (the helper never recovers)
    # Reap any leftover helper processes.
    get_subprocess_gateway().run(
        ["pkill", "-f", str(battery_helper)],
        timeout=5.0,
        capture_output=True,
        check=False,
        offline_tooling=True,
        source="maintenance_tooling:dnu_supervised_test_cleanup",
    )
