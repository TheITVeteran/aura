from __future__ import annotations

from pathlib import Path

from tools.closeout.audit_resource_observation_ownership import run_audit


def _write(path: Path, source: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source, encoding="utf-8")


def test_resource_observation_audit_accepts_canonical_observer_usage(tmp_path):
    _write(
        tmp_path / "service.py",
        "from core.runtime.resource_observation import get_resource_observer\n"
        "snapshot = get_resource_observer().snapshot()\n",
    )

    report = run_audit(root=tmp_path)

    assert report["passed"] is True
    assert report["finding_count"] == 0
    assert report["parse_errors"] == []


def test_resource_observation_audit_rejects_direct_host_probes(tmp_path):
    _write(
        tmp_path / "service.py",
        "import os\nimport psutil\nimport resource\nimport shutil\n"
        "memory = psutil.virtual_memory()\n"
        "processes = list(psutil.process_iter())\n"
        "reader = psutil.process_iter\n"
        "rss = psutil.Process(123).memory_info().rss\n"
        "cpu = psutil.cpu_percent()\n"
        "battery = psutil.sensors_battery()\n"
        "disk = shutil.disk_usage('/')\n"
        "disk_reader = shutil.disk_usage\n"
        "load = os.getloadavg()\n"
        "cores = os.cpu_count()\n"
        "pages = os.sysconf('SC_PHYS_PAGES')\n"
        "usage = resource.getrusage(resource.RUSAGE_SELF)\n"
        "footprint = libproc.proc_pid_rusage(123, 4, None)\n",
    )

    report = run_audit(root=tmp_path)
    codes = {finding["code"] for finding in report["findings"]}

    assert report["passed"] is False
    assert report["finding_count"] == 13
    assert codes == {
        "direct_disk_observation",
        "direct_disk_observation_reference",
        "direct_process_observation",
        "direct_platform_resource_observation",
        "direct_psutil_resource_reference",
        "direct_psutil_resource_observation",
        "direct_standard_resource_observation",
    }


def test_resource_observation_audit_fails_on_unparseable_production_source(tmp_path):
    _write(tmp_path / "broken.py", "def broken(:\n")

    report = run_audit(root=tmp_path)

    assert report["passed"] is False
    assert report["finding_count"] == 0
    assert report["parse_errors"][0]["path"] == "broken.py"
