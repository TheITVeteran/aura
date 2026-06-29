"""Safety checks for CRSM LoRA train/fuse execution."""
from __future__ import annotations

from types import SimpleNamespace

from training import resume_training, run_unattended, train_and_fuse

GIB = 1024**3


def _patch_resources(monkeypatch, *, available_gb=40.0, percent=50.0, free_disk_gb=200.0):
    monkeypatch.setattr(
        train_and_fuse.psutil,
        "virtual_memory",
        lambda: SimpleNamespace(available=int(available_gb * GIB), percent=percent),
    )
    monkeypatch.setattr(
        train_and_fuse.shutil,
        "disk_usage",
        lambda _path: SimpleNamespace(free=int(free_disk_gb * GIB)),
    )


def test_training_preflight_passes_with_headroom(monkeypatch, tmp_path):
    _patch_resources(monkeypatch)
    monkeypatch.setattr(train_and_fuse, "_live_aura_processes", lambda: [])

    report = train_and_fuse.training_preflight(base_model=tmp_path / "Qwen2.5-32B-Instruct-4bit", skip_train=False)

    assert report["passed"] is True
    assert report["mode"] == "train_fuse_publish"
    assert report["requirements"]["min_available_gb"] == 28.0


def test_training_preflight_blocks_low_memory(monkeypatch, tmp_path):
    _patch_resources(monkeypatch, available_gb=9.0, percent=91.0)
    monkeypatch.setattr(train_and_fuse, "_live_aura_processes", lambda: [])

    report = train_and_fuse.training_preflight(base_model=tmp_path / "Qwen2.5-32B-Instruct-4bit", skip_train=False)

    assert report["passed"] is False
    assert any("available_memory" in blocker for blocker in report["blockers"])
    assert any("memory_pressure" in blocker for blocker in report["blockers"])


def test_training_preflight_blocks_live_aura_unless_explicitly_allowed(monkeypatch, tmp_path):
    _patch_resources(monkeypatch)
    monkeypatch.setattr(train_and_fuse, "_live_aura_processes", lambda: [{"pid": 123, "cmdline": "aura_main.py"}])

    blocked = train_and_fuse.training_preflight(base_model=tmp_path / "Qwen2.5-32B-Instruct-4bit", skip_train=False)
    assert blocked["passed"] is False
    assert "live_aura_processes:1" in blocked["blockers"]

    monkeypatch.setenv("AURA_TRAINING_ALLOW_LIVE_AURA", "1")
    allowed = train_and_fuse.training_preflight(base_model=tmp_path / "Qwen2.5-32B-Instruct-4bit", skip_train=False)
    assert allowed["passed"] is True


def test_run_unattended_accepts_resume_and_preflight_only_flags():
    args = run_unattended.parse_args(["--resume", "--preflight-only", "--tag", "crsm-closeout"])

    assert args.resume is True
    assert args.preflight_only is True
    assert args.tag == "crsm-closeout"


def test_run_unattended_resume_fuse_publish_marks_crsm_consumed(monkeypatch):
    commands = []
    monkeypatch.setattr(run_unattended, "update_state", lambda **_kwargs: {})
    monkeypatch.setattr(run_unattended, "_spawn", lambda cmd, *, started_at: commands.append(cmd) or 0)
    args = run_unattended.parse_args(["--tag", "crsm-closeout"])

    rc = run_unattended.run_fuse_publish(args, started_at="now")

    assert rc == 0
    assert commands
    assert "--mark-crsm-consumed" in commands[0]


def test_resume_training_parses_zenith_resume_log(monkeypatch, tmp_path):
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    base = adapter_dir / "0066000_adapters.safetensors"
    later = adapter_dir / "0066500_adapters.safetensors"
    base.write_text("base", encoding="utf-8")
    later.write_text("later", encoding="utf-8")
    log_path = tmp_path / "train.log"
    log_path.write_text(
        "\n--- Resume Zenith from 0066000_adapters.safetensors, "
        "24153 iters remaining, target_total=90153, seq=4096 ---\n"
        "Iter 500: Saved adapter weights to "
        f"{later}\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(resume_training, "ADAPTER_PATH", adapter_dir)
    monkeypatch.setattr(resume_training, "LOG_PATH", log_path)

    checkpoint, remaining = resume_training._resume_state_from_log()

    assert checkpoint == later
    assert remaining == 23653


def test_run_unattended_memory_guard_blocks_process_tree_rss(monkeypatch):
    monkeypatch.setenv("AURA_TRAINING_MAX_PROCESS_TREE_RSS_GB", "12")
    monkeypatch.setattr(run_unattended, "_process_tree_rss_gb", lambda _pid: 12.5)

    reason = run_unattended._memory_guard_reason(123)

    assert reason == "process_tree_rss:12.5GB/12.0GB"


def test_run_unattended_memory_guard_blocks_host_pressure(monkeypatch):
    monkeypatch.setenv("AURA_TRAINING_MAX_PROCESS_TREE_RSS_GB", "80")
    monkeypatch.setenv("AURA_TRAINING_MAX_HOST_MEMORY_PERCENT", "90")
    monkeypatch.setattr(run_unattended, "_process_tree_rss_gb", lambda _pid: 10.0)
    monkeypatch.setattr(run_unattended.psutil, "virtual_memory", lambda: SimpleNamespace(percent=93.0))

    reason = run_unattended._memory_guard_reason(123)

    assert reason == "host_memory_pressure:93.0%/90.0%"
