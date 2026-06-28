from __future__ import annotations

import json
from pathlib import Path

from tools.proof.validate_contrastive_kv_live_proof import validate


def _write_proof(root: Path, *, stdout: str, peak_rss_mb: float = 20_000.0, passed: bool = True) -> None:
    verdict = {
        "passed": passed,
        "peak_rss_mb": peak_rss_mb,
        "git_commit": "abc123",
        "git_dirty": False,
        "steps": [
            {"step": "boot_health", "ok": True},
            {"step": "chat_identity", "ok": True},
            {"step": "chat_continuity", "ok": True},
            {"step": "chat_conversation_soak", "ok": True},
            {"step": "desktop_action", "ok": True},
            {"step": "shutdown", "ok": True},
            {"step": "runtime_stream_scan", "ok": True},
        ],
    }
    (root / "LATEST_VERDICT.json").write_text(json.dumps(verdict), encoding="utf-8")
    (root / "run_stdout.log").write_text(stdout, encoding="utf-8")


def test_contrastive_kv_validator_requires_actual_cache_marker(tmp_path: Path) -> None:
    _write_proof(
        tmp_path,
        stdout="\n".join(
            [
                "contrastive decoding active (alpha=0.50 beta=0.10)",
                "Reasoning processors ACTIVE (1: steer=False cd=True).",
            ]
        ),
    )

    ok, report = validate(tmp_path, max_peak_rss_mb=32_768)

    assert ok is False
    assert any("amateur KV cache active" in finding for finding in report["findings"])


def test_contrastive_kv_validator_accepts_clean_live_artifact(tmp_path: Path) -> None:
    _write_proof(
        tmp_path,
        stdout="\n".join(
            [
                "amateur KV cache active for /models/Qwen2.5-1.5B (max_tokens=4096)",
                "contrastive decoding active (alpha=0.50 beta=0.10)",
                "Reasoning processors ACTIVE (1: steer=False cd=True).",
                "LIVE PROOF PASSED",
            ]
        ),
    )

    ok, report = validate(tmp_path, max_peak_rss_mb=32_768)

    assert ok is True
    assert report["findings"] == []


def test_contrastive_kv_validator_rejects_rss_over_ceiling(tmp_path: Path) -> None:
    _write_proof(
        tmp_path,
        peak_rss_mb=40_000.0,
        stdout="\n".join(
            [
                "amateur KV cache active for /models/Qwen2.5-1.5B (max_tokens=4096)",
                "contrastive decoding active (alpha=0.50 beta=0.10)",
                "Reasoning processors ACTIVE (1: steer=False cd=True).",
            ]
        ),
    )

    ok, report = validate(tmp_path, max_peak_rss_mb=32_768)

    assert ok is False
    assert any("exceeded ceiling" in finding for finding in report["findings"])
