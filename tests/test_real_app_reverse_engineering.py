"""The DNA engine reverse-engineers REAL host binaries, proved by differential
against their held-out outputs. Pins the pipeline: a correct clean-room reimpl
is `supported`, a wrong one is `refuted`, verified against the actual binary.

Uses base64 and rev, which exist on both macOS and Linux, so no skip is needed."""
from __future__ import annotations

import shutil

import pytest

from tools.proof.run_real_app_reverse_engineering_proof import _TARGETS, run_self_test


@pytest.mark.parametrize("target_name", ["base64", "rev"])
def test_real_binary_reverse_engineering_pipeline_is_real(target_name):
    target = _TARGETS[target_name]()
    assert shutil.which(target.binary), f"{target.binary} must exist on this host"
    report = run_self_test(target)
    # correct clean-room reimplementation reproduces the REAL binary's held-out I/O
    assert report["correct_reimpl"]["status"] == "supported", report["correct_reimpl"]
    # a deliberately-wrong reimplementation is caught (the harness can fail)
    assert report["broken_reimpl"]["status"] == "refuted", report["broken_reimpl"]
    assert report["pipeline_proven"] is True
    assert report["held_out_cases"] >= 3
