"""Online LoRA governor: an exclusion check that could not see the trainers."""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from core.adaptation.online_lora_governor import OnlineLoRAGovernor

pytestmark = pytest.mark.unit


class _Table:
    available = True
    error = ""

    def __init__(self, cmdlines):
        self.processes = [
            SimpleNamespace(pid=1000 + i, cmdline=c) for i, c in enumerate(cmdlines)
        ]


class _Observer:
    provenance = SimpleNamespace(
        source=SimpleNamespace(value="test"), scenario_id="s1"
    )

    def __init__(self, cmdlines):
        self._cmdlines = cmdlines

    def process_table(self):
        return _Table(self._cmdlines)


def _governor(cmdlines, tmp_path):
    gov = OnlineLoRAGovernor(observer=_Observer(cmdlines))
    gov.receipt_path = tmp_path / "receipts.jsonl"
    return gov


# ── the exclusion check must see the trainers that actually exist ──────────


@pytest.mark.parametrize("cmdline,label", [
    (["python", "-m", "mlx_lm", "lora", "--train"], "mlx_lm.lora"),
    (["python", "-m", "mlx_lm.fuse", "--model", "x"], "mlx_lm.fuse"),
    (["python", "core/adaptation/self_optimizer.py"], "self_optimizer"),
    (["python", "train_resident_recurrence.py", "--train"], "resident_trainer"),
    (["python", "-m", "grpo", "run"], "grpo_trainer"),
    (["torchrun", "--nproc", "2", "train.py"], "torch_run"),
    (["accelerate", "launch", "finetune.py"], "accelerate_launch"),
])
def test_training_processes_are_detected(cmdline, label, tmp_path):
    """Matching only 'mlx_lm ... lora' made the recurrence-native 32B trainer
    and every other training entrypoint INVISIBLE, so this governor would start
    a competing update beside a running trainer while reporting it had checked."""
    found = _governor([cmdline], tmp_path).active_lora_processes()

    assert found, f"{label} must be visible to the exclusion check"
    assert found[0]["matched"] == label


def test_unrelated_processes_are_not_flagged(tmp_path):
    found = _governor([
        ["python", "-m", "http.server"],
        ["bash", "-lc", "ls"],
    ], tmp_path).active_lora_processes()

    assert found == []


def test_the_governor_does_not_detect_itself(tmp_path):
    """Self-detection would deadlock the check against its own process."""
    import os

    gov = OnlineLoRAGovernor(observer=_Observer([]))
    gov.receipt_path = tmp_path / "r.jsonl"
    table = _Table([["python", "-m", "mlx_lm", "lora"]])
    table.processes[0].pid = os.getpid()
    gov._observer = SimpleNamespace(
        process_table=lambda: table,
        provenance=_Observer([]).provenance,
    )

    assert gov.active_lora_processes() == []


# ── force is not authority over a physical collision ───────────────────────


def test_force_does_not_override_an_active_trainer(tmp_path):
    """force may override the POLICY gate (feature flag off). Two concurrent
    LoRA updates corrupt the adapter they share, so there is no emergency
    authority that makes that safe."""
    gov = _governor([["python", "-m", "mlx_lm", "lora", "--train"]], tmp_path)

    receipt = asyncio.run(
        gov.maybe_update_from_reflection("a reflection", force=True)
    )

    assert receipt.status == "blocked_existing_training"
    assert "force does not override" in receipt.reason


def test_force_still_overrides_the_disabled_flag(tmp_path, monkeypatch):
    """The policy gate remains overridable, or force would mean nothing."""
    monkeypatch.setenv("AURA_ONLINE_LORA", "0")
    gov = _governor([], tmp_path)

    receipt = asyncio.run(
        gov.maybe_update_from_reflection("a reflection", force=True)
    )

    assert receipt.status != "disabled"


# ── a receipt id is not approval just for existing ─────────────────────────


def test_unverifiable_receipt_id_does_not_authorise_by_itself(tmp_path, monkeypatch):
    """Any non-empty string used to authorise a weight mutation."""
    gov = _governor([], tmp_path)

    class _Will:
        def get_receipt(self, rid):
            return None  # no such receipt

        def decide(self, **kwargs):
            return SimpleNamespace(is_approved=lambda: False,
                                   receipt_id="w-1", reason="denied")

    monkeypatch.setattr("core.will.get_will", lambda: _Will())

    decision = gov._decide("reflection", will_receipt_id="totally-made-up")

    assert decision["approved"] is False


def test_a_confirmed_receipt_is_accepted(tmp_path, monkeypatch):
    gov = _governor([], tmp_path)

    class _Will:
        def get_receipt(self, rid):
            return SimpleNamespace(is_approved=lambda: True)

    monkeypatch.setattr("core.will.get_will", lambda: _Will())

    decision = gov._decide("reflection", will_receipt_id="w-real")

    assert decision["approved"] is True
    assert decision["receipt_id"] == "w-real"
