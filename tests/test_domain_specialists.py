"""Contract tests for domain-specialist adapter training.

Real machinery under test: battery generation + domain concentration, store
slicing with provenance normalization, collision exclusion, the promote/refuse
gate (domain gain AND general non-collapse), and library registration. Only
the train/eval subprocesses are scripted fakes (same FakeRunner pattern as the
compounding suite).
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from core.learning.domain_specialists import (
    GENERAL_SEED,
    DomainSpecialistTrainer,
    SpecialistConfig,
    _domain_seed,
)

pytestmark = pytest.mark.unit

DOMAIN = "arithmetic_chain"
DOMAIN_SEED = _domain_seed(DOMAIN)


class FakeResult:
    def __init__(self, ok=True, stdout="", stderr=""):
        self.ok = ok
        self.returncode = 0 if ok else 1
        self.stdout, self.stderr = stdout, stderr


class FakeRunner:
    """Scripted train/eval subprocesses; accuracy keyed by (seed, has_adapter)."""

    def __init__(self, accuracy_script=None, *, fail_train=False):
        self.accuracy_script = accuracy_script or {}
        self.fail_train = fail_train
        self.commands: list[tuple[str, ...]] = []

    @staticmethod
    def _arg(argv, flag, default=""):
        argv = list(argv)
        return argv[argv.index(flag) + 1] if flag in argv else default

    def __call__(self, command, timeout_s):
        self.commands.append(tuple(command))
        joined = " ".join(command)
        if "heldout_eval.py" in joined:
            seed = int(self._arg(command, "--seed", "0"))
            has_adapter = bool(self._arg(command, "--adapter-path"))
            accuracy = self.accuracy_script.get((seed, has_adapter), 0.5)
            output = Path(self._arg(command, "--output"))
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps({"accuracy": accuracy}), encoding="utf-8")
            return FakeResult(stdout="eval ok")
        if "--train" in command:
            if self.fail_train:
                return FakeResult(ok=False, stderr="loss exploded")
            adapter = Path(self._arg(command, "--adapter-path"))
            adapter.mkdir(parents=True, exist_ok=True)
            (adapter / "adapters.safetensors").write_bytes(b"fake-lora")
            (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
            return FakeResult(stdout="train ok")
        return FakeResult(ok=False, stderr=f"unexpected: {joined}")


class FakeLibrary:
    def __init__(self):
        self.registered = []

    def register(self, adapter):
        self.registered.append(adapter)
        return True


def write_store(path: Path, domain: str, n: int, *, tag_prefix: str = "") -> None:
    with path.open("a", encoding="utf-8") as fh:
        for i in range(n):
            fh.write(json.dumps({
                "prompt": f"{domain} practice problem number {i}",
                "chosen": f"Answer: {i}",
                "rejected": f"Answer: {i + 999}",
                "domain": f"{tag_prefix}{domain}",
            }) + "\n")


@pytest.fixture
def setup(tmp_path):
    store = tmp_path / "prefs.jsonl"
    store.touch()
    config = SpecialistConfig(
        work_root=tmp_path / "specialists",
        store_path=store,
        base_model=str(tmp_path / "base-model"),
        min_pairs=8,
        battery_size=16,
        general_size=16,
    )
    (tmp_path / "base-model").mkdir()
    return config, store


def test_pair_counts_normalize_selfplay_prefix(setup):
    config, store = setup
    write_store(store, DOMAIN, 5, tag_prefix="selfplay:")
    write_store(store, DOMAIN, 4)
    write_store(store, "modular", 2)
    trainer = DomainSpecialistTrainer(config, command_runner=FakeRunner(), library=FakeLibrary())
    counts = trainer.domain_pair_counts()
    assert counts[DOMAIN] == 9
    assert counts["modular"] == 2
    assert trainer.eligible_domains() == [DOMAIN]  # modular below min_pairs


def test_promotes_on_domain_gain_without_collapse(setup):
    config, store = setup
    write_store(store, DOMAIN, 12)
    library = FakeLibrary()
    runner = FakeRunner({
        (DOMAIN_SEED, False): 0.50, (DOMAIN_SEED, True): 0.75,   # domain gain
        (GENERAL_SEED, False): 0.60, (GENERAL_SEED, True): 0.58,  # no collapse
    })
    trainer = DomainSpecialistTrainer(config, command_runner=runner, library=library)
    receipt = trainer.train_domain(DOMAIN)
    assert receipt.status == "promoted", receipt.reasons
    assert receipt.domain_candidate_accuracy == 0.75
    assert len(library.registered) == 1
    adapter = library.registered[0]
    assert adapter.task_types == {DOMAIN}
    assert adapter.quality == 0.75
    assert adapter.base_model == config.base_model
    assert receipt.registered_as == adapter.name
    # receipt persisted for audit
    receipts = list((config.work_root / "receipts").glob("*.json"))
    assert len(receipts) == 1


def test_refuses_without_domain_gain(setup):
    config, store = setup
    write_store(store, DOMAIN, 12)
    library = FakeLibrary()
    runner = FakeRunner({
        (DOMAIN_SEED, False): 0.70, (DOMAIN_SEED, True): 0.70,   # flat = refuse
        (GENERAL_SEED, False): 0.60, (GENERAL_SEED, True): 0.60,
    })
    trainer = DomainSpecialistTrainer(config, command_runner=runner, library=library)
    receipt = trainer.train_domain(DOMAIN)
    assert receipt.status == "refused"
    assert any("no_domain_gain" in r for r in receipt.reasons)
    assert library.registered == []


def test_refuses_on_general_collapse(setup):
    config, store = setup
    write_store(store, DOMAIN, 12)
    library = FakeLibrary()
    runner = FakeRunner({
        (DOMAIN_SEED, False): 0.50, (DOMAIN_SEED, True): 0.90,   # big domain gain
        (GENERAL_SEED, False): 0.70, (GENERAL_SEED, True): 0.40,  # bought with collapse
    })
    trainer = DomainSpecialistTrainer(config, command_runner=runner, library=library)
    receipt = trainer.train_domain(DOMAIN)
    assert receipt.status == "refused"
    assert any("general_collapse" in r for r in receipt.reasons)
    assert library.registered == []


def test_blocked_on_insufficient_pairs(setup):
    config, store = setup
    write_store(store, DOMAIN, 3)
    trainer = DomainSpecialistTrainer(config, command_runner=FakeRunner(), library=FakeLibrary())
    receipt = trainer.train_domain(DOMAIN)
    assert receipt.status == "blocked"
    assert any("insufficient_pairs" in r for r in receipt.reasons)


def test_train_failure_is_a_receipt_not_a_crash(setup):
    config, store = setup
    write_store(store, DOMAIN, 12)
    trainer = DomainSpecialistTrainer(
        config, command_runner=FakeRunner(fail_train=True), library=FakeLibrary()
    )
    receipt = trainer.train_domain(DOMAIN)
    assert receipt.status == "train_failed"


def test_specialist_seeds_stay_in_reserved_band():
    from core.learning.heldout_battery import BatterySpec, generate_battery

    for domain in ("arithmetic_chain", "modular", "string_transform", "program_output"):
        seed = _domain_seed(domain)
        assert 2000 <= seed < 3000  # above the general eval floor AND its gates
        pool = generate_battery(BatterySpec(seed=seed, size=160))
        assert sum(1 for t in pool if t.domain == domain) >= 16


def test_store_rows_carry_domain_provenance(tmp_path):
    from core.learning.verifiable_preference_harness import (
        Attempt,
        VerifiablePreferenceHarness,
    )

    store = tmp_path / "prefs.jsonl"
    harness = VerifiablePreferenceHarness(store_path=store)
    harness.ingest(
        "compute 2+2",
        [
            Attempt(candidate="Answer: 4", verified=True, checked=True, confidence=1.0),
            Attempt(candidate="Answer: 5", verified=False, checked=True),
        ],
        domain="selfplay:arithmetic_chain",
    )
    row = json.loads(store.read_text().splitlines()[0])
    assert row["domain"] == "selfplay:arithmetic_chain"
    # trainer-facing export stays bare DPO format
    rows = harness.export_dpo_rows()
    assert set(rows[0].keys()) == {"prompt", "chosen", "rejected"}
