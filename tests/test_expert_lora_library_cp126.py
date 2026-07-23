"""CP126 expert_lora_library — residency attestation, transactions, budget.

Every test pins one finding from artifacts/closeout/semantic_review/cp126/.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from core.brain.expert_lora_library import (
    ExpertLoRALibrary,
    LoRAAdapter,
    NoopApplier,
    _attest_adapter_artifact,
)


def _artifact(root: Path, name: str) -> str:
    d = root / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "adapter_config.json").write_text("{}", encoding="utf-8")
    return str(d)


def _adapter(root: Path, name: str, *, size_mb: float = 1.0) -> LoRAAdapter:
    return LoRAAdapter(
        name=name,
        path=_artifact(root, name),
        task_types={"math"},
        keywords={name},
        size_mb=size_mb,
    )


class _Applier:
    """A real weight-attaching applier."""

    def __init__(self, *, load_ok=True, unload_ok=True):
        self.load_ok, self.unload_ok = load_ok, unload_ok
        self.loaded: list[str] = []
        self.unloaded: list[str] = []

    def load(self, adapter):
        self.loaded.append(adapter.name)
        return self.load_ok

    def unload(self, adapter):
        self.unloaded.append(adapter.name)
        return self.unload_ok


class _AsyncApplier:
    def __init__(self, *, load_ok=True, unload_ok=True):
        self.load_ok, self.unload_ok = load_ok, unload_ok
        self.loaded: list[str] = []
        self.unloaded: list[str] = []

    async def load(self, adapter):
        self.loaded.append(adapter.name)
        return self.load_ok

    async def unload(self, adapter):
        self.unloaded.append(adapter.name)
        return self.unload_ok


class TestNoopApplierIsHonest:
    """20a12402: the default applier must not attest physical residency."""

    def test_noop_load_reports_failure(self):
        applier = NoopApplier()
        adapter = LoRAAdapter(name="x", path="/nope")
        assert applier.load(adapter) is False
        assert applier.unload(adapter) is True
        assert applier.attaches_weights is False

    def test_default_library_never_claims_residency(self, tmp_path):
        lib = ExpertLoRALibrary(tmp_path / "lib.json", max_resident=2)
        assert lib.register(_adapter(tmp_path, "a"))
        assert lib.activate("a") is False
        assert lib.resident() == []

    def test_select_and_activate_returns_none_without_a_real_applier(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.setenv("AURA_EXPERT_LORA_LIBRARY", "1")
        lib = ExpertLoRALibrary(tmp_path / "lib.json", max_resident=2)
        lib.register(_adapter(tmp_path, "a"))
        assert lib.select_and_activate("math task", "math") is None


class TestArtifactAttestation:
    """70c50967: registration proves the artifact before anything loads it."""

    def test_missing_artifact_refused(self, tmp_path):
        lib = ExpertLoRALibrary(tmp_path / "lib.json", applier=_Applier())
        assert lib.register(LoRAAdapter(name="ghost", path=str(tmp_path / "nope"))) is False
        assert lib.get("ghost") is None

    def test_directory_without_markers_refused(self, tmp_path):
        empty = tmp_path / "empty"
        empty.mkdir()
        ok, reason = _attest_adapter_artifact(LoRAAdapter(name="e", path=str(empty)))
        assert ok is False and reason == "no_adapter_markers"

    def test_real_artifact_accepted(self, tmp_path):
        lib = ExpertLoRALibrary(tmp_path / "lib.json", applier=_Applier())
        assert lib.register(_adapter(tmp_path, "real")) is True

    def test_unsupported_file_type_refused(self, tmp_path):
        bad = tmp_path / "weights.bin"
        bad.write_bytes(b"x")
        ok, reason = _attest_adapter_artifact(LoRAAdapter(name="b", path=str(bad)))
        assert ok is False and reason == "unsupported_artifact_type"

    def test_empty_path_refused(self):
        ok, reason = _attest_adapter_artifact(LoRAAdapter(name="n", path="  "))
        assert ok is False and reason == "empty_path"


class TestUnregisterUnloadsFirst:
    """4637a168: unregister must not orphan attached weights."""

    def test_unload_happens_while_adapter_is_resolvable(self, tmp_path):
        applier = _Applier()
        lib = ExpertLoRALibrary(tmp_path / "lib.json", applier=applier)
        lib.register(_adapter(tmp_path, "a"))
        assert lib.activate("a") is True
        assert lib.unregister("a") is True
        assert applier.unloaded == ["a"], "weights were left attached"
        assert lib.resident() == []

    def test_refused_unload_keeps_registration(self, tmp_path):
        applier = _Applier(unload_ok=False)
        lib = ExpertLoRALibrary(tmp_path / "lib.json", applier=applier)
        lib.register(_adapter(tmp_path, "a"))
        lib.activate("a")
        assert lib.unregister("a") is False
        assert lib.get("a") is not None
        assert lib.resident() == ["a"], "residency was freed without a confirmed unload"


class TestEvictionRequiresConfirmedUnload:
    """b022149a: never claim capacity is free before the unload confirms."""

    def test_failed_unload_keeps_residency(self, tmp_path):
        applier = _Applier(unload_ok=False)
        lib = ExpertLoRALibrary(tmp_path / "lib.json", max_resident=1, applier=applier)
        lib.register(_adapter(tmp_path, "a"))
        lib.register(_adapter(tmp_path, "b"))
        assert lib.activate("a") is True
        # "b" cannot be admitted because "a" refuses to unload.
        assert lib.activate("b") is False
        assert lib.resident() == ["a"]

    def test_unload_exception_keeps_residency(self, tmp_path):
        class _Boom(_Applier):
            def unload(self, adapter):
                raise RuntimeError("metal detached mid-unload")

        lib = ExpertLoRALibrary(tmp_path / "lib.json", max_resident=1, applier=_Boom())
        lib.register(_adapter(tmp_path, "a"))
        lib.register(_adapter(tmp_path, "b"))
        lib.activate("a")
        assert lib.activate("b") is False
        assert lib.resident() == ["a"]


class TestDuplicateRegistration:
    """c6f1e993: replacing a resident adapter must not orphan its weights."""

    def test_changed_artifact_evicts_the_live_one_first(self, tmp_path):
        applier = _Applier()
        lib = ExpertLoRALibrary(tmp_path / "lib.json", applier=applier)
        lib.register(_adapter(tmp_path, "a"))
        lib.activate("a")
        replacement = LoRAAdapter(name="a", path=_artifact(tmp_path, "a_v2"))
        assert lib.register(replacement) is True
        assert applier.unloaded == ["a"], "old weights were orphaned"
        assert lib.resident() == []
        assert lib.get("a").path.endswith("a_v2")

    def test_blocked_when_old_weights_cannot_be_unloaded(self, tmp_path):
        applier = _Applier(unload_ok=False)
        lib = ExpertLoRALibrary(tmp_path / "lib.json", applier=applier)
        lib.register(_adapter(tmp_path, "a"))
        lib.activate("a")
        replacement = LoRAAdapter(name="a", path=_artifact(tmp_path, "a_v2"))
        assert lib.register(replacement) is False
        assert lib.get("a").path.endswith("/a")


class TestMemoryBudget:
    """bddade3a: the advertised RAM budget must actually gate admission."""

    def test_large_adapters_are_budgeted_not_just_counted(self, tmp_path):
        applier = _Applier()
        lib = ExpertLoRALibrary(tmp_path / "lib.json", max_resident=8, applier=applier)
        lib.register(_adapter(tmp_path, "big1", size_mb=3000.0))
        lib.register(_adapter(tmp_path, "big2", size_mb=3000.0))
        assert lib.activate("big1") is True
        # Count budget allows 8, but 3000+3000 MB exceeds the 4096 MB budget:
        # admission evicts the LRU rather than exceeding memory.
        assert lib.activate("big2") is True
        assert lib.resident() == ["big2"]
        assert applier.unloaded == ["big1"]

    def test_small_adapters_coexist(self, tmp_path):
        lib = ExpertLoRALibrary(tmp_path / "lib.json", max_resident=4, applier=_Applier())
        for name in ("s1", "s2", "s3"):
            lib.register(_adapter(tmp_path, name, size_mb=10.0))
            assert lib.activate(name) is True
        assert set(lib.resident()) == {"s1", "s2", "s3"}

    def test_stats_report_the_budget(self, tmp_path):
        lib = ExpertLoRALibrary(tmp_path / "lib.json", applier=_Applier())
        lib.register(_adapter(tmp_path, "a", size_mb=12.0))
        lib.activate("a")
        stats = lib.stats()
        assert stats["resident_mb"] == pytest.approx(12.0)
        assert stats["max_resident_mb"] > 0


class TestApplierOwnership:
    """58bc5a8d: residency is per-applier, not a shared name set."""

    @pytest.mark.asyncio
    async def test_second_applier_does_not_inherit_residency(self, tmp_path):
        lib = ExpertLoRALibrary(tmp_path / "lib.json", max_resident=2)
        lib.register(_adapter(tmp_path, "a"))
        worker_one = _AsyncApplier()
        worker_two = _AsyncApplier()
        assert await lib.activate_async("a", worker_one) is True
        # worker_two never loaded it — claiming True would attest weights that
        # are not attached in ITS model.
        assert await lib.activate_async("a", worker_two) is False
        assert worker_two.loaded == []
        assert lib.resident_for(worker_one) == ["a"]
        assert lib.resident_for(worker_two) == []

    @pytest.mark.asyncio
    async def test_eviction_uses_the_loading_applier(self, tmp_path):
        lib = ExpertLoRALibrary(tmp_path / "lib.json", max_resident=1)
        lib.register(_adapter(tmp_path, "a"))
        lib.register(_adapter(tmp_path, "b"))
        owner = _AsyncApplier()
        assert await lib.activate_async("a", owner) is True
        assert await lib.activate_async("b", owner) is True
        assert owner.unloaded == ["a"]


class TestAsyncSlotReservation:
    """be41d7a1: concurrent async activations cannot exceed the budget."""

    @pytest.mark.asyncio
    async def test_concurrent_activations_respect_max_resident(self, tmp_path):
        import asyncio

        class _SlowApplier(_AsyncApplier):
            async def load(self, adapter):
                await asyncio.sleep(0.02)  # release the loop mid-load
                return await super().load(adapter)

        lib = ExpertLoRALibrary(tmp_path / "lib.json", max_resident=1)
        for name in ("a", "b"):
            lib.register(_adapter(tmp_path, name))
        applier = _SlowApplier()
        results = await asyncio.gather(
            lib.activate_async("a", applier),
            lib.activate_async("b", applier),
        )
        assert len(lib.resident()) <= 1, "residency limit exceeded under concurrency"
        assert any(results)

    @pytest.mark.asyncio
    async def test_duplicate_concurrent_activation_loads_once(self, tmp_path):
        import asyncio

        class _SlowApplier(_AsyncApplier):
            async def load(self, adapter):
                await asyncio.sleep(0.02)
                return await super().load(adapter)

        lib = ExpertLoRALibrary(tmp_path / "lib.json", max_resident=2)
        lib.register(_adapter(tmp_path, "a"))
        applier = _SlowApplier()
        await asyncio.gather(
            lib.activate_async("a", applier),
            lib.activate_async("a", applier),
        )
        assert applier.loaded == ["a"], "the same adapter was loaded twice"

    @pytest.mark.asyncio
    async def test_refused_evictee_unload_blocks_the_new_activation(self, tmp_path):
        lib = ExpertLoRALibrary(tmp_path / "lib.json", max_resident=1)
        lib.register(_adapter(tmp_path, "a"))
        lib.register(_adapter(tmp_path, "b"))
        applier = _AsyncApplier()
        assert await lib.activate_async("a", applier) is True
        applier.unload_ok = False
        assert await lib.activate_async("b", applier) is False
        assert lib.resident() == ["a"]
        assert "b" not in applier.loaded
