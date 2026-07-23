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


class TestActivationRollback:
    """fcc5ab93: a failed load must not destroy the adapters it evicted."""

    def test_failed_load_restores_the_evictee(self, tmp_path):
        class _FailsSecondLoad(_Applier):
            def load(self, adapter):
                self.loaded.append(adapter.name)
                return adapter.name != "b"

        applier = _FailsSecondLoad()
        lib = ExpertLoRALibrary(tmp_path / "lib.json", max_resident=1, applier=applier)
        lib.register(_adapter(tmp_path, "a"))
        lib.register(_adapter(tmp_path, "b"))
        assert lib.activate("a") is True
        assert lib.activate("b") is False
        # "a" was evicted for "b"; "b" failed, so "a" must be back.
        assert lib.resident() == ["a"]
        assert applier.loaded == ["a", "b", "a"]

    @pytest.mark.asyncio
    async def test_async_failed_load_restores_the_evictee(self, tmp_path):
        class _FailsSecondLoad(_AsyncApplier):
            async def load(self, adapter):
                self.loaded.append(adapter.name)
                return adapter.name != "b"

        applier = _FailsSecondLoad()
        lib = ExpertLoRALibrary(tmp_path / "lib.json", max_resident=1)
        lib.register(_adapter(tmp_path, "a"))
        lib.register(_adapter(tmp_path, "b"))
        assert await lib.activate_async("a", applier) is True
        assert await lib.activate_async("b", applier) is False
        assert lib.resident() == ["a"]


class TestSyncActivationDoesNotHoldTheLock:
    """0889dd98: applier I/O must not block registry readers."""

    def test_registry_is_readable_during_a_slow_load(self, tmp_path):
        import threading

        started = threading.Event()
        release = threading.Event()
        observed: dict[str, object] = {}

        class _SlowApplier(_Applier):
            def load(self, adapter):
                started.set()
                release.wait(timeout=5.0)
                return super().load(adapter)

        lib = ExpertLoRALibrary(tmp_path / "lib.json", applier=_SlowApplier())
        lib.register(_adapter(tmp_path, "a"))

        worker = threading.Thread(target=lambda: lib.activate("a"))
        worker.start()
        assert started.wait(timeout=5.0)
        # The registry must answer WHILE the load is in flight.
        reader = threading.Thread(target=lambda: observed.update(n=len(lib.list())))
        reader.start()
        reader.join(timeout=3.0)
        release.set()
        worker.join(timeout=5.0)
        assert reader.is_alive() is False, "registry read blocked on applier I/O"
        assert observed.get("n") == 1


class TestSyncLoadExceptionIsContained:
    """34f303bc: a raising load must not escape after state mutation."""

    def test_exception_becomes_a_failed_activation(self, tmp_path):
        class _Boom(_Applier):
            def load(self, adapter):
                raise RuntimeError("metal OOM during attach")

        lib = ExpertLoRALibrary(tmp_path / "lib.json", applier=_Boom())
        lib.register(_adapter(tmp_path, "a"))
        assert lib.activate("a") is False
        assert lib.resident() == []


class TestBaseModelCompatibility:
    """a0cf1594: unknown base is unverified, not compatible."""

    def test_unknown_base_is_not_selected_when_a_base_is_required(self, tmp_path):
        lib = ExpertLoRALibrary(tmp_path / "lib.json", applier=_Applier())
        untagged = _adapter(tmp_path, "untagged")
        untagged.base_model = ""
        lib.register(untagged)
        assert lib.select_for("math task", "math", base_model="Qwen2.5-32B") is None

    def test_matching_base_still_selected(self, tmp_path):
        lib = ExpertLoRALibrary(tmp_path / "lib.json", applier=_Applier())
        adapter = _adapter(tmp_path, "matched")
        adapter.base_model = "Qwen2.5-32B"
        lib.register(adapter)
        assert lib.select_for("math task", "math", base_model="Qwen2.5-32B") is not None


class TestUntaggedAdaptersAreNotUniversal:
    """9abdd2bf: an empty task_types set is not 'matches everything'."""

    def test_untagged_adapter_is_never_selected(self, tmp_path):
        lib = ExpertLoRALibrary(tmp_path / "lib.json", applier=_Applier())
        untagged = LoRAAdapter(name="scanned", path=_artifact(tmp_path, "scanned"))
        lib.register(untagged)
        assert lib.select_for("anything at all", "math") is None
        assert lib.select_for("anything at all", "") is None


class TestMetadataValidation:
    """079e8448: malformed manifest values must not enter ranking."""

    def test_string_tags_are_one_tag_not_characters(self):
        adapter = LoRAAdapter.from_dict(
            {"name": "a", "path": "/p", "task_types": "math", "keywords": "algebra"}
        )
        assert adapter.task_types == {"math"}
        assert adapter.keywords == {"algebra"}

    def test_non_finite_values_are_rejected(self):
        adapter = LoRAAdapter.from_dict(
            {"name": "a", "path": "/p", "quality": float("nan"), "size_mb": float("inf")}
        )
        assert adapter.quality == 0.5
        assert adapter.size_mb == 0.0

    def test_quality_is_bounded(self):
        assert LoRAAdapter.from_dict({"name": "a", "path": "/p", "quality": 99}).quality == 1.0
        assert LoRAAdapter.from_dict({"name": "a", "path": "/p", "quality": -5}).quality == 0.0

    def test_non_mapping_rejected(self):
        with pytest.raises(ValueError):
            LoRAAdapter.from_dict(["not", "a", "mapping"])


class TestRegistryEncapsulation:
    """37e18573: reads must not hand out mutable internal objects."""

    def test_mutating_a_returned_adapter_does_not_change_the_registry(self, tmp_path):
        lib = ExpertLoRALibrary(tmp_path / "lib.json", applier=_Applier())
        lib.register(_adapter(tmp_path, "a"))
        got = lib.get("a")
        got.path = "/evil"
        got.quality = 1.0
        got.task_types.add("everything")
        fresh = lib.get("a")
        assert fresh.path != "/evil"
        assert fresh.quality == 0.5
        assert "everything" not in fresh.task_types

    def test_list_returns_snapshots(self, tmp_path):
        lib = ExpertLoRALibrary(tmp_path / "lib.json", applier=_Applier())
        lib.register(_adapter(tmp_path, "a"))
        lib.list()[0].name = "hijacked"
        assert lib.get("a") is not None


class TestPersistenceTruth:
    """d09b9a74: registration is durable or it failed."""

    def test_failed_persist_rolls_back_registration(self, tmp_path, monkeypatch):
        lib = ExpertLoRALibrary(tmp_path / "lib.json", applier=_Applier())
        monkeypatch.setattr(ExpertLoRALibrary, "_persist", lambda self: False)
        assert lib.register(_adapter(tmp_path, "a")) is False
        assert lib.get("a") is None


class TestManifestKeyIntegrity:
    """68e1d462: the manifest key is authoritative."""

    def test_mismatched_inner_name_is_reconciled(self, tmp_path):
        import json as _json

        manifest = tmp_path / "lib.json"
        manifest.write_text(
            _json.dumps(
                {
                    "schema_version": 1,
                    "adapters": {
                        "outer_key": {
                            "name": "inner_name",
                            "path": _artifact(tmp_path, "art"),
                            "task_types": ["math"],
                        }
                    },
                }
            ),
            encoding="utf-8",
        )
        lib = ExpertLoRALibrary(manifest, applier=_Applier())
        adapter = lib.get("outer_key")
        assert adapter is not None
        assert adapter.name == "outer_key", "registry key and adapter name disagree"


class TestScanIntegrity:
    """315080b7: a config file alone is not an adapter."""

    def test_config_only_directory_is_skipped(self, tmp_path):
        lib = ExpertLoRALibrary(tmp_path / "lib.json", applier=_Applier())
        d = tmp_path / "scan" / "config_only"
        d.mkdir(parents=True)
        (d / "adapter_config.json").write_text("{}", encoding="utf-8")
        assert lib.scan(tmp_path / "scan") == 0

    def test_directory_with_weights_is_registered_and_sized_recursively(self, tmp_path):
        lib = ExpertLoRALibrary(tmp_path / "lib.json", applier=_Applier())
        d = tmp_path / "scan" / "real"
        (d / "nested").mkdir(parents=True)
        (d / "adapter_config.json").write_text("{}", encoding="utf-8")
        (d / "adapters.safetensors").write_bytes(b"0" * 1024)
        (d / "nested" / "extra.safetensors").write_bytes(b"0" * 2048)
        assert lib.scan(tmp_path / "scan") == 1
        adapter = lib.get("real")
        assert adapter.size_mb > 0
        # Nested weights must be counted, not omitted.
        assert adapter.size_mb == pytest.approx((1024 + 2048 + 2) / (1024 * 1024), rel=0.2)


class TestResetReleasesWeights:
    """21661496: model modifications must not outlive the registry."""

    def test_reset_unloads_resident_adapters(self, tmp_path, monkeypatch):
        from core.brain import expert_lora_library as mod

        applier = _Applier()
        lib = ExpertLoRALibrary(tmp_path / "lib.json", applier=applier)
        lib.register(_adapter(tmp_path, "a"))
        assert lib.activate("a") is True
        monkeypatch.setattr(mod, "_singleton", lib)
        mod.reset_expert_lora_library()
        assert applier.unloaded == ["a"], "weights outlived the registry"
        assert mod._singleton is None
