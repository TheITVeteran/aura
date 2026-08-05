"""CP126: cancellation attribution, clock discrimination, bounded worker input."""
from __future__ import annotations

import time

import pytest

from core.brain.llm.mlx_client import (
    MLXLocalClient,
    _bounded_progress_value,
    _sleep_inclusive_monotonic,
)

TEST_MODEL = "/models/Qwen2.5-7B-Instruct-4bit"


@pytest.fixture
def client() -> MLXLocalClient:
    return MLXLocalClient(model_path=TEST_MODEL)


class TestExpectedCancellationIsRequestBound:
    def test_an_unrelated_cancellation_cannot_spend_a_planned_claim(self, client):
        client._note_expected_generation_cancellation("reboot", request_ids=["req-a"])
        assert client._consume_expected_generation_cancellation("req-b") == ""
        # The real one is still claimable — the credit form had already spent it.
        assert client._consume_expected_generation_cancellation("req-a") == "reboot"

    def test_a_claim_is_consumed_once(self, client):
        client._note_expected_generation_cancellation("reboot", request_ids=["req-a"])
        assert client._consume_expected_generation_cancellation("req-a") == "reboot"
        assert client._consume_expected_generation_cancellation("req-a") == ""

    def test_claims_expire_and_do_not_accumulate(self, client):
        client._expected_cancels = {"old": ("reboot", time.time() - 3600.0)}
        assert client._consume_expected_generation_cancellation("old") == ""
        assert client._expected_cancels == {}

    def test_an_empty_request_id_claims_nothing(self, client):
        client._note_expected_generation_cancellation("reboot", request_ids=["req-a"])
        assert client._consume_expected_generation_cancellation("") == ""
        assert client._consume_expected_generation_cancellation(None) == ""


class TestWorkerProgressIsBounded:
    def test_an_oversized_string_is_clamped(self):
        assert len(_bounded_progress_value("A" * 10_000)) == 200

    def test_a_long_sequence_is_clamped(self):
        assert len(_bounded_progress_value(list(range(1000)))) == 32

    def test_non_finite_numbers_become_none(self):
        assert _bounded_progress_value(float("inf")) is None
        assert _bounded_progress_value(float("nan")) is None
        assert _bounded_progress_value(3.5) == 3.5

    def test_an_unsupported_shape_is_named_not_dropped(self):
        assert _bounded_progress_value(object()).startswith("<unsupported:")

    def test_recorded_progress_is_clamped_end_to_end(self, client):
        client._current_request_id = "req-1"
        client._record_latent_progress({"id": "req-1", "stage": "S" * 5000, "elapsed_s": 1.0})
        assert len(client._latent_progress_by_request["req-1"]["stage"]) == 200

    def test_finished_requests_expire_from_the_progress_map(self, client):
        client._latent_progress_by_request = {
            "done": {"received_at_unix": time.time() - 10_000.0}
        }
        client._expire_latent_progress()
        assert "done" not in client._latent_progress_by_request
        assert client._latent_progress_evicted == 1

    def test_a_live_request_never_expires(self, client):
        client._current_request_id = "live"
        client._latent_progress_by_request = {
            "live": {"received_at_unix": time.time() - 10_000.0}
        }
        client._expire_latent_progress()
        assert "live" in client._latent_progress_by_request


class TestClockJumpIsNotHostSleep:
    def test_the_platform_offers_a_sleep_inclusive_clock(self):
        # Darwin: CLOCK_MONOTONIC counts suspend, time.monotonic() does not.
        value = _sleep_inclusive_monotonic()
        assert value is None or value > 0.0

    def test_a_forward_wall_jump_is_reported_as_a_clock_shift(self, client, monkeypatch):
        base_wall = 1_000_000.0
        base_mono = 500.0
        base_sleep_inclusive = 900.0
        client._clock_sample_wall = base_wall
        client._clock_sample_monotonic = base_mono
        client._clock_sample_sleep_inclusive = base_sleep_inclusive
        client._current_request_started_at = base_wall - 10.0

        # One second of real running time, and the wall clock jumped 600s.
        monkeypatch.setattr(time, "time", lambda: base_wall + 601.0)
        monkeypatch.setattr(time, "monotonic", lambda: base_mono + 1.0)
        monkeypatch.setattr(
            "core.brain.llm.mlx_client._sleep_inclusive_monotonic",
            lambda: base_sleep_inclusive + 1.0,
        )

        rebased = client._rebase_after_system_sleep()
        assert rebased == pytest.approx(600.0)
        assert client._clock_shift_events == 1
        assert client._clock_shift_total_s == pytest.approx(600.0)

    def test_real_sleep_is_not_reported_as_a_clock_shift(self, client, monkeypatch):
        base_wall = 1_000_000.0
        base_mono = 500.0
        base_sleep_inclusive = 900.0
        client._clock_sample_wall = base_wall
        client._clock_sample_monotonic = base_mono
        client._clock_sample_sleep_inclusive = base_sleep_inclusive
        client._current_request_started_at = base_wall - 10.0

        # The host slept 600s: wall advanced, the sleep-inclusive clock
        # advanced with it, and time.monotonic() did not.
        monkeypatch.setattr(time, "time", lambda: base_wall + 600.0)
        monkeypatch.setattr(time, "monotonic", lambda: base_mono)
        monkeypatch.setattr(
            "core.brain.llm.mlx_client._sleep_inclusive_monotonic",
            lambda: base_sleep_inclusive + 600.0,
        )

        rebased = client._rebase_after_system_sleep()
        assert rebased == pytest.approx(600.0)
        assert client._clock_shift_events == 0

    def test_without_a_sleep_inclusive_clock_the_old_heuristic_stands(
        self, client, monkeypatch
    ):
        client._clock_sample_wall = 1_000_000.0
        client._clock_sample_monotonic = 500.0
        client._clock_sample_sleep_inclusive = None
        monkeypatch.setattr(time, "time", lambda: 1_000_600.0)
        monkeypatch.setattr(time, "monotonic", lambda: 500.0)
        monkeypatch.setattr(
            "core.brain.llm.mlx_client._sleep_inclusive_monotonic", lambda: None
        )
        assert client._rebase_after_system_sleep() == pytest.approx(600.0)
        assert client._clock_shift_events == 0


class TestArtifactPromotionIsATransaction:
    """CP126 a996d77f/41fa9f3c/8ccdcd3b/df8e3045/7f4435f5."""

    def _artifact(self, root, name, arch="Qwen2ForCausalLM"):
        artifact = root / name
        artifact.mkdir()
        (artifact / "config.json").write_text('{"architectures": ["%s"]}' % arch)
        (artifact / "tokenizer.json").write_text("{}")
        (artifact / "model.safetensors").write_bytes(b"\x00" * 16)
        return artifact

    def test_a_servable_artifact_validates(self, tmp_path):
        from core.brain.llm.mlx_client import _validate_model_artifact

        verdict = _validate_model_artifact(self._artifact(tmp_path, "good"))
        assert verdict.ok
        assert verdict.architectures == ("Qwen2ForCausalLM",)
        assert verdict.weight_files == 1

    def test_an_unparseable_config_is_refused(self, tmp_path):
        from core.brain.llm.mlx_client import _validate_model_artifact

        artifact = tmp_path / "broken"
        artifact.mkdir()
        (artifact / "config.json").write_text("{not json")
        verdict = _validate_model_artifact(artifact)
        assert not verdict.ok
        assert verdict.reason.startswith("artifact_config_unreadable")

    def test_a_missing_tokenizer_is_refused(self, tmp_path):
        from core.brain.llm.mlx_client import _validate_model_artifact

        artifact = tmp_path / "no-tok"
        artifact.mkdir()
        (artifact / "config.json").write_text("{}")
        (artifact / "model.safetensors").write_bytes(b"\x00")
        assert _validate_model_artifact(artifact).reason == "artifact_missing_tokenizer"

    def test_registry_rebind_moves_the_client_to_its_new_key(self, client):
        import core.brain.llm.mlx_client as mod

        with mod._CLIENTS_LOCK:
            mod._CLIENTS["/old/path"] = client
        try:
            assert mod._rebind_client_registry_key("/old/path", "/new/path", client)
            snapshot = dict(mod.clients_snapshot())
            assert snapshot.get("/new/path") is client
            assert "/old/path" not in snapshot
        finally:
            with mod._CLIENTS_LOCK:
                mod._CLIENTS.pop("/new/path", None)
                mod._CLIENTS.pop("/old/path", None)

    def test_registry_rebind_refuses_to_evict_a_different_client(self, client):
        import core.brain.llm.mlx_client as mod

        other = MLXLocalClient(model_path=TEST_MODEL)
        with mod._CLIENTS_LOCK:
            mod._CLIENTS["/old/path"] = client
            mod._CLIENTS["/new/path"] = other
        try:
            assert not mod._rebind_client_registry_key("/old/path", "/new/path", client)
            assert dict(mod.clients_snapshot())["/new/path"] is other
        finally:
            with mod._CLIENTS_LOCK:
                mod._CLIENTS.pop("/new/path", None)
                mod._CLIENTS.pop("/old/path", None)

    @pytest.mark.asyncio
    async def test_a_failed_recycle_reports_failed_not_ok(self, client, tmp_path, monkeypatch):
        async def _boom(reason="", mark_failed=True):
            raise RuntimeError("spawn refused")

        monkeypatch.setattr(client, "reboot_worker", _boom)
        target = self._artifact(tmp_path, "fused")
        receipt = await client._activate_promoted_artifact(str(target))
        assert receipt["ok"] is False
        assert receipt["state"] == "failed"
        assert receipt["reason"].startswith("recycle_failed")
