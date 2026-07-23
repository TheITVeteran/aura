"""CP126 inference_gate — observation purity and honest cloud provenance."""
from __future__ import annotations

from core.brain.inference_gate import _verified_cloud_generation_metadata


class TestCloudProvenanceIsAttributedNotAsserted:
    """8ff3084b: configuration-derived attribution is not a verification."""

    def _result(self, **overrides):
        result = {
            "ok": True,
            "is_local": False,
            "provider_verified": True,
            "endpoint": "Gemini-Fast",
            "provider": "google",
            "model": "gemini-2.5-flash",
        }
        result.update(overrides)
        return result

    def test_configuration_attributed_result_is_still_accepted(self):
        assert _verified_cloud_generation_metadata(
            self._result(provider_attribution="router_configuration")
        ) is True

    def test_receipt_can_be_required(self):
        assert _verified_cloud_generation_metadata(
            self._result(provider_attribution="router_configuration"),
            require_provider_receipt=True,
        ) is False
        assert _verified_cloud_generation_metadata(
            self._result(provider_attribution="provider_receipt"),
            require_provider_receipt=True,
        ) is True

    def test_local_results_are_never_cloud(self):
        assert _verified_cloud_generation_metadata(self._result(is_local=True)) is False

    def test_unverified_endpoint_is_refused(self):
        assert _verified_cloud_generation_metadata(
            self._result(endpoint="all_failed")
        ) is False
        assert _verified_cloud_generation_metadata(
            self._result(provider="unknown")
        ) is False

    def test_failed_results_are_refused(self):
        assert _verified_cloud_generation_metadata(self._result(ok=False)) is False
        assert _verified_cloud_generation_metadata(None) is False
