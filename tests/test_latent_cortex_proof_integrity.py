"""CP126: latent-cortex integrity proofs must not outlive what they prove.

These findings share one shape — an integrity claim that survives the thing
it was about, or an absent proof read as a passing one.
"""
from __future__ import annotations

import inspect


class TestMissingProofIsNotSafety:
    """requires_worker_recycle tested `is False` only.

    A parameter-integrity check that FAILED to run, or was skipped, leaves
    params_unchanged as None — and the worker kept serving weights whose
    integrity had never been established. Absent proof is exactly when a
    recycle matters most.
    """

    def _source(self) -> str:
        from core.brain.llm.latent_cortex import worker_handler

        return inspect.getsource(worker_handler)

    def test_recycle_requires_an_explicit_pass(self):
        source = self._source()
        assert "result.receipt.params_unchanged is not True" in source
        assert "result.receipt.params_unchanged is False" not in source

    def test_absent_proof_is_reported(self):
        source = self._source()
        assert "no parameter-integrity proof" in source

    def test_recycle_decision_truth_table(self):
        """None and False must both recycle; only True may skip it."""

        def decide(params_unchanged, fw_applied=False, fw_erased=True):
            return params_unchanged is not True or (fw_applied and fw_erased is not True)

        assert decide(None) is True, "absent proof must recycle"
        assert decide(False) is True, "failed proof must recycle"
        assert decide(True) is False, "a passing proof may skip the recycle"


class TestRequestedVerifierCannotVanish:
    def _source(self) -> str:
        from core.brain.llm.latent_cortex import worker_handler

        return inspect.getsource(worker_handler)

    def test_lost_verifier_is_recorded(self):
        source = self._source()
        assert "verifier_guidance_requested_without_tokenizer" in source

    def test_receipt_distinguishes_unwanted_from_lost(self):
        source = self._source()
        assert '"requested": True' in source
        assert '"available": False' in source
        assert '"reason": "tokenizer_unavailable"' in source


class TestFastWeightProofDoesNotCrossEpisodes:
    def test_attach_clears_the_previous_episode_proof(self):
        from core.brain.llm.latent_cortex import fast_weights

        source = inspect.getsource(fast_weights.EpisodicFastWeights.attach)
        # detach() empties `handles`, the only thing the re-attach guard
        # checks, so a stale erase_proven=True could vouch for weights it
        # never saw.
        for field in (
            "self.lifecycle.erase_proven = None",
            "self.lifecycle.exported = False",
            "self.lifecycle.erased = False",
            "self._exported_handles = []",
        ):
            assert field in source, field

    def test_reattach_is_still_blocked_while_attached(self):
        from core.brain.llm.latent_cortex import fast_weights

        source = inspect.getsource(fast_weights.EpisodicFastWeights.attach)
        assert "already attached — one episode at a time" in source
