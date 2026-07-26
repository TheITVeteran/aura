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
        assert 'body["requires_worker_recycle"] = not integrity_safe' in source
        assert "runtime_integrity_safe(" in source

    def test_absent_proof_is_reported(self):
        source = self._source()
        assert "no complete worker-bound runtime-integrity" in source

    def test_recycle_decision_truth_table(self):
        """Only a complete measured, worker-bound proof may skip recycle."""
        from core.brain.llm.latent_cortex.runtime_integrity import (
            runtime_integrity_safe,
        )
        from tests.fixtures.rlc_runtime_integrity import (
            bound_runtime_integrity,
        )

        assert runtime_integrity_safe({}) is False
        assert runtime_integrity_safe(
            bound_runtime_integrity(
                episode_id="proof-integrity-test",
                input_tokens_sha256="7" * 64,
            ),
            expected_fast_weights_applied=False,
        )


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
            # Replacing the complete lifecycle is stronger than clearing a
            # hand-picked subset: optimizer trails, lease identity, conflict
            # counts, export state, and erase evidence all reset together.
            "self.lifecycle = FastWeightsLifecycle()",
            "self._exported_handles = []",
        ):
            assert field in source, field

    def test_reattach_is_still_blocked_while_attached(self):
        from core.brain.llm.latent_cortex import fast_weights

        source = inspect.getsource(fast_weights.EpisodicFastWeights.attach)
        assert "already attached — one episode at a time" in source
