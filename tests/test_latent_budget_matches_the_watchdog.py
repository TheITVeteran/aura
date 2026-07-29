"""Two budgets, neither aware of the other, and a turn died between them.

Measured live 2026-07-28. "Did you know the Earth's core is cold? Everyone is
wrong about that." — a pushback she had answered correctly twice in about
twelve seconds earlier the same evening — produced:

    I couldn't get to an answer I'd stand behind on that one, and I won't send
    you a thinner one and pass it off as the real thing.

The neural feed said why:

    worker_loop_stalled: age_s=40.42 : budget_s=40.0
    Soft-cancel requested for job seq=4 (worker_loop_stalled)
    Recursive Latent Cortex exhausted the single resident owner
    (soft_cancelled); refusing a late ordinary generation.
    stage=branch_select spent_layer_apps=177120 elapsed_s=88.56

Two independent defects met:

1. The latent lane granted itself up to 120s of wall clock, while the MLX
   worker cancels any job showing no TOKEN activity past its first-token
   ceiling — 40s on the foreground lane. branch_select does layer
   applications and emits no tokens, so a healthy latent episode is
   indistinguishable from a livelock. The lane could never use the allowance
   it gave itself.

2. When the watchdog then cancelled the job — which is what a cancel is FOR,
   releasing the resident model — the phase treated that as an exhausted
   owner and refused the ordinary generation that would have worked. An
   enhancement lane took the answer with it.
"""

from __future__ import annotations

import pytest

from core.phases.response_generation import ResponseGenerationPhase


class TestASoftCancelReleasesTheOwner:
    RECEIPT = {
        "episode_id": "cff541683fa2416c8bc390215718ec63",
        "last_stage": "failed",
        "input_token_count": 2257,
    }

    @pytest.mark.parametrize(
        "reason", ["soft_cancelled", "soft_cancel", "cancelled", "canceled"]
    )
    def test_a_cancel_is_not_exhaustion(self, reason):
        """The watchdog cancels precisely so the model stops being held."""
        assert not ResponseGenerationPhase._latent_owner_exhausted(
            reason, self.RECEIPT
        )

    @pytest.mark.parametrize(
        "reason",
        [
            "latent_timeout:40",
            "worker_identity_failed:x",
            "runtime_identity_unbound",
        ],
    )
    def test_the_genuinely_exhausted_cases_still_refuse(self, reason):
        assert ResponseGenerationPhase._latent_owner_exhausted(reason, self.RECEIPT)

    def test_a_receipt_failure_after_completion_still_falls_back(self):
        """The earlier fix in this family, kept."""
        assert not ResponseGenerationPhase._latent_owner_exhausted(
            "receipt_contract_failed:terminal_disposition_unproven",
            {"episode_id": "e", "last_stage": "complete", "input_token_count": 10},
        )


class TestTheLaneCannotOutrunTheWatchdog:
    def test_the_allowance_is_clamped_below_the_ceiling(self, monkeypatch):
        import core.brain.latent_cortex_service as service

        class _Client:
            def _first_token_hard_ceiling(self, *, foreground_request=False):
                return 40.0  # what the live foreground lane actually allowed

        monkeypatch.setattr(
            "core.brain.llm.mlx_client.get_mlx_client", lambda: _Client()
        )
        bounded = service._runtime_bounded_wall_clock_s(120.0, foreground_request=True)
        assert bounded < 40.0, bounded
        assert bounded == pytest.approx(40.0 - service._LATENT_WATCHDOG_MARGIN_S)

    def test_a_request_inside_the_ceiling_is_untouched(self, monkeypatch):
        import core.brain.latent_cortex_service as service

        class _Client:
            def _first_token_hard_ceiling(self, *, foreground_request=False):
                return 120.0

        monkeypatch.setattr(
            "core.brain.llm.mlx_client.get_mlx_client", lambda: _Client()
        )
        assert service._runtime_bounded_wall_clock_s(30.0, foreground_request=True) == 30.0

    def test_an_unreachable_client_does_not_shrink_the_budget(self, monkeypatch):
        """Guessing a smaller number here is the same mistake reversed."""
        import core.brain.latent_cortex_service as service

        def _boom():
            raise RuntimeError("no client")

        monkeypatch.setattr("core.brain.llm.mlx_client.get_mlx_client", _boom)
        assert service._runtime_bounded_wall_clock_s(120.0, foreground_request=True) == 120.0
