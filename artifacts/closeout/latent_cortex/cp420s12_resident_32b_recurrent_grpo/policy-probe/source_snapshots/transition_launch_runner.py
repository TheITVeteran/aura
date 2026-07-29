#!/usr/bin/env python3
"""Launch recurrent GRPO from one externally digest-pinned provider bundle."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Sequence
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.learning.verified_recurrent_transition_repository import (
    CAMPAIGN_FINALIZER_ID,
    DURABLE_REPLAY_LOADER_ID,
    INDEPENDENT_SCORER_ID,
    PRODUCTION_EVIDENCE_PRODUCER_ID,
    TOKEN_CODEC_ID,
    finalize_verified_recurrent_transition_campaign,
    load_recurrent_replay_packages,
    produce_verified_recurrent_transition_group,
    recurrent_trace_token_decoder,
    recurrent_trace_token_encoder,
    score_verified_recurrent_training_task,
)
from core.learning.verified_transition_launch_bundle import (
    VerifiedTransitionRuntimeComponents,
    load_verified_transition_provider_factory,
)


def verified_recurrent_runtime_components() -> VerifiedTransitionRuntimeComponents:
    """Return the fixed production component set bound by the launch contract."""

    return VerifiedTransitionRuntimeComponents(
        evidence_producer=produce_verified_recurrent_transition_group,
        evidence_producer_identity=PRODUCTION_EVIDENCE_PRODUCER_ID,
        durable_artifact_loader=load_recurrent_replay_packages,
        durable_artifact_loader_identity=DURABLE_REPLAY_LOADER_ID,
        campaign_finalizer=finalize_verified_recurrent_transition_campaign,
        campaign_finalizer_identity=CAMPAIGN_FINALIZER_ID,
        independent_scorer=score_verified_recurrent_training_task,
        scorer_identity=INDEPENDENT_SCORER_ID,
        token_encoder=recurrent_trace_token_encoder,
        token_decoder=recurrent_trace_token_decoder,
        token_codec_identity=TOKEN_CODEC_ID,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Validate an externally pinned verified-transition launch bundle, "
            "then execute recurrent GRPO through its post-load provider factory."
        )
    )
    parser.add_argument("--verified-launch-bundle", required=True)
    parser.add_argument(
        "--expected-launch-bundle-sha256",
        required=True,
        help="SHA-256 supplied by an external campaign controller",
    )
    parser.add_argument(
        "--expected-preregistration-sha256",
        required=True,
        help="SHA-256 of the externally preregistered campaign contract",
    )
    wrapper_args, training_argv = parser.parse_known_args(argv)
    if not training_argv:
        parser.error("the recurrent GRPO training arguments are required")

    factory = load_verified_transition_provider_factory(
        wrapper_args.verified_launch_bundle,
        expected_bundle_sha256=wrapper_args.expected_launch_bundle_sha256,
        expected_preregistration_sha256=(
            wrapper_args.expected_preregistration_sha256
        ),
        components=verified_recurrent_runtime_components(),
        now_unix=int(time.time()),
    )
    supplied_training_argv = ("tools/train_grpo.py", *training_argv)
    if tuple(factory.training_argv) != supplied_training_argv:
        parser.error(
            "training arguments differ from the externally frozen provider contract"
        )

    from tools import train_grpo

    previous = list(sys.argv)
    try:
        sys.argv = list(supplied_training_argv)
        return int(train_grpo.main(verified_group_provider_factory=factory))
    finally:
        sys.argv = previous


if __name__ == "__main__":
    raise SystemExit(main())
