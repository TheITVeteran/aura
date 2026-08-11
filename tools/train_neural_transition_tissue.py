#!/usr/bin/env python3
"""Train and seal Aura's bounded teacher-removed neural transition tissue."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.brain.llm.latent_cortex.neural_transition_tissue import (  # noqa: E402
    DEFAULT_NEURAL_TRANSITION_ARTIFACT,
    load_neural_transition_tissue,
)
from core.learning.neural_transition_training import (  # noqa: E402
    train_and_write_neural_transition_artifact,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_NEURAL_TRANSITION_ARTIFACT)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.2)
    args = parser.parse_args()
    manifest = train_and_write_neural_transition_artifact(
        args.out,
        steps=args.steps,
        learning_rate=args.learning_rate,
    )
    loaded = load_neural_transition_tissue(args.out)
    if loaded.tissue_sha256 != manifest["weights_sha256"]:
        raise RuntimeError("sealed neural transition artifact did not reload exactly")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
