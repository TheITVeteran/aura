#!/usr/bin/env python3
"""Train and seal Aura's systematic neural ALU artifact."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.brain.llm.latent_cortex.systematic_neural_alu import (  # noqa: E402
    DEFAULT_SYSTEMATIC_NEURAL_ALU_ARTIFACT,
)
from core.learning.systematic_neural_alu_training import (  # noqa: E402
    train_and_write_systematic_neural_alu_artifact,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_SYSTEMATIC_NEURAL_ALU_ARTIFACT,
    )
    args = parser.parse_args()
    manifest = train_and_write_systematic_neural_alu_artifact(args.out)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
