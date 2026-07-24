#!/usr/bin/env python3
"""Render docs/RLC_SPARK_LITERATURE.md from the typed dossier registry."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.brain.llm.latent_cortex.literature import (  # noqa: E402
    render_literature_markdown,
    validate_literature,
)

TARGET = REPO_ROOT / "docs" / "RLC_SPARK_LITERATURE.md"


def main() -> int:
    receipt = validate_literature()
    TARGET.write_text(render_literature_markdown(), encoding="utf-8")
    print(f"wrote {TARGET}")
    print(f"entries: {receipt['entry_count']}")
    print(f"registry_sha256: {receipt['registry_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
