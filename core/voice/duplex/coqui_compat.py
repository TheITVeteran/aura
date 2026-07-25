"""core/voice/duplex/coqui_compat.py — Make coqui-TTS import under transformers 5.x.

coqui-TTS 0.27 imports ``transformers.pytorch_utils.isin_mps_friendly``, a
helper that transformers removed in 5.x. The import fails at module load, so
*every* coqui path in this repo is currently dead — including the XTTS voice
cloning the legacy engine already tried to use.

The obvious fix is to pin transformers back. That is the wrong fix here: this
venv's transformers is what mlx-lm and the resident 32B are built against, and
moving it to satisfy a TTS package risks the actual mind to gain a voice
option. So instead we reinstate the one missing symbol.

The shim is the upstream implementation. ``torch.isin`` had no MPS kernel when
the helper was written, hence the broadcast-and-compare fallback; on every
other device it defers to ``torch.isin`` directly.

Idempotent, and never overwrites a real symbol — if a future transformers
brings it back, that one wins.
"""
from __future__ import annotations

import logging

from core.runtime.errors import record_degradation

logger = logging.getLogger("Aura.Voice.CoquiCompat")

_applied = False


def _isin_mps_friendly(elements, test_elements):
    import torch

    if not torch.is_tensor(test_elements):
        test_elements = torch.tensor(test_elements, device=elements.device)
    if elements.device.type == "mps":
        # MPS lacked an isin kernel: compare against every test element and
        # reduce, which is equivalent for the 1-D test sets callers use.
        return elements.unsqueeze(-1).eq(test_elements.flatten()).any(dim=-1)
    return torch.isin(elements, test_elements)


def apply() -> bool:
    """Install the compatibility shim. Returns True if coqui can now import."""
    global _applied
    if _applied:
        return True
    try:
        import transformers.pytorch_utils as pytorch_utils

        if not hasattr(pytorch_utils, "isin_mps_friendly"):
            pytorch_utils.isin_mps_friendly = _isin_mps_friendly
            logger.info("Installed transformers.isin_mps_friendly shim for coqui-TTS")
        _applied = True
        return True
    except (ImportError, AttributeError) as exc:
        record_degradation(
            "voice_duplex.coqui_compat",
            exc,
            action="cloned-voice engine unavailable; preset voices unaffected",
            severity="warning",
        )
        return False


def license_accepted() -> bool:
    """Has the user accepted the Coqui Public Model License for XTTS?

    XTTS-v2 ships under CPML, which is a licence decision for the operator to
    make, not something this code may assume. The same env flags the legacy
    engine honours are honoured here so there is one answer, not two.
    """
    import os

    return any(
        str(os.environ.get(name, "")).strip().lower() in ("1", "true", "yes", "on")
        for name in (
            "AURA_COQUI_CPML_ACCEPTED",
            "AURA_COQUI_COMMERCIAL_LICENSED",
            "COQUI_TOS_AGREED",
        )
    )
