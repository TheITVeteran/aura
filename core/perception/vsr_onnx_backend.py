"""core/perception/vsr_onnx_backend.py
────────────────────────────────────
Open-vocabulary VSR backend on ONNX Runtime.

This is the real inference path behind VisualSpeechEngine's ``Backend``
protocol: it takes mouth-crop frames, preprocesses them the way lip-
reading models expect (grayscale, resize to 88×88, per-video mean/std
normalization, temporal tensor), runs a CTC-head ONNX model, and
decodes the frame logits into an open-vocabulary transcript with
prefix-beam search.

Model provenance is explicit and honest. Frontier checkpoints (the
auto_avsr conformer family) reach ~20% WER but their WEIGHTS inherit the
LRS3/LRS2/VoxCeleb2 research-data license — that term is the model
provider's, not something a user can waive by clicking. So:

- The *pipeline* here is complete, production-grade, and fully tested
  against a real ONNX CTC model.
- ``load_onnx_backend`` surfaces the license/provenance of whatever
  checkpoint is dropped in and records the operator's acknowledgement —
  it does not silently ship restricted weights.

Swap the model file; the open-vocabulary machinery does not change.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from core.perception.vsr_ctc import (
    Vocabulary,
    beam_search_decode,
    default_vocabulary,
    greedy_decode,
)
from core.runtime.errors import record_degradation

logger = logging.getLogger("Aura.Perception.VSR")

_VSR_ERRORS = (ImportError, AttributeError, OSError, RuntimeError, TypeError, ValueError)
MOUTH_SIZE = 88


def preprocess_mouth_crops(mouth_crops: NDArray[np.uint8]) -> NDArray[np.float32]:
    """(T, H, W, 3|1) uint8 → (1, 1, T, 88, 88) float32, grayscale,
    per-video standardized — the canonical VSR front end."""
    frames = np.asarray(mouth_crops)
    if frames.ndim == 4 and frames.shape[-1] == 3:
        # BT.601 luma.
        gray = (0.299 * frames[..., 2] + 0.587 * frames[..., 1]
                + 0.114 * frames[..., 0])
    elif frames.ndim == 4 and frames.shape[-1] == 1:
        gray = frames[..., 0].astype(np.float64)
    elif frames.ndim == 3:
        gray = frames.astype(np.float64)
    else:
        raise ValueError("mouth_crops must be (T,H,W,3), (T,H,W,1) or (T,H,W)")
    if gray.shape[0] < 1:
        raise ValueError("need at least one frame")

    resized = _resize_stack(gray, MOUTH_SIZE)
    resized = resized / 255.0
    mean = float(resized.mean())
    std = float(resized.std()) or 1.0
    normalized = (resized - mean) / std
    return normalized.astype(np.float32)[None, None, :, :, :]


def _resize_stack(gray: np.ndarray, size: int) -> np.ndarray:
    """Bilinear resize each frame to size×size (pure numpy, no cv2 dep)."""
    frames, h, w = gray.shape
    if (h, w) == (size, size):
        return gray
    ys = np.linspace(0, h - 1, size)
    xs = np.linspace(0, w - 1, size)
    y0 = np.clip(np.floor(ys).astype(int), 0, h - 2)
    x0 = np.clip(np.floor(xs).astype(int), 0, w - 2)
    wy = (ys - y0)[None, :, None]
    wx = (xs - x0)[None, None, :]
    top = gray[:, y0][:, :, x0] * (1 - wx) + gray[:, y0][:, :, x0 + 1] * wx
    bot = gray[:, y0 + 1][:, :, x0] * (1 - wx) + gray[:, y0 + 1][:, :, x0 + 1] * wx
    return top * (1 - wy) + bot * wy


@dataclass
class ModelProvenance:
    model_id: str
    license: str
    training_data: str
    acknowledged: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "license": self.license,
            "training_data": self.training_data,
            "acknowledged": self.acknowledged,
        }


class OnnxVSRBackend:
    """Implements the VisualSpeechBackend protocol on ONNX Runtime."""

    def __init__(
        self,
        session: Any,
        *,
        vocab: Vocabulary | None = None,
        model_id: str = "onnx-vsr",
        beam_width: int = 12,
        input_name: str | None = None,
    ):
        self._session = session
        self._vocab = vocab or default_vocabulary()
        self._model_id = model_id
        self._beam_width = beam_width
        self._input_name = input_name

    def available(self) -> tuple[bool, str]:
        return (self._session is not None), (
            "ready" if self._session is not None else "no_onnx_session")

    def _resolve_input_name(self) -> str:
        if self._input_name:
            return self._input_name
        return self._session.get_inputs()[0].name

    async def infer(self, mouth_crops: NDArray[np.uint8], *, fps: float):
        from core.perception.visual_speech import BackendPrediction

        tensor = preprocess_mouth_crops(mouth_crops)
        outputs = self._session.run(None, {self._resolve_input_name(): tensor})
        logits = np.asarray(outputs[0])
        # Accept (T, V) or (1, T, V).
        if logits.ndim == 3:
            logits = logits[0]
        if logits.ndim != 2:
            raise ValueError(f"unexpected VSR logits shape {logits.shape}")

        transcript, confidence = beam_search_decode(
            logits, self._vocab, beam_width=self._beam_width)
        greedy = greedy_decode(logits, self._vocab)
        return BackendPrediction(
            transcript=transcript.strip(),
            confidence=confidence,
            calibrated=True,
            backend="onnx-vsr-ctc-beam",
            model_id=self._model_id,
            alternatives=((greedy.strip(), confidence),) if greedy != transcript else (),
        )


def load_onnx_backend(
    model_path: str | Path,
    *,
    provenance: ModelProvenance,
    vocab: Vocabulary | None = None,
    beam_width: int = 12,
) -> OnnxVSRBackend:
    """Load an ONNX VSR model, refusing to proceed unless the operator
    has acknowledged its license/provenance. Restricted-data weights
    (e.g. LRS3-derived) are never shipped silently."""
    if not provenance.acknowledged:
        raise PermissionError(
            f"VSR model '{provenance.model_id}' carries license "
            f"'{provenance.license}' (training data: {provenance.training_data}). "
            "Set provenance.acknowledged=True to use it under those terms.")
    path = Path(model_path)
    if not path.exists() or path.suffix.lower() != ".onnx":
        raise FileNotFoundError(f"VSR model must be an existing .onnx file: {path}")
    try:
        import onnxruntime

        session = onnxruntime.InferenceSession(
            str(path), providers=["CPUExecutionProvider"])
    except Exception as exc:  # noqa: BLE001 — onnxruntime pybind error surface
        record_degradation("perception.vsr.load", exc)
        raise RuntimeError(f"failed to load VSR ONNX model: {exc}") from None
    _record_provenance(path, provenance)
    logger.info("Loaded open-vocab VSR backend: %s (%s)",
                provenance.model_id, provenance.license)
    return OnnxVSRBackend(
        session, vocab=vocab, model_id=provenance.model_id, beam_width=beam_width)


def _record_provenance(path: Path, provenance: ModelProvenance) -> None:
    """Write the acknowledged provenance beside the model for audit.

    An audit sidecar is a consequential write: it goes through the governed
    file-write gateway like every other internal maintenance artifact."""
    try:
        from core.governance_context import local_internal_governed_scope
        from core.runtime.file_write_gateway import get_file_write_gateway

        sidecar = path.with_suffix(".provenance.json")
        with local_internal_governed_scope(
            "perception.vsr.provenance", domain="file_write"
        ):
            get_file_write_gateway().write_text(
                sidecar,
                json.dumps(provenance.to_dict(), indent=2),
                source="perception.vsr.provenance",
            )
    except _VSR_ERRORS as exc:
        record_degradation("perception.vsr.provenance", exc)
