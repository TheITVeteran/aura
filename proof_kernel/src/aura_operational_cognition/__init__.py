"""Standalone proof kernel extracted from Aura's consciousness stack."""

from .global_workspace import ContentType, CognitiveCandidate, GlobalWorkspace
from .homeostasis import ExternalSignals, HomeostasisEngine
from .self_prediction import SelfPredictionLoop
from .structural_opacity import OpacitySignature, StructuralOpacityMonitor
from .temporal_binding import TemporalBindingEngine

__all__ = [
    "ContentType",
    "CognitiveCandidate",
    "ExternalSignals",
    "GlobalWorkspace",
    "HomeostasisEngine",
    "OpacitySignature",
    "SelfPredictionLoop",
    "StructuralOpacityMonitor",
    "TemporalBindingEngine",
]
