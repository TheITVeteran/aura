"""Evidence-based source attribution and attention policy for heard speech."""

from __future__ import annotations

import re
import time
from dataclasses import asdict, dataclass
from typing import Any

_DIRECT_ADDRESS_RE = re.compile(
    r"\b(?:(?:hey|hi|hello|okay|ok)\s+)?aura\b",
    re.IGNORECASE,
)
_QUESTION_OR_INTEREST_RE = re.compile(
    r"\b(?:why|how|what|idea|discover|research|learn|explain|wonder|news|story)\b",
    re.IGNORECASE,
)
_MEDIA_APP_MARKERS = {
    "chrome",
    "firefox",
    "music",
    "podcasts",
    "quicktime",
    "safari",
    "spotify",
    "tv",
    "vlc",
    "youtube",
}


@dataclass(frozen=True, slots=True)
class AudioAttentionAssessment:
    source: str
    confidence: float
    addressed_to_aura: bool
    response_authorized: bool
    attention_mode: str
    attention_score: float
    reasons: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def classify_audio_attention(
    text: str,
    *,
    rms_db: float,
    transcript_confidence: float,
    duration_s: float,
    active_app: str = "",
    explicit_command: bool = False,
    visual_context: dict[str, Any] | None = None,
) -> AudioAttentionAssessment:
    """Classify heard speech without treating uncertain audio as user intent.

    The classifier intentionally emits likelihoods rather than identity claims.
    Wake/session logic remains the authority boundary for conversation dispatch.
    """

    normalized = " ".join(str(text or "").split())
    app = str(active_app or "").strip().lower()
    addressed = bool(_DIRECT_ADDRESS_RE.search(normalized))
    media_context = any(marker in app for marker in _MEDIA_APP_MARKERS)
    long_narrative = duration_s >= 7.0 and len(normalized.split()) >= 8
    near_field = rms_db >= -22.0 and transcript_confidence >= -0.45
    visual = dict(visual_context or {})
    visual_updated_at = float(visual.get("updated_at", 0.0) or 0.0)
    visual_fresh = bool(
        visual_updated_at > 0.0
        and max(0.0, time.time() - visual_updated_at) <= 6.0
    )
    visible_person = bool(visual_fresh and visual.get("face_present"))
    speaking_likelihood = _clamp(float(visual.get("speaking_likelihood", 0.0) or 0.0))
    visible_speaker = bool(visible_person and speaking_likelihood >= 0.22)

    reasons: list[str] = []
    if explicit_command:
        source = "direct_user"
        confidence = 0.99
        reasons.append("explicit_voice_capture")
    elif addressed:
        source = "direct_address"
        confidence = 0.97
        reasons.append("wake_or_name_address")
    elif visible_speaker:
        source = "nearby_visible_speaker"
        confidence = max(0.72, speaking_likelihood)
        reasons.extend(("fresh_visible_face", "lower_face_motion"))
    elif media_context and long_narrative:
        source = "device_media"
        confidence = 0.84 if visual_fresh and not visible_person else 0.78
        reasons.extend(("media_app_context", "long_narrative_audio"))
    elif near_field:
        source = "nearby_person"
        confidence = 0.68
        reasons.extend(("near_field_energy", "speech_confidence"))
    elif long_narrative or rms_db < -25.0:
        source = "ambient_speech"
        confidence = 0.64
        reasons.append("ambient_or_distant_acoustics")
    else:
        source = "unknown_speech"
        confidence = 0.45
        reasons.append("insufficient_source_evidence")

    attention_score = 1.0 if explicit_command or addressed else 0.18
    if near_field:
        attention_score += 0.18
    if visible_speaker:
        attention_score += 0.22
    if _QUESTION_OR_INTEREST_RE.search(normalized):
        attention_score += 0.16
        reasons.append("semantic_interest_signal")
    if source == "device_media":
        attention_score -= 0.08
    attention_score = _clamp(attention_score)

    if addressed or explicit_command:
        attention_mode = "conversation_candidate"
    elif attention_score >= 0.48:
        attention_mode = "attend"
    elif attention_score >= 0.24:
        attention_mode = "observe"
    else:
        attention_mode = "ignore"

    return AudioAttentionAssessment(
        source=source,
        confidence=_clamp(confidence),
        addressed_to_aura=addressed,
        response_authorized=bool(explicit_command),
        attention_mode=attention_mode,
        attention_score=attention_score,
        reasons=tuple(dict.fromkeys(reasons)),
    )
