"""Native face, mouth, and event-time extraction for visual speech.

OpenCV owns bounded video decoding so Aura never loads two competing FFmpeg
runtimes into the macOS process. Native AVFoundation metadata detects audio
streams without decoding them. The primary detector uses macOS Vision face
landmarks entirely in memory; OpenCV's approximate mouth ROI is an availability
fallback and is never represented as a real lip landmark.
"""
from __future__ import annotations

import hashlib
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray

from core.perception.visual_speech import VisualSpeechEvidence, VisualSpeechPolicy

BoundingBox = tuple[float, float, float, float]


def _iou(left: BoundingBox, right: BoundingBox) -> float:
    lx, ly, lw, lh = left
    rx, ry, rw, rh = right
    x1 = max(lx, rx)
    y1 = max(ly, ry)
    x2 = min(lx + lw, rx + rw)
    y2 = min(ly + lh, ry + rh)
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = lw * lh + rw * rh - intersection
    return intersection / union if union > 0.0 else 0.0


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _probe_audio_presence(video_path: Path) -> tuple[bool, bool, str]:
    """Return present, known, and a bounded reason without decoding audio."""
    if sys.platform != "darwin":
        return False, False, "native_media_metadata_unavailable"
    try:
        from AVFoundation import AVMediaTypeAudio, AVURLAsset
        from Foundation import NSURL

        url = NSURL.fileURLWithPath_(str(video_path))
        asset = AVURLAsset.URLAssetWithURL_options_(url, None)
        tracks = asset.tracksWithMediaType_(AVMediaTypeAudio)
        return bool(len(tracks)), True, "macos_avfoundation"
    except (ImportError, OSError, RuntimeError, TypeError, ValueError) as exc:
        return False, False, f"native_media_metadata_error_{type(exc).__name__}"[:160]


@dataclass(frozen=True)
class MouthDetection:
    face_count: int
    selected_bbox: BoundingBox | None
    crop: NDArray[np.uint8] | None
    landmarks_present: bool
    competing_face_ratio: float
    detector: str


class MouthDetector(Protocol):
    name: str

    def detect(
        self,
        frame_rgb: NDArray[np.uint8],
        previous_bbox: BoundingBox | None,
    ) -> MouthDetection: ...


def _select_face(
    boxes: list[BoundingBox],
    previous_bbox: BoundingBox | None,
) -> tuple[int, float]:
    if not boxes:
        raise ValueError("cannot select a face from an empty set")
    areas = [box[2] * box[3] for box in boxes]
    largest_index = max(range(len(boxes)), key=lambda index: areas[index])
    selected_index = largest_index
    if previous_bbox is not None:
        overlaps = [_iou(box, previous_bbox) for box in boxes]
        overlap_index = max(range(len(boxes)), key=lambda index: overlaps[index])
        if overlaps[overlap_index] >= 0.12:
            selected_index = overlap_index
    selected_area = max(1e-9, areas[selected_index])
    competitor = max(
        (area / selected_area for index, area in enumerate(areas) if index != selected_index),
        default=0.0,
    )
    return selected_index, max(0.0, min(1.0, competitor))


def _crop_square(
    frame_rgb: NDArray[np.uint8],
    *,
    center_x: float,
    center_y: float,
    size: float,
    output_size: int = 96,
) -> NDArray[np.uint8] | None:
    import cv2

    height, width = frame_rgb.shape[:2]
    half = max(8.0, size / 2.0)
    x1 = max(0, int(round(center_x - half)))
    y1 = max(0, int(round(center_y - half)))
    x2 = min(width, int(round(center_x + half)))
    y2 = min(height, int(round(center_y + half)))
    if x2 - x1 < 16 or y2 - y1 < 16:
        return None
    crop = frame_rgb[y1:y2, x1:x2]
    resized = cv2.resize(crop, (output_size, output_size), interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray(resized, dtype=np.uint8)


class MacOSVisionMouthDetector:
    """In-memory native face and outer-lip landmark detector."""

    name = "macos_vision_face_landmarks"

    def __init__(self) -> None:
        if sys.platform != "darwin":
            raise RuntimeError("macOS Vision mouth tracking is available only on Darwin")
        try:
            import Vision  # noqa: F401
            from Foundation import NSData  # noqa: F401
            from Quartz import CGImageSourceCreateWithData  # noqa: F401
        except ImportError as exc:
            raise RuntimeError("macOS Vision face landmarks are unavailable") from exc

    def detect(
        self,
        frame_rgb: NDArray[np.uint8],
        previous_bbox: BoundingBox | None,
    ) -> MouthDetection:
        import cv2
        import Vision
        from Foundation import NSData
        from Quartz import CGImageSourceCreateImageAtIndex, CGImageSourceCreateWithData

        bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        ok, encoded = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
        if not ok:
            raise RuntimeError("could not encode frame for native Vision")
        payload = encoded.tobytes()
        data = NSData.dataWithBytes_length_(payload, len(payload))
        image_source = CGImageSourceCreateWithData(data, None)
        if image_source is None:
            raise RuntimeError("native Vision could not construct image source")
        image = CGImageSourceCreateImageAtIndex(image_source, 0, None)
        if image is None:
            raise RuntimeError("native Vision could not decode frame")
        request = Vision.VNDetectFaceLandmarksRequest.alloc().init()
        handler = Vision.VNImageRequestHandler.alloc().initWithCGImage_options_(image, {})
        succeeded, error = handler.performRequests_error_([request], None)
        if not succeeded:
            raise RuntimeError(f"native Vision face request failed: {error}")
        observations = list(request.results() or [])
        if not observations:
            return MouthDetection(0, None, None, False, 0.0, self.name)

        boxes: list[BoundingBox] = []
        for observation in observations:
            box = observation.boundingBox()
            boxes.append(
                (
                    float(box.origin.x),
                    float(box.origin.y),
                    float(box.size.width),
                    float(box.size.height),
                )
            )
        selected_index, competitor = _select_face(boxes, previous_bbox)
        selected = observations[selected_index]
        selected_bbox = boxes[selected_index]
        landmarks = selected.landmarks()
        lips = landmarks.outerLips() if landmarks is not None else None
        if lips is None or int(lips.pointCount()) < 6:
            return MouthDetection(
                len(observations),
                selected_bbox,
                None,
                False,
                competitor,
                self.name,
            )

        points = lips.normalizedPoints()
        point_count = int(lips.pointCount())
        face_x, face_y, face_w, face_h = selected_bbox
        full_points = [
            (
                face_x + float(points[index].x) * face_w,
                face_y + float(points[index].y) * face_h,
            )
            for index in range(point_count)
        ]
        height, width = frame_rgb.shape[:2]
        pixel_x = [point[0] * width for point in full_points]
        pixel_y = [(1.0 - point[1]) * height for point in full_points]
        lip_width = max(pixel_x) - min(pixel_x)
        lip_height = max(pixel_y) - min(pixel_y)
        face_pixel_width = face_w * width
        crop = _crop_square(
            frame_rgb,
            center_x=float(np.mean(pixel_x)),
            center_y=float(np.mean(pixel_y)),
            size=max(lip_width * 1.9, lip_height * 3.2, face_pixel_width * 0.42),
        )
        return MouthDetection(
            len(observations),
            selected_bbox,
            crop,
            crop is not None,
            competitor,
            self.name,
        )


class OpenCVFaceMouthDetector:
    """Availability fallback; approximate ROI is never called a lip landmark."""

    name = "opencv_face_approximate_mouth"

    def __init__(self) -> None:
        import cv2

        self._cv2: Any = cv2
        self._cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades  # type: ignore[attr-defined]
            + "haarcascade_frontalface_default.xml"
        )
        if self._cascade.empty():
            raise RuntimeError("OpenCV face cascade failed to load")

    def detect(
        self,
        frame_rgb: NDArray[np.uint8],
        previous_bbox: BoundingBox | None,
    ) -> MouthDetection:
        height, width = frame_rgb.shape[:2]
        gray = self._cv2.cvtColor(frame_rgb, self._cv2.COLOR_RGB2GRAY)
        faces = self._cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(32, 32),
        )
        boxes: list[BoundingBox] = [
            (float(x) / width, 1.0 - float(y + h) / height, float(w) / width, float(h) / height)
            for x, y, w, h in faces
        ]
        if not boxes:
            return MouthDetection(0, None, None, False, 0.0, self.name)
        selected_index, competitor = _select_face(boxes, previous_bbox)
        x, y_bottom, w, h = boxes[selected_index]
        center_x = (x + 0.5 * w) * width
        center_y = (1.0 - (y_bottom + 0.28 * h)) * height
        crop = _crop_square(
            frame_rgb,
            center_x=center_x,
            center_y=center_y,
            size=w * width * 0.55,
        )
        return MouthDetection(
            len(boxes),
            boxes[selected_index],
            crop,
            False,
            competitor,
            self.name,
        )


def default_mouth_detector() -> MouthDetector:
    try:
        return MacOSVisionMouthDetector()
    except (ImportError, RuntimeError):
        return OpenCVFaceMouthDetector()


def _interpolate_crops(
    crops: list[NDArray[np.uint8] | None],
) -> NDArray[np.uint8]:
    valid_indices = [index for index, crop in enumerate(crops) if crop is not None]
    if not valid_indices:
        return np.empty((0, 96, 96, 3), dtype=np.uint8)
    output: NDArray[np.uint8] = np.empty(
        (len(crops), 96, 96, 3),
        dtype=np.uint8,
    )
    first = valid_indices[0]
    last = valid_indices[-1]
    first_crop = crops[first]
    last_crop = crops[last]
    if first_crop is None or last_crop is None:
        raise RuntimeError("valid crop index unexpectedly contains no crop")
    output[: first + 1] = first_crop
    output[last:] = last_crop
    for left_index, right_index in zip(valid_indices, valid_indices[1:], strict=False):
        left_crop = crops[left_index]
        right_crop = crops[right_index]
        if left_crop is None or right_crop is None:
            raise RuntimeError("valid crop interpolation endpoint is missing")
        output[left_index] = left_crop
        gap = right_index - left_index
        for offset in range(1, gap):
            weight = offset / gap
            blended = (
                (1.0 - weight) * left_crop.astype(np.float32)
                + weight * right_crop.astype(np.float32)
            )
            output[left_index + offset] = np.clip(blended, 0, 255).astype(np.uint8)
        output[right_index] = right_crop
    return output


class NativeVisualSpeechExtractor:
    """Decode bounded video frames and extract a stable mouth sequence."""

    def __init__(self, detector: MouthDetector | None = None) -> None:
        self.detector = detector or default_mouth_detector()

    def extract(self, video_path: Path, policy: VisualSpeechPolicy) -> VisualSpeechEvidence:
        import cv2

        source_digest = _sha256_file(video_path)
        source_audio_present, source_audio_presence_known, audio_probe = (
            _probe_audio_presence(video_path)
        )
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise ValueError("video source could not be opened")
        try:
            source_fps = float(capture.get(cv2.CAP_PROP_FPS))
            if not math.isfinite(source_fps) or source_fps <= 0.0:
                source_fps = policy.target_fps
            if source_fps > 240.0:
                raise ValueError("video source frame rate exceeds decoder bounds")
            reported_frame_count = float(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            stream_duration = (
                reported_frame_count / source_fps
                if math.isfinite(reported_frame_count) and reported_frame_count > 0.0
                else 0.0
            )

            timestamps: list[float] = []
            raw_crops: list[NDArray[np.uint8] | None] = []
            face_frames = 0
            landmark_frames = 0
            ambiguous_frames = 0
            max_competing_ratio = 0.0
            track_switches = 0
            previous_bbox: BoundingBox | None = None
            initial_bbox: BoundingBox | None = None
            next_sample_s = 0.0
            sampled_frames = 0
            decoded_source_frames = 0
            detector_names: set[str] = set()
            truncated = False
            max_source_decode_frames = (
                math.ceil(
                    policy.max_frames
                    * max(source_fps, policy.target_fps)
                    / policy.target_fps
                )
                + 2
            )

            for _source_index in range(max_source_decode_frames):
                decoded, frame_bgr = capture.read()
                if not decoded or frame_bgr is None:
                    break
                decoded_source_frames += 1
                reported_timestamp_s = float(capture.get(cv2.CAP_PROP_POS_MSEC)) / 1000.0
                fallback_timestamp_s = (decoded_source_frames - 1) / source_fps
                timestamp_s = (
                    reported_timestamp_s
                    if math.isfinite(reported_timestamp_s)
                    and reported_timestamp_s >= fallback_timestamp_s - (0.5 / source_fps)
                    else fallback_timestamp_s
                )
                if timestamp_s + 1e-6 < next_sample_s:
                    continue
                next_sample_s += 1.0 / policy.target_fps
                if sampled_frames >= policy.max_frames:
                    truncated = True
                    break
                frame_rgb = np.ascontiguousarray(
                    cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB),
                    dtype=np.uint8,
                )
                detection = self.detector.detect(frame_rgb, previous_bbox)
                detector_names.add(detection.detector)
                sampled_frames += 1
                timestamps.append(timestamp_s)
                raw_crops.append(detection.crop)
                if detection.face_count > 0:
                    face_frames += 1
                if detection.landmarks_present and detection.crop is not None:
                    landmark_frames += 1
                max_competing_ratio = max(max_competing_ratio, detection.competing_face_ratio)
                if detection.competing_face_ratio > policy.max_competing_face_ratio:
                    ambiguous_frames += 1
                if detection.selected_bbox is not None:
                    if previous_bbox is not None and _iou(previous_bbox, detection.selected_bbox) < 0.12:
                        track_switches += 1
                    previous_bbox = detection.selected_bbox
                    if initial_bbox is None:
                        initial_bbox = detection.selected_bbox
            else:
                truncated = True

            if sampled_frames == 0:
                raise ValueError("video decoder returned no frames")
            interpolated = _interpolate_crops(raw_crops)
            if interpolated.shape[0] > 0:
                gray_crops = np.stack(
                    [cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY) for crop in interpolated],
                    axis=0,
                )
                brightness_values = gray_crops.mean(axis=(1, 2))
                blur_values = np.asarray(
                    [cv2.Laplacian(crop, cv2.CV_64F).var() for crop in gray_crops],
                    dtype=np.float64,
                )
                motion_values = np.zeros(interpolated.shape[0], dtype=np.float64)
                if interpolated.shape[0] > 1:
                    motion_values[1:] = np.mean(
                        np.abs(np.diff(gray_crops.astype(np.float32), axis=0)),
                        axis=(1, 2),
                    )
                normalized_activity = np.clip(motion_values / 32.0, 0.0, 1.0)
                evidence_timestamps = tuple(timestamps)
                mean_brightness = float(np.mean(brightness_values))
                mean_blur = float(np.mean(blur_values))
                mean_motion = float(np.mean(motion_values[1:])) if len(motion_values) > 1 else 0.0
            else:
                normalized_activity = np.empty((0,), dtype=np.float64)
                evidence_timestamps = ()
                mean_brightness = 0.0
                mean_blur = 0.0
                mean_motion = 0.0

            last_timestamp = timestamps[-1] if timestamps else 0.0
            observed_duration = max(0.0, last_timestamp - timestamps[0]) if timestamps else 0.0
            duration_s = max(stream_duration, observed_duration)
            track_seed = f"{source_digest}:{initial_bbox!r}".encode()
            speaker_track_id = "visual-track-" + hashlib.sha256(track_seed).hexdigest()[:16]
            quality_flags = [
                "video_stream_only_decoded",
                "raw_frames_not_retained",
                "mouth_crops_ephemeral",
                *sorted(detector_names),
            ]
            if source_audio_present:
                quality_flags.append("source_audio_present_not_decoded")
            elif not source_audio_presence_known:
                quality_flags.append(f"source_audio_presence_unknown:{audio_probe}"[:160])
            if truncated:
                quality_flags.append("source_truncated_at_frame_budget")
            if decoded_source_frames > sampled_frames:
                quality_flags.append("source_resampled_to_target_fps")

            return VisualSpeechEvidence(
                source_digest=source_digest,
                mouth_crops=interpolated,
                timestamps_s=evidence_timestamps,
                mouth_activity=tuple(float(value) for value in normalized_activity),
                source_fps=source_fps,
                sampled_fps=policy.target_fps,
                duration_s=duration_s,
                decoded_frames=sampled_frames,
                mouth_frames=int(interpolated.shape[0]),
                face_detection_coverage=face_frames / sampled_frames,
                mouth_landmark_coverage=landmark_frames / sampled_frames,
                mean_brightness=mean_brightness,
                mean_blur_variance=mean_blur,
                mean_mouth_motion=mean_motion,
                competing_face_ratio=max_competing_ratio,
                ambiguous_face_frames=ambiguous_frames,
                track_switches=track_switches,
                speaker_track_id=speaker_track_id,
                source_audio_present=source_audio_present,
                source_audio_presence_known=source_audio_presence_known,
                extractor="+".join(sorted(detector_names)) or self.detector.name,
                quality_flags=tuple(quality_flags[:16]),
            )
        finally:
            capture.release()


__all__ = [
    "BoundingBox",
    "MacOSVisionMouthDetector",
    "MouthDetection",
    "MouthDetector",
    "NativeVisualSpeechExtractor",
    "OpenCVFaceMouthDetector",
    "_probe_audio_presence",
    "default_mouth_detector",
]
