import asyncio
import json
from pathlib import Path

import numpy as np

from senses.audio_service import AudioServiceConfig, run_audio_loop


async def _no_delay(_seconds: float) -> None:
    return None


def _read_snapshot(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_audio_loop_records_and_transcribes_detected_voice(tmp_path: Path) -> None:
    output_path = tmp_path / "state" / "sensory_audio.json"
    written_paths: list[Path] = []

    def recorder(frame_count: int, _sample_rate: int) -> np.ndarray:
        return np.full((frame_count, 1), 0.2, dtype=np.float32)

    def wav_writer(path: Path, _sample_rate: int, _recording: np.ndarray) -> None:
        written_paths.append(path)
        path.write_bytes(b"RIFF-aura-audio")

    async def transcriber(path: Path) -> str:
        audio_bytes = await asyncio.to_thread(path.read_bytes)
        assert audio_bytes.startswith(b"RIFF")
        return "hello aura"

    asyncio.run(
        run_audio_loop(
            config=AudioServiceConfig(
                sample_rate=1_000,
                chunk_seconds=0.01,
                voice_threshold=0.05,
                output_path=output_path,
            ),
            max_iterations=1,
            recorder=recorder,
            wav_writer=wav_writer,
            transcriber=transcriber,
            sleep=_no_delay,
        )
    )

    snapshot = _read_snapshot(output_path)
    assert snapshot["status"] == "active"
    assert snapshot["transcript"] == "hello aura"
    assert snapshot["rms"] > 0.05
    assert written_paths
    assert all(not path.exists() for path in written_paths)


def test_audio_loop_reports_unavailable_transcription(tmp_path: Path) -> None:
    output_path = tmp_path / "sensory_audio.json"

    def recorder(frame_count: int, _sample_rate: int) -> np.ndarray:
        return np.full((frame_count, 1), 0.3, dtype=np.float32)

    def wav_writer(path: Path, _sample_rate: int, _recording: np.ndarray) -> None:
        path.write_bytes(b"RIFF-aura-audio")

    asyncio.run(
        run_audio_loop(
            config=AudioServiceConfig(
                sample_rate=1_000,
                chunk_seconds=0.01,
                voice_threshold=0.05,
                output_path=output_path,
            ),
            max_iterations=1,
            recorder=recorder,
            wav_writer=wav_writer,
            transcriber=None,
            sleep=_no_delay,
        )
    )

    snapshot = _read_snapshot(output_path)
    assert snapshot["status"] == "audio_detected_transcription_unavailable"
    assert snapshot["transcript"] == ""
    assert snapshot["rms"] > 0.05


def test_audio_loop_writes_listening_state_for_quiet_input(tmp_path: Path) -> None:
    output_path = tmp_path / "sensory_audio.json"

    def recorder(frame_count: int, _sample_rate: int) -> np.ndarray:
        return np.zeros((frame_count, 1), dtype=np.float32)

    def wav_writer(path: Path, _sample_rate: int, _recording: np.ndarray) -> None:
        path.write_bytes(b"RIFF-aura-audio")

    asyncio.run(
        run_audio_loop(
            config=AudioServiceConfig(
                sample_rate=1_000,
                chunk_seconds=0.01,
                voice_threshold=0.05,
                output_path=output_path,
            ),
            max_iterations=1,
            recorder=recorder,
            wav_writer=wav_writer,
            transcriber=None,
            sleep=_no_delay,
        )
    )

    snapshot = _read_snapshot(output_path)
    assert snapshot["status"] == "listening"
    assert snapshot["transcript"] == ""
    assert snapshot["rms"] == 0.0


def test_audio_loop_stops_after_error_limit(tmp_path: Path) -> None:
    output_path = tmp_path / "sensory_audio.json"
    capture_attempts: list[tuple[int, int]] = []

    def recorder(_frame_count: int, _sample_rate: int) -> np.ndarray:
        capture_attempts.append((_frame_count, _sample_rate))
        raise RuntimeError("capture failed")

    def wav_writer(path: Path, _sample_rate: int, _recording: np.ndarray) -> None:
        path.write_bytes(b"RIFF-aura-audio")

    asyncio.run(
        run_audio_loop(
            config=AudioServiceConfig(
                sample_rate=1_000,
                chunk_seconds=0.01,
                max_consecutive_errors=1,
                output_path=output_path,
            ),
            recorder=recorder,
            wav_writer=wav_writer,
            transcriber=None,
            sleep=_no_delay,
        )
    )

    snapshot = _read_snapshot(output_path)
    assert snapshot["status"] == "audio_error_limit_reached"
    assert snapshot["detail"] == "capture failed"
    assert capture_attempts == [(10, 1_000)]


def test_audio_loop_persists_invalid_configuration(tmp_path: Path) -> None:
    output_path = tmp_path / "sensory_audio.json"

    asyncio.run(
        run_audio_loop(
            config=AudioServiceConfig(
                sample_rate=0,
                chunk_seconds=0.0,
                max_consecutive_errors=0,
                output_path=output_path,
            ),
            recorder=lambda _frames, _rate: np.zeros((1, 1), dtype=np.float32),
            wav_writer=lambda path, _rate, _recording: path.write_bytes(b"RIFF-aura-audio"),
            transcriber=None,
            sleep=_no_delay,
        )
    )

    snapshot = _read_snapshot(output_path)
    assert snapshot["status"] == "audio_configuration_invalid"
    assert "sample_rate must be positive" in snapshot["detail"]
    assert "chunk_seconds must be positive" in snapshot["detail"]
    assert "max_consecutive_errors must be positive" in snapshot["detail"]
