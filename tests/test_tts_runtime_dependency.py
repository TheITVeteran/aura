import importlib.util
from importlib.metadata import version
from pathlib import Path

from packaging.version import Version


def test_transformers_tts_compat_restores_removed_helper(monkeypatch):
    from transformers import pytorch_utils

    monkeypatch.delattr(pytorch_utils, "isin_mps_friendly", raising=False)

    from core.utils.transformers_tts_compat import install_transformers_tts_compat

    assert install_transformers_tts_compat() is True
    assert callable(pytorch_utils.isin_mps_friendly)


def test_voice_engine_loads_coqui_tts_on_transformers_5_lane(monkeypatch):
    import core.senses.voice_engine as voice_engine

    if importlib.util.find_spec("TTS") is None:
        return

    monkeypatch.setattr(voice_engine, "TTS", None)
    monkeypatch.setattr(voice_engine, "_tts_api_import_attempted", False)
    monkeypatch.setattr(voice_engine, "_tts_api_import_error", None)

    assert Version(version("transformers")).major >= 5
    assert voice_engine._load_tts_api() is not None
    assert voice_engine._tts_api_import_error is None


def test_voice_engine_exposes_installed_tts_backends(tmp_path):
    import core.senses.voice_engine as voice_engine

    assert importlib.util.find_spec("piper") is not None
    assert voice_engine.PiperVoice is not None

    engine = voice_engine.SovereignVoiceEngine(data_dir=str(tmp_path))
    status = engine.get_status()

    assert status["tts_available"] is True
    assert status["piper_tts_available"] is True
    assert status["pyttsx3_available"] is True


def test_default_voice_manifest_keeps_torch_backends_optional():
    root = Path(__file__).resolve().parents[1]
    default_requirements = (root / "requirements.txt").read_text(encoding="utf-8")
    optional_requirements = (root / "requirements" / "voice-high-fidelity.txt").read_text(encoding="utf-8")

    assert "coqui-tts" not in default_requirements
    assert "torchaudio" not in default_requirements
    assert "mlx-whisper" not in default_requirements
    assert "coqui-tts" in optional_requirements
    assert "torchaudio" in optional_requirements
    assert "mlx-whisper" in optional_requirements


def test_voice_output_skill_resolves_project_venv_piper(monkeypatch, tmp_path):
    import core.skills.voice_output as voice_output

    venv_bin = tmp_path / "bin"
    venv_bin.mkdir()
    fake_python = venv_bin / "python"
    fake_piper = venv_bin / "piper"
    fake_python.write_text("#!/bin/sh\n", encoding="utf-8")
    fake_piper.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    fake_python.chmod(0o755)
    fake_piper.chmod(0o755)

    calls = []

    class Gateway:
        def run(self, command, **_kwargs):
            calls.append(command)
            return type("Result", (), {"returncode": 0})()

    monkeypatch.delenv("AURA_PIPER_BIN", raising=False)
    monkeypatch.setenv("PATH", "")
    monkeypatch.setattr(voice_output.sys, "executable", str(fake_python))
    monkeypatch.setattr(voice_output, "get_subprocess_gateway", lambda: Gateway())

    skill = voice_output.VoiceOutputSkill()

    assert skill._check_piper() is True
    assert skill._piper_command == str(fake_piper)
    assert calls == [[str(fake_piper), "--help"]]
