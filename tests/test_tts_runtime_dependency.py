from importlib.metadata import version
import importlib.util

from packaging.version import Version


def test_transformers_tts_compat_restores_removed_helper(monkeypatch):
    from transformers import pytorch_utils

    monkeypatch.delattr(pytorch_utils, "isin_mps_friendly", raising=False)

    from core.utils.transformers_tts_compat import install_transformers_tts_compat

    assert install_transformers_tts_compat() is True
    assert callable(pytorch_utils.isin_mps_friendly)


def test_voice_engine_loads_coqui_tts_on_transformers_5_lane(monkeypatch):
    import core.senses.voice_engine as voice_engine

    monkeypatch.setattr(voice_engine, "TTS", None)
    monkeypatch.setattr(voice_engine, "_tts_api_import_attempted", False)
    monkeypatch.setattr(voice_engine, "_tts_api_import_error", None)

    assert Version(version("transformers")).major >= 5
    assert voice_engine._load_tts_api() is not None
    assert voice_engine._tts_api_import_error is None


def test_voice_engine_exposes_installed_tts_backends(tmp_path):
    import core.senses.voice_engine as voice_engine

    assert importlib.util.find_spec("TTS") is not None
    assert importlib.util.find_spec("piper") is not None
    assert voice_engine.PiperVoice is not None

    engine = voice_engine.SovereignVoiceEngine(data_dir=str(tmp_path))
    status = engine.get_status()

    assert status["tts_available"] is True
    assert status["coqui_tts_available"] is True
    assert status["piper_tts_available"] is True
    assert status["pyttsx3_available"] is True
