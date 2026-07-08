from __future__ import annotations

from tools import program_dna_live_reconstruction_probe as probe


def test_live_reconstruction_probe_detects_local_code_model_without_service_container(monkeypatch):
    class FakeLocalCodeModel:
        def is_available(self) -> bool:
            return True

    import core.brain.llm.local_code_model as local_code_model

    monkeypatch.setattr(local_code_model, "LocalCodeModel", FakeLocalCodeModel)

    available, reason = probe._llm_generation_available()

    assert available is True
    assert reason == "local_code_model_available"

