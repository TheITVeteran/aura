from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def test_legacy_llm_clients_do_not_swallow_generic_exceptions():
    for relative in ("llm/mlx_client.py", "llm/client.py"):
        source = (PROJECT_ROOT / relative).read_text(encoding="utf-8")

        assert "except Exception" not in source
        assert "except BaseException" not in source


def test_legacy_llm_clients_record_expected_failures():
    mlx_source = (PROJECT_ROOT / "llm" / "mlx_client.py").read_text(encoding="utf-8")
    client_source = (PROJECT_ROOT / "llm" / "client.py").read_text(encoding="utf-8")

    assert "legacy_mlx_client" in mlx_source
    # The legacy shim now delegates to the canonical client; the degradation
    # action names the canonical compatibility path (81f2b64b).
    assert "failed during canonical legacy-compatibility generation" in mlx_source
    assert "legacy_sovereign_llm_client" in client_source
    assert "failed during legacy sovereign LLM call" in client_source
