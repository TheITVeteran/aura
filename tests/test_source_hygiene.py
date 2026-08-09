from tools.check_source_hygiene import collect_violations, hygiene_violation


def test_closeout_source_snapshots_are_rejected():
    path = "artifacts/closeout/latent_cortex/run/source_snapshots/trainer.py"

    assert hygiene_violation(path) == "duplicated_source_snapshot"


def test_real_source_and_authored_closeout_probes_are_allowed():
    assert hygiene_violation("core/brain/llm/mlx_client.py") is None
    assert (
        hygiene_violation(
            "artifacts/closeout/latent_cortex/cp210/trajectory_probe.py"
        )
        is None
    )


def test_cache_detection_preserves_all_reasons():
    assert collect_violations(
        ["core/__pycache__/module.pyc", "state.sqlite3", "tests/test_real.py"]
    ) == [
        ("core/__pycache__/module.pyc", "generated_cache"),
        ("state.sqlite3", "generated_cache"),
    ]
