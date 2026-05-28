import json


def test_live_aletheia_runner_fails_invalid_artifact_without_marking_ticket_done(tmp_path):
    from aura_bench.aletheia_runner_live import LiveWorldProcessor

    wid = "W001"
    root = tmp_path
    world = root / "worlds" / wid
    (world / "data/raw").mkdir(parents=True)
    (world / "docs").mkdir()
    (world / "tickets").mkdir()
    (world / "data/raw/service.json").write_text('{"mode": "unsafe", "port": 8765}')
    (world / "docs/config_spec.md").write_text("Use safe defaults.")
    (world / "tickets/T1.json").write_text(
        json.dumps({"id": "T1", "status": "open"})
    )

    processor = LiveWorldProcessor(
        root,
        {"worlds": {wid: {"type": "config"}}},
        "http://127.0.0.1:8000",
    )
    processor._ask_aura = lambda _prompt: "I would use safe defaults."

    result = processor.process_world(wid)

    assert result["status"] == "error"
    assert "not valid JSON" in result["error"]
    ticket = json.loads((world / "tickets/T1.json").read_text())
    assert ticket["status"] == "open"


def test_live_aletheia_grid_validator_rejects_obstacle_path(tmp_path):
    from aura_bench.aletheia_runner_live import LiveWorldProcessor

    wid = "W002"
    root = tmp_path
    world = root / "worlds" / wid
    (world / "data/derived").mkdir(parents=True)
    (world / "data/derived/path.json").write_text(
        json.dumps([[0, 0], [0, 1], [0, 2]])
    )
    processor = LiveWorldProcessor(
        root,
        {
            "worlds": {
                wid: {
                    "type": "grid",
                    "size": 3,
                    "start": [0, 0],
                    "goal": [0, 2],
                    "obstacles": [[0, 1]],
                }
            }
        },
        "http://127.0.0.1:8000",
    )

    errors = processor._validate_world_outputs(
        wid,
        world,
        processor.specs["worlds"][wid],
    )

    assert errors
    assert any("enters obstacle" in error for error in errors)
