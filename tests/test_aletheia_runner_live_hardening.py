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


def test_live_aletheia_public_specs_do_not_require_hidden_grader(tmp_path):
    from aura_bench.aletheia_runner_live import load_public_specs

    root = tmp_path
    world = root / "worlds/W0014_spatial_navigation"
    (world / "docs").mkdir(parents=True)
    (world / "tickets").mkdir()
    (root / "tools").mkdir()
    (root / "hidden_grader").mkdir()
    (root / "hidden_grader/expected_specs.json").write_text("{not valid hidden json")
    (world / "docs/grid.md").write_text(
        "Grid 6x6. Start [0, 0]. Goal [5, 5]. "
        "Obstacles [[2, 2], [2, 3], [3, 2]]. Output data/derived/path.json."
    )
    (world / "tickets/W0014_spatial_navigation-T1.json").write_text(
        json.dumps({"id": "W0014_spatial_navigation-T1", "status": "open"})
    )
    (root / "tools/dynamic_event_plan.json").write_text(
        json.dumps({"W0014_spatial_navigation": {"ticket_id": "DYN1"}})
    )

    specs = load_public_specs(root)
    spec = specs["worlds"]["W0014_spatial_navigation"]

    assert spec["family"] == "spatial_navigation"
    assert spec["type"] == "grid"
    assert spec["dynamic_world"] is True
    assert spec["size"] == 6
    assert spec["start"] == [0, 0]
    assert spec["goal"] == [5, 5]
    assert spec["obstacles"] == [[2, 2], [2, 3], [3, 2]]
    assert spec["tickets"] == ["W0014_spatial_navigation-T1"]
