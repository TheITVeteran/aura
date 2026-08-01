from __future__ import annotations

from pathlib import Path

import pytest

from tests.test_recurrent_policy_warm_start import _write_source
from tools import prepare_recurrent_policy_warm_start as prepare


def test_cli_builds_replays_and_refuses_output_rebinding(tmp_path: Path) -> None:
    complete, training, spec = _write_source(tmp_path)
    output = tmp_path / "warm_start.json"
    arguments = [
        "--repo-root",
        tmp_path.anchor,
        "build",
        "--complete",
        str(complete),
        "--training-config",
        str(training),
        "--execution-spec",
        str(spec),
        "--output",
        str(output),
    ]

    assert prepare.main(arguments) == 0
    assert prepare.main(arguments) == 0
    assert prepare.main(
        [
            "--repo-root",
            tmp_path.anchor,
            "verify",
            "--contract",
            str(output),
        ]
    ) == 0

    with pytest.raises(ValueError, match="warm_start_output_rebind_forbidden"):
        prepare.main(
            [
                *arguments,
                "--copy-targets",
                "v_proj",
            ]
        )
