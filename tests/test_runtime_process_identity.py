from __future__ import annotations

from types import SimpleNamespace

from core.runtime.process_identity import (
    command_invokes_python_script,
    python_script_argument,
    select_script_process_tree,
)


def test_python_script_identity_rejects_shell_and_embedded_text(tmp_path) -> None:
    script = tmp_path / "aura_main.py"

    assert command_invokes_python_script(
        ["/usr/bin/python3", "-u", str(script), "--desktop"],
        expected_script=script,
    )
    assert not command_invokes_python_script(
        [
            "/bin/zsh",
            "-lc",
            f"pgrep -fl '{script.name}' | head -2",
        ],
        expected_script=script,
        cwd=tmp_path,
    )
    assert not command_invokes_python_script(
        ["/usr/bin/python3", "-c", f"print('{script.name}')"],
        expected_script=script,
        cwd=tmp_path,
    )
    assert not command_invokes_python_script(
        ["/usr/bin/python3", "-m", "pytest", script.name],
        expected_script=script,
        cwd=tmp_path,
    )


def test_python_script_identity_resolves_relative_script_from_observed_cwd(tmp_path) -> None:
    script = tmp_path / "aura_main.py"

    assert python_script_argument(["python3.12", "-B", "-u", "aura_main.py"]) == "aura_main.py"
    assert command_invokes_python_script(
        ["python3.12", "-B", "-u", "aura_main.py", "--desktop"],
        expected_script=script,
        cwd=tmp_path,
    )
    assert not command_invokes_python_script(
        ["python3.12", "aura_main.py"],
        expected_script=script,
        cwd=tmp_path / "different-repository",
    )


def test_script_process_tree_selects_exact_root_and_descendants_only(tmp_path) -> None:
    script = tmp_path / "aura_main.py"
    runtime = SimpleNamespace(
        pid=100,
        cmdline=("python3", str(script), "--desktop"),
        cwd=str(tmp_path),
        ancestor_pids=(50,),
    )
    worker = SimpleNamespace(
        pid=101,
        cmdline=("python3", "-c", "from multiprocessing.spawn import spawn_main"),
        cwd=str(tmp_path),
        ancestor_pids=(100, 50),
    )
    shell_probe = SimpleNamespace(
        pid=102,
        cmdline=("zsh", "-lc", f"ps ax | grep {script.name}"),
        cwd=str(tmp_path),
        ancestor_pids=(50,),
    )
    unrelated = SimpleNamespace(
        pid=103,
        cmdline=("python3", str(tmp_path / "other" / "aura_main.py")),
        cwd=str(tmp_path),
        ancestor_pids=(50,),
    )

    selected = select_script_process_tree(
        [shell_probe, worker, unrelated, runtime],
        expected_scripts=[script],
        protected_pids=[50],
    )

    assert [process.pid for process in selected] == [100, 101]
