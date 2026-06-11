from __future__ import annotations

import json
from pathlib import Path

import core.self_improvement.blinded_workspace as blinded_workspace_mod
from core.self_improvement.blinded_workspace import BlindedWorkspaceFactory
from core.self_improvement.interface_contract import (
    FunctionSignature,
    InterfaceContract,
    ModuleSpec,
    TestCase,
)


def test_blinded_workspace_writes_artifacts_through_file_gateway(tmp_path, monkeypatch):
    project_root = tmp_path / "project"
    source_tests = project_root / "tests"
    source_tests.mkdir(parents=True)
    (source_tests / "test_visible.py").write_text(
        "def test_visible():\n    assert True\n",
        encoding="utf-8",
    )
    calls: list[tuple[Path, str]] = []

    class FakeFileWriteGateway:
        def write_text(self, path, text, *, encoding="utf-8", source="unknown"):
            target = Path(path)
            calls.append((target, source))
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(text, encoding=encoding)

        def write_bytes(self, path, payload, *, source="unknown"):
            target = Path(path)
            calls.append((target, source))
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(payload)

    monkeypatch.setattr(
        blinded_workspace_mod,
        "get_file_write_gateway",
        lambda: FakeFileWriteGateway(),
    )

    spec = ModuleSpec(
        module_path="pkg/module.py",
        module_name="module",
        interface=InterfaceContract(
            module_path="pkg/module.py",
            functions=[
                FunctionSignature(
                    name="add",
                    parameters=("a: int", "b: int"),
                    return_annotation="int",
                    docstring="Add two numbers.",
                )
            ],
            all_names=frozenset({"add"}),
        ),
        module_docstring="Visible public contract.",
        test_cases=[
            TestCase(
                name="test_visible",
                source="",
                file_path="tests/test_visible.py",
            )
        ],
    )

    workspace = BlindedWorkspaceFactory(project_root=str(project_root)).create(
        spec,
        "pkg/module.py",
    )
    try:
        sources = {source for _path, source in calls}
        assert {
            "core.self_improvement.blinded_workspace.interface_stub",
            "core.self_improvement.blinded_workspace.package_init",
            "core.self_improvement.blinded_workspace.tests_init",
            "core.self_improvement.blinded_workspace.copied_test",
            "core.self_improvement.blinded_workspace.spec_reference",
            "core.self_improvement.blinded_workspace.audit_manifest",
        }.issubset(sources)
        assert (workspace.workspace_dir / "tests" / "test_visible.py").read_text(
            encoding="utf-8"
        ) == "def test_visible():\n    assert True\n"
        manifest = json.loads(workspace.audit_manifest_path.read_text(encoding="utf-8"))
        assert manifest["schema"] == "aura.blinded_workspace.audit_manifest.v1"
        assert manifest["module_path"] == "pkg/module.py"
        assert manifest["forbidden_path_count"] == 2
        assert len(manifest["forbidden_path_hashes"]) == 2
        assert manifest["generated_interface"]["path"] == "pkg/module.py"
        assert manifest["generated_interface"]["sha256"]
        assert manifest["copied_tests"][0]["source_name"] == "test_visible.py"
        assert manifest["copied_tests"][0]["sha256"]
        assert "original implementation" not in json.dumps(manifest).lower()
    finally:
        workspace.cleanup()
