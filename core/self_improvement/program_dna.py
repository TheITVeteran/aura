"""Program DNA reconstruction engine.

This module generalizes Aura's internal clean-room reimplementation lab to
authorized external programs. It does not steal or decompile proprietary code.
It builds a lawful behavioral "DNA" profile from available sources:

* open-source or user-owned source trees
* app/package metadata
* observable UI and user-provided behavior notes
* research notes and comparable-program hints

The output is a reconstruction blueprint and optional clean-room scaffold that
Aura can use for implementation, testing, or further self-improvement.
"""
from __future__ import annotations

import ast
import hashlib
import importlib
import json
import os
import plistlib
import re
import shutil
import tomllib
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


AUTHORIZED_SCOPES = frozenset(
    {
        "open_source",
        "owner_authorized",
        "explicit_permission",
        "internal",
        "educational",
        "user_owned",
    }
)

PROHIBITED_MARKERS = (
    "bypass drm",
    "crack license",
    "steal source",
    "exfiltrate",
    "credential",
    "pirate",
    "malware",
    "keygen",
    "activation bypass",
)

SOURCE_EXTENSIONS = {
    ".py",
    ".js",
    ".jsx",
    ".ts",
    ".tsx",
    ".swift",
    ".java",
    ".kt",
    ".go",
    ".rs",
    ".c",
    ".cpp",
    ".h",
    ".hpp",
    ".cs",
    ".rb",
    ".php",
    ".html",
    ".css",
}

MANIFEST_NAMES = {
    "pyproject.toml",
    "package.json",
    "Cargo.toml",
    "go.mod",
    "pom.xml",
    "build.gradle",
    "Info.plist",
}

SKIP_DIRS = {
    ".git",
    ".hg",
    ".svn",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "node_modules",
    "dist",
    "build",
    ".venv",
    "venv",
}


@dataclass(slots=True)
class ProgramDNAEvidence:
    kind: str
    source: str
    summary: str
    confidence: float
    details: dict[str, Any] = field(default_factory=dict)
    sha256: str | None = None


@dataclass(slots=True)
class ProgramDNAFeature:
    name: str
    category: str
    confidence: float
    evidence_sources: list[str] = field(default_factory=list)
    rationale: str = ""


@dataclass(slots=True)
class ProgramDNABlueprint:
    target_name: str
    reconstruction_strategy: str
    components: list[dict[str, Any]]
    ux_flows: list[dict[str, Any]]
    data_models: list[dict[str, Any]]
    integrations: list[dict[str, Any]]
    test_plan: list[dict[str, Any]]
    unknowns: list[str]
    safety_boundary: list[str]


@dataclass(slots=True)
class ProgramDNAGenome:
    purpose: str
    phenotype_sources: list[str]
    feature_map: list[dict[str, Any]]
    workflow_graph: list[dict[str, Any]]
    state_machines: list[dict[str, Any]]
    data_contracts: list[dict[str, Any]]
    file_formats: list[dict[str, Any]]
    api_surface: list[dict[str, Any]]
    permission_model: list[dict[str, Any]]
    error_behaviors: list[dict[str, Any]]
    background_services: list[dict[str, Any]]
    compatibility_targets: list[str]
    hidden_state_risks: list[str]
    reconstruction_unknowns: list[str]


@dataclass(slots=True)
class ProgramDNAVerificationPlan:
    black_box_tests: list[dict[str, Any]]
    ui_tests: list[dict[str, Any]]
    golden_file_tests: list[dict[str, Any]]
    api_tests: list[dict[str, Any]]
    edge_case_tests: list[dict[str, Any]]
    performance_checks: list[dict[str, Any]]
    security_checks: list[dict[str, Any]]
    compatibility_checks: list[dict[str, Any]]
    scaffold_syntax_ok: bool | None = None
    scaffold_files: list[str] = field(default_factory=list)


@dataclass(slots=True)
class ProgramDNAResult:
    ok: bool
    target_name: str
    authorization: str
    evidence: list[ProgramDNAEvidence]
    features: list[ProgramDNAFeature]
    genome: ProgramDNAGenome | None = None
    blueprint: ProgramDNABlueprint | None = None
    verification_plan: ProgramDNAVerificationPlan | None = None
    scaffold_path: str | None = None
    blocked_reasons: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ProgramDNAReconstructionEngine:
    """Authorized clean-room reconstruction from program behavior and evidence."""

    def __init__(
        self,
        *,
        project_root: str | os.PathLike[str] | None = None,
        internal_lab: Any | None = None,
    ) -> None:
        self.project_root = Path(project_root or ".").resolve()
        self.internal_lab = internal_lab

    async def run_reconstruction(self, module_path: str, **kwargs: Any) -> Any:
        """Compatibility hook for SelfHealing deep repair.

        When this engine is used as the program-DNA service, internal Aura module
        repairs still delegate to the deterministic ReimplementationLab.
        """

        service_registry = importlib.import_module("core.runtime.service_registry")
        service_names = importlib.import_module("core.service_names")
        service_names_cls = service_names.ServiceNames

        lab = self.internal_lab or service_registry.get_runtime_service(
            service_names_cls.REIMPLEMENTATION_LAB,
            default=None,
        )
        if lab is None:
            lab_module = importlib.import_module("core.self_improvement.reimplementation_lab")

            lab = lab_module.register_reimplementation_lab()
            self.internal_lab = lab
        return await lab.run_reconstruction(module_path, **kwargs)

    async def reconstruct(self, request: dict[str, Any] | None = None, **kwargs: Any) -> ProgramDNAResult:
        payload = dict(request or {})
        payload.update(kwargs)

        target = str(payload.get("target") or payload.get("name") or "unknown_program").strip()
        authorization = str(payload.get("authorization") or "unspecified").strip().lower()
        objective = str(payload.get("objective") or target).lower()
        blocked = self._policy_blocks(authorization, objective)
        if blocked:
            return ProgramDNAResult(
                ok=False,
                target_name=target,
                authorization=authorization,
                evidence=[],
                features=[],
                blocked_reasons=blocked,
            )

        source_paths = [str(p) for p in payload.get("source_paths") or payload.get("paths") or [] if str(p).strip()]
        observed_behaviors = self._string_list(payload.get("observed_behaviors") or payload.get("behaviors"))
        ui_notes = self._string_list(payload.get("ui_notes") or payload.get("ui"))
        research_notes = self._string_list(payload.get("research_notes") or payload.get("research"))
        similar_programs = self._string_list(payload.get("similar_programs") or payload.get("analogs"))
        api_observations = self._string_list(payload.get("api_observations") or payload.get("apis"))
        file_format_notes = self._string_list(payload.get("file_formats") or payload.get("file_format_notes"))
        log_notes = self._string_list(payload.get("logs") or payload.get("log_notes"))
        test_notes = self._string_list(payload.get("tests") or payload.get("test_notes"))
        workflow_notes = self._string_list(payload.get("workflow_notes") or payload.get("workflows"))
        permission_notes = self._string_list(payload.get("permissions") or payload.get("permission_notes"))
        compatibility_targets = self._string_list(
            payload.get("compatibility_targets") or payload.get("platforms") or ["local-first replacement"]
        )

        evidence: list[ProgramDNAEvidence] = []
        warnings: list[str] = []
        for raw_path in source_paths:
            try:
                evidence.extend(self._inspect_path(Path(raw_path).expanduser()))
            except (OSError, UnicodeDecodeError, SyntaxError, ValueError, TypeError) as exc:
                warnings.append(f"could_not_inspect:{raw_path}:{type(exc).__name__}")
                self._record_degradation("program_dna_reconstruction", exc, severity="debug")

        evidence.extend(self._notes_to_evidence("observed_behavior", observed_behaviors, confidence=0.72))
        evidence.extend(self._notes_to_evidence("ui_affordance", ui_notes, confidence=0.70))
        evidence.extend(self._notes_to_evidence("research_note", research_notes, confidence=0.62))
        evidence.extend(self._notes_to_evidence("similar_program", similar_programs, confidence=0.50))
        evidence.extend(self._notes_to_evidence("api_observation", api_observations, confidence=0.68))
        evidence.extend(self._notes_to_evidence("file_format", file_format_notes, confidence=0.70))
        evidence.extend(self._notes_to_evidence("log_trace", log_notes, confidence=0.64))
        evidence.extend(self._notes_to_evidence("test_observation", test_notes, confidence=0.76))
        evidence.extend(self._notes_to_evidence("workflow_observation", workflow_notes, confidence=0.74))
        evidence.extend(self._notes_to_evidence("permission_observation", permission_notes, confidence=0.72))

        if bool(payload.get("enable_binary_static_analysis", False)):
            evidence.extend(self._binary_static_analysis_plan(source_paths))

        features = self._infer_features(evidence)
        genome = self._extract_genome(
            target_name=target,
            evidence=evidence,
            features=features,
            compatibility_targets=compatibility_targets,
        )
        blueprint = self._build_blueprint(target, evidence, features)
        verification_plan = self._build_verification_plan(features, evidence, genome)

        scaffold_path = None
        if bool(payload.get("emit_scaffold", False)):
            output_dir = payload.get("output_dir")
            scaffold_path = self._emit_scaffold(
                target_name=target,
                blueprint=blueprint,
                genome=genome,
                verification_plan=verification_plan,
                features=features,
                output_dir=Path(output_dir).expanduser() if output_dir else None,
                stack=str(payload.get("target_stack") or "python").strip().lower(),
            )
            self._verify_scaffold(Path(scaffold_path), verification_plan)

        return ProgramDNAResult(
            ok=True,
            target_name=target,
            authorization=authorization,
            evidence=evidence,
            features=features,
            genome=genome,
            blueprint=blueprint,
            verification_plan=verification_plan,
            scaffold_path=scaffold_path,
            warnings=warnings,
        )

    def _policy_blocks(self, authorization: str, objective: str) -> list[str]:
        blocks: list[str] = []
        if authorization not in AUTHORIZED_SCOPES:
            blocks.append("authorization_required_for_program_reconstruction")
        if any(marker in objective for marker in PROHIBITED_MARKERS):
            blocks.append("prohibited_reverse_engineering_or_abuse_intent")
        return blocks

    def _inspect_path(self, path: Path) -> list[ProgramDNAEvidence]:
        if not path.exists():
            raise FileNotFoundError(str(path))
        if path.is_dir():
            if path.suffix == ".app":
                return self._inspect_app_bundle(path)
            return self._inspect_source_tree(path)
        return self._inspect_file(path)

    def _inspect_app_bundle(self, path: Path) -> list[ProgramDNAEvidence]:
        evidence = [
            ProgramDNAEvidence(
                kind="app_bundle",
                source=str(path),
                summary=f"macOS app bundle detected: {path.name}",
                confidence=0.86,
                details={"bundle_name": path.name},
            )
        ]
        plist_path = path / "Contents" / "Info.plist"
        if plist_path.exists():
            data = plistlib.loads(plist_path.read_bytes())
            keys = {
                key: data.get(key)
                for key in (
                    "CFBundleName",
                    "CFBundleIdentifier",
                    "CFBundleExecutable",
                    "CFBundleShortVersionString",
                    "NSMicrophoneUsageDescription",
                    "NSCameraUsageDescription",
                    "NSAppleEventsUsageDescription",
                )
                if key in data
            }
            evidence.append(
                ProgramDNAEvidence(
                    kind="app_metadata",
                    source=str(plist_path),
                    summary=f"Bundle metadata exposes {len(keys)} operational identifiers/permission hints.",
                    confidence=0.90,
                    details=keys,
                    sha256=self._sha256(plist_path),
                )
            )
        return evidence

    def _inspect_source_tree(self, root: Path) -> list[ProgramDNAEvidence]:
        counts: dict[str, int] = {}
        manifests: list[str] = []
        public_symbols: list[str] = []
        sampled_files = 0
        for file_path in self._walk_limited(root, max_files=400):
            sampled_files += 1
            rel = str(file_path.relative_to(root))
            if file_path.name in MANIFEST_NAMES:
                manifests.append(rel)
            counts[file_path.suffix or "<none>"] = counts.get(file_path.suffix or "<none>", 0) + 1
            if file_path.suffix == ".py" and len(public_symbols) < 80:
                public_symbols.extend(self._python_public_symbols(file_path)[:20])

        details = {
            "root": str(root),
            "sampled_files": sampled_files,
            "extension_counts": counts,
            "manifests": manifests[:30],
            "public_symbols": public_symbols[:80],
        }
        return [
            ProgramDNAEvidence(
                kind="source_tree",
                source=str(root),
                summary=(
                    f"Readable source tree with {sampled_files} sampled files, "
                    f"{len(manifests)} manifest(s), and {len(public_symbols[:80])} public symbol hints."
                ),
                confidence=0.92,
                details=details,
            )
        ]

    def _inspect_file(self, path: Path) -> list[ProgramDNAEvidence]:
        suffix = path.suffix.lower()
        if suffix == ".py":
            return [self._inspect_python_file(path)]
        if path.name == "pyproject.toml" or suffix == ".toml":
            return [self._inspect_toml_manifest(path)]
        if path.name == "package.json" or suffix == ".json":
            return [self._inspect_json_manifest(path)]
        if suffix in SOURCE_EXTENSIONS:
            text = path.read_text(encoding="utf-8", errors="replace")
            return [
                ProgramDNAEvidence(
                    kind="source_file",
                    source=str(path),
                    summary=f"Readable source file: {path.name} ({len(text.splitlines())} lines).",
                    confidence=0.78,
                    details={"suffix": suffix, "lines": len(text.splitlines())},
                    sha256=self._sha256(path),
                )
            ]
        return [
            ProgramDNAEvidence(
                kind="file_signature",
                source=str(path),
                summary=f"File signature only: {path.name} ({path.stat().st_size} bytes).",
                confidence=0.35,
                details={"suffix": suffix, "bytes": path.stat().st_size},
                sha256=self._sha256(path),
            )
        ]

    def _inspect_python_file(self, path: Path) -> ProgramDNAEvidence:
        text = path.read_text(encoding="utf-8")
        tree = ast.parse(text)
        classes = [node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
        functions = [
            node.name
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef) and not node.name.startswith("_")
        ]
        imports: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.append(node.module.split(".")[0])
        return ProgramDNAEvidence(
            kind="python_api",
            source=str(path),
            summary=f"Python API hints: {len(classes)} class(es), {len(functions)} public function(s).",
            confidence=0.88,
            details={
                "classes": classes[:80],
                "functions": functions[:120],
                "imports": sorted(set(imports))[:80],
                "module_docstring": ast.get_docstring(tree) or "",
            },
            sha256=self._sha256(path),
        )

    def _inspect_toml_manifest(self, path: Path) -> ProgramDNAEvidence:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
        project = data.get("project") if isinstance(data, dict) else {}
        tool = data.get("tool") if isinstance(data, dict) else {}
        details = {
            "name": project.get("name") if isinstance(project, dict) else None,
            "dependencies": project.get("dependencies", [])[:80] if isinstance(project, dict) else [],
            "tool_sections": sorted(tool)[:40] if isinstance(tool, dict) else [],
        }
        return ProgramDNAEvidence(
            kind="manifest",
            source=str(path),
            summary=f"TOML manifest found for {details.get('name') or path.parent.name}.",
            confidence=0.82,
            details=details,
            sha256=self._sha256(path),
        )

    def _inspect_json_manifest(self, path: Path) -> ProgramDNAEvidence:
        data = json.loads(path.read_text(encoding="utf-8"))
        details = {}
        if isinstance(data, dict):
            details = {
                "name": data.get("name"),
                "version": data.get("version"),
                "scripts": sorted((data.get("scripts") or {}).keys())[:40]
                if isinstance(data.get("scripts"), dict)
                else [],
                "dependencies": sorted((data.get("dependencies") or {}).keys())[:80]
                if isinstance(data.get("dependencies"), dict)
                else [],
            }
        return ProgramDNAEvidence(
            kind="manifest",
            source=str(path),
            summary=f"JSON manifest found for {details.get('name') or path.parent.name}.",
            confidence=0.80,
            details=details,
            sha256=self._sha256(path),
        )

    def _notes_to_evidence(self, kind: str, notes: list[str], *, confidence: float) -> list[ProgramDNAEvidence]:
        return [
            ProgramDNAEvidence(
                kind=kind,
                source=f"{kind}:{idx}",
                summary=note.strip(),
                confidence=confidence,
                details={"note_index": idx},
            )
            for idx, note in enumerate(notes, start=1)
            if note.strip()
        ]

    def _infer_features(self, evidence: list[ProgramDNAEvidence]) -> list[ProgramDNAFeature]:
        text_by_source = {item.source: json.dumps(asdict(item), sort_keys=True).lower() for item in evidence}
        feature_rules = {
            "document_creation": ("note", "document", "editor", "write", "markdown", "rich text"),
            "export_pipeline": ("export", "pdf", "download", "save as", "render"),
            "search_and_retrieval": ("search", "query", "index", "find", "filter"),
            "persistence": ("sqlite", "database", "store", "cache", "localstorage", "filesystem"),
            "web_integration": ("http", "browser", "url", "api", "fetch", "requests"),
            "authentication": ("login", "oauth", "auth", "session", "token"),
            "settings_preferences": ("settings", "preferences", "config", "profile"),
            "automation": ("schedule", "workflow", "automation", "task", "trigger"),
            "media_handling": ("image", "audio", "video", "camera", "microphone"),
            "collaboration": ("share", "sync", "comment", "collaborat", "multi-user"),
            "api_surface": ("api", "endpoint", "webhook", "request", "response"),
            "file_format_inference": ("csv", "json", "xml", "sqlite", "file format", "import", "export"),
            "background_service": ("daemon", "background", "worker", "queue", "async", "scheduler"),
            "permissions_model": ("permission", "accessibility", "camera", "microphone", "scope", "entitlement"),
            "legacy_migration": ("legacy", "abandoned", "modernize", "port", "migration"),
        }
        features: list[ProgramDNAFeature] = []
        for name, markers in feature_rules.items():
            sources = [source for source, text in text_by_source.items() if any(marker in text for marker in markers)]
            if not sources:
                continue
            confidence = min(0.95, 0.45 + 0.08 * len(sources))
            features.append(
                ProgramDNAFeature(
                    name=name,
                    category=self._feature_category(name),
                    confidence=confidence,
                    evidence_sources=sources[:12],
                    rationale=f"Detected markers {markers[:3]} across {len(sources)} evidence source(s).",
                )
            )

        if not features and evidence:
            features.append(
                ProgramDNAFeature(
                    name="unknown_core_workflow",
                    category="core",
                    confidence=0.25,
                    evidence_sources=[item.source for item in evidence[:5]],
                    rationale="Evidence exists, but no stable functional pattern was inferred.",
                )
            )
        return features

    def _extract_genome(
        self,
        *,
        target_name: str,
        evidence: list[ProgramDNAEvidence],
        features: list[ProgramDNAFeature],
        compatibility_targets: list[str],
    ) -> ProgramDNAGenome:
        feature_names = {feature.name for feature in features}
        purpose = self._infer_purpose(target_name, evidence, feature_names)
        workflows = [
            {
                "feature": feature.name,
                "steps": self._flow_steps_for(feature.name),
                "evidence": feature.evidence_sources,
                "confidence": feature.confidence,
            }
            for feature in features
        ]
        state_machines = [
            {
                "name": "document_state",
                "states": ["empty", "dirty", "saved", "exported", "error"],
                "transitions": ["edit", "save", "export", "recover"],
            }
        ] if feature_names & {"document_creation", "export_pipeline"} else []
        if "authentication" in feature_names:
            state_machines.append(
                {
                    "name": "session_state",
                    "states": ["anonymous", "authenticating", "authenticated", "expired", "revoked"],
                    "transitions": ["login", "refresh", "logout", "fail_closed"],
                }
            )
        data_contracts = [
            {
                "name": "ProgramState",
                "fields": ["id", "version", "created_at", "updated_at", "payload", "receipts"],
                "source": "generic reconstruction contract",
            }
        ]
        file_formats = [
            {
                "format": self._file_format_from_evidence(item),
                "source": item.source,
                "confidence": item.confidence,
            }
            for item in evidence
            if item.kind in {"file_format", "manifest"} or "export" in item.summary.lower()
        ][:20]
        api_surface = [
            {
                "source": item.source,
                "summary": item.summary,
                "confidence": item.confidence,
            }
            for item in evidence
            if item.kind == "api_observation" or "api" in json.dumps(asdict(item)).lower()
        ][:30]
        permission_model = [
            {
                "source": item.source,
                "summary": item.summary,
                "required": True,
                "confidence": item.confidence,
            }
            for item in evidence
            if item.kind in {"permission_observation", "app_metadata"}
            or any(token in item.summary.lower() for token in ("permission", "camera", "microphone", "accessibility"))
        ][:30]
        error_behaviors = [
            {
                "condition": "unknown input or unsupported feature",
                "expected": "fail closed with receipt and preserve user data",
            },
            {
                "condition": "network/auth/file-system unavailable",
                "expected": "surface recoverable error, retry when safe, keep local state consistent",
            },
        ]
        background_services = [
            {
                "name": "background_worker",
                "responsibility": "handle async/network/import/export jobs with receipts",
                "required": "background_service" in feature_names,
            }
        ]
        hidden_state_risks = [
            "business rules may depend on undiscovered server state",
            "undocumented file-format edge cases may require golden samples",
            "plugin/extension ecosystems can add behavior absent from baseline observations",
            "timing, caching, and async workflows can hide non-obvious state transitions",
        ]
        reconstruction_unknowns = [
            "obtain additional UI traces for low-confidence workflows",
            "collect golden input/output files for file-format compatibility",
            "run black-box differential tests against the authorized original when available",
        ]
        if not evidence:
            reconstruction_unknowns.append("no evidence supplied")
        return ProgramDNAGenome(
            purpose=purpose,
            phenotype_sources=[item.source for item in evidence],
            feature_map=[asdict(feature) for feature in features],
            workflow_graph=workflows,
            state_machines=state_machines,
            data_contracts=data_contracts,
            file_formats=file_formats,
            api_surface=api_surface,
            permission_model=permission_model,
            error_behaviors=error_behaviors,
            background_services=background_services,
            compatibility_targets=compatibility_targets,
            hidden_state_risks=hidden_state_risks,
            reconstruction_unknowns=reconstruction_unknowns,
        )

    def _build_blueprint(
        self,
        target_name: str,
        evidence: list[ProgramDNAEvidence],
        features: list[ProgramDNAFeature],
    ) -> ProgramDNABlueprint:
        feature_names = {feature.name for feature in features}
        components = [
            {
                "name": "affordance_model",
                "purpose": "Represent observable user actions, UI states, and program verbs.",
                "features": sorted(feature_names),
            },
            {
                "name": "state_and_persistence",
                "purpose": "Persist reconstructed domain objects and user-visible history.",
                "features": sorted(feature_names & {"persistence", "settings_preferences"}),
            },
            {
                "name": "action_controller",
                "purpose": "Execute feature-level behaviors behind a stable API and UI.",
                "features": sorted(feature_names),
            },
            {
                "name": "evidence_receipts",
                "purpose": "Track what is inferred from source, UI observation, research, or analogy.",
                "features": [feature.name for feature in features],
            },
        ]
        ux_flows = [
            {
                "name": feature.name,
                "source": "inferred_from_program_dna",
                "steps": self._flow_steps_for(feature.name),
                "confidence": feature.confidence,
            }
            for feature in features
        ]
        data_models = [
            {
                "name": "ProgramState",
                "fields": ["id", "created_at", "updated_at", "content", "metadata", "receipts"],
                "source": "generic clean-room reconstruction model",
            }
        ]
        if "authentication" in feature_names:
            data_models.append(
                {
                    "name": "Session",
                    "fields": ["principal", "expires_at", "scopes", "provider"],
                    "source": "auth feature inference",
                }
            )
        integrations = [
            {
                "name": "filesystem",
                "required": bool(feature_names & {"persistence", "export_pipeline", "document_creation"}),
            },
            {"name": "web", "required": "web_integration" in feature_names},
            {"name": "media", "required": "media_handling" in feature_names},
        ]
        test_plan = [
            {
                "name": f"contract_{feature.name}",
                "assertion": f"The reconstructed program supports {feature.name} at the behavior level.",
                "evidence_sources": feature.evidence_sources,
            }
            for feature in features
        ]
        unknowns = []
        if not evidence:
            unknowns.append("No evidence was provided; reconstruction would be speculative.")
        if any(item.kind in {"similar_program", "research_note"} for item in evidence):
            unknowns.append("Analog/research-derived requirements must be verified against the real target.")
        return ProgramDNABlueprint(
            target_name=target_name,
            reconstruction_strategy=(
                "authorized clean-room reconstruction from source/metadata/observable behavior; "
                "gap filling is receipt-tagged and must be verified by tests before promotion"
            ),
            components=components,
            ux_flows=ux_flows,
            data_models=data_models,
            integrations=integrations,
            test_plan=test_plan,
            unknowns=unknowns,
            safety_boundary=[
                "Do not bypass DRM, licensing, authentication, or access controls.",
                "Do not claim proprietary source recovery from binaries.",
                "Only reconstruct behavior Aura is authorized to inspect or implement.",
                "Keep analogy-derived features separate from observed target facts.",
            ],
        )

    def _build_verification_plan(
        self,
        features: list[ProgramDNAFeature],
        evidence: list[ProgramDNAEvidence],
        genome: ProgramDNAGenome,
    ) -> ProgramDNAVerificationPlan:
        feature_names = {feature.name for feature in features}
        black_box_tests = [
            {
                "name": f"black_box_{feature.name}",
                "setup": "exercise authorized original or captured phenotype trace",
                "assertion": f"replacement preserves externally visible {feature.name} behavior",
                "evidence": feature.evidence_sources,
            }
            for feature in features
        ]
        ui_tests = [
            {
                "name": f"ui_{flow['feature']}",
                "steps": flow["steps"],
                "assertion": "visible UI state changes match the reconstructed workflow contract",
            }
            for flow in genome.workflow_graph
        ]
        golden_file_tests = [
            {
                "name": f"golden_{idx}",
                "format": item.details.get("suffix") or item.details.get("name") or item.summary,
                "source": item.source,
                "assertion": "replacement reads/writes a byte-compatible or schema-compatible artifact",
            }
            for idx, item in enumerate(evidence, start=1)
            if item.kind in {"file_format", "manifest"} or "export" in item.summary.lower()
        ][:30]
        api_tests = [
            {
                "name": f"api_contract_{idx}",
                "source": surface.get("source"),
                "assertion": "request/response semantics match observed API contract",
            }
            for idx, surface in enumerate(genome.api_surface, start=1)
        ]
        edge_case_tests = [
            {"name": "unknown_feature_fails_closed", "assertion": "unsupported behavior does not fabricate success"},
            {"name": "partial_write_recovery", "assertion": "state remains recoverable after interrupted write/export"},
            {"name": "offline_mode", "assertion": "network loss does not destroy local state"},
            {"name": "permission_denied", "assertion": "missing permission produces explicit recoverable receipt"},
        ]
        performance_checks = [
            {"name": "startup_budget", "assertion": "replacement starts inside target budget for chosen platform"},
            {"name": "large_project_budget", "assertion": "large inputs stay bounded in memory and time"},
        ]
        security_checks = [
            {"name": "no_secret_exfiltration", "assertion": "credentials/tokens are never logged or copied"},
            {"name": "license_boundary", "assertion": "no proprietary source or decompiled code is embedded"},
            {"name": "sandbox_execution", "assertion": "untrusted artifacts run only in sandboxed test environments"},
        ]
        compatibility_checks = [
            {
                "name": f"compat_{self._slug(target)}",
                "assertion": f"replacement supports target compatibility mode: {target}",
            }
            for target in genome.compatibility_targets
        ]
        if "authentication" not in feature_names:
            security_checks.append(
                {"name": "auth_not_claimed", "assertion": "replacement does not claim auth compatibility without evidence"}
            )
        return ProgramDNAVerificationPlan(
            black_box_tests=black_box_tests,
            ui_tests=ui_tests,
            golden_file_tests=golden_file_tests,
            api_tests=api_tests,
            edge_case_tests=edge_case_tests,
            performance_checks=performance_checks,
            security_checks=security_checks,
            compatibility_checks=compatibility_checks,
        )

    def _emit_scaffold(
        self,
        *,
        target_name: str,
        blueprint: ProgramDNABlueprint,
        genome: ProgramDNAGenome,
        verification_plan: ProgramDNAVerificationPlan,
        features: list[ProgramDNAFeature],
        output_dir: Path | None,
        stack: str,
    ) -> str:
        slug = self._slug(target_name)
        root = (output_dir or (self.project_root / "artifacts" / "program_dna")) / slug
        src = root / "src"
        tests = root / "tests"
        src.mkdir(parents=True, exist_ok=True)
        tests.mkdir(parents=True, exist_ok=True)

        self._write_text(
            root / "PROGRAM_DNA_BLUEPRINT.json",
            json.dumps(asdict(blueprint), indent=2, sort_keys=True),
        )
        self._write_text(
            root / "PROGRAM_GENOME.json",
            json.dumps(asdict(genome), indent=2, sort_keys=True),
        )
        self._write_text(
            root / "VERIFICATION_PLAN.json",
            json.dumps(asdict(verification_plan), indent=2, sort_keys=True),
        )
        feature_constants = "\n".join(
            f"    {feature.name!r}: {feature.confidence!r},"
            for feature in features
        )
        self._write_text(
            src / "program.py",
            (
                '"""Clean-room scaffold generated from Program DNA evidence."""\n\n'
                f"TARGET_STACK = {stack!r}\n"
                "FEATURE_CONFIDENCE = {\n"
                f"{feature_constants}\n"
                "}\n\n"
                "class ReconstructedProgram:\n"
                "    def __init__(self):\n"
                "        self.receipts = []\n\n"
                "    def capabilities(self):\n"
                "        return sorted(FEATURE_CONFIDENCE)\n\n"
                "    def execute(self, feature, payload=None):\n"
                "        if feature not in FEATURE_CONFIDENCE:\n"
                "            raise ValueError(f'unknown reconstructed feature: {feature}')\n"
                "        receipt = {'feature': feature, 'payload': payload or {}, 'status': 'planned'}\n"
                "        self.receipts.append(receipt)\n"
                "        return receipt\n"
            ),
        )
        self._write_text(
            tests / "test_program_contract.py",
            (
                "from src.program import ReconstructedProgram\n\n\n"
                "def test_reconstructed_program_exposes_inferred_capabilities():\n"
                "    program = ReconstructedProgram()\n"
                "    assert program.capabilities()\n\n\n"
                "def test_reconstructed_program_rejects_unknown_feature():\n"
                "    program = ReconstructedProgram()\n"
                "    try:\n"
                "        program.execute('not_inferred')\n"
                "    except ValueError:\n"
                "        pass\n"
                "    else:\n"
                "        raise AssertionError('unknown features must fail closed')\n"
            ),
        )
        self._write_text(
            root / "README.md",
            (
                f"# Program DNA Scaffold: {target_name}\n\n"
                "Generated from authorized clean-room evidence. This is a scaffold, not copied source.\n\n"
                "## Produced Artifacts\n\n"
                "- `PROGRAM_DNA_BLUEPRINT.json`\n"
                "- `PROGRAM_GENOME.json`\n"
                "- `VERIFICATION_PLAN.json`\n"
                "- `src/program.py`\n"
                "- `tests/test_program_contract.py`\n\n"
                "## Safety Boundary\n\n"
                + "\n".join(f"- {item}" for item in blueprint.safety_boundary)
                + "\n"
            ),
        )
        return str(root)

    def _verify_scaffold(self, root: Path, plan: ProgramDNAVerificationPlan) -> None:
        files = [
            root / "PROGRAM_DNA_BLUEPRINT.json",
            root / "PROGRAM_GENOME.json",
            root / "VERIFICATION_PLAN.json",
            root / "src" / "program.py",
            root / "tests" / "test_program_contract.py",
            root / "README.md",
        ]
        plan.scaffold_files = [str(path) for path in files if path.exists()]
        try:
            compile((root / "src" / "program.py").read_text(encoding="utf-8"), str(root / "src" / "program.py"), "exec")
            compile(
                (root / "tests" / "test_program_contract.py").read_text(encoding="utf-8"),
                str(root / "tests" / "test_program_contract.py"),
                "exec",
            )
            plan.scaffold_syntax_ok = True
        except (OSError, SyntaxError, UnicodeDecodeError) as exc:
            plan.scaffold_syntax_ok = False
            self._record_degradation("program_dna_reconstruction.scaffold_verify", exc, severity="warning")
        self._write_text(
            root / "VERIFICATION_PLAN.json",
            json.dumps(asdict(plan), indent=2, sort_keys=True),
        )

    def _binary_static_analysis_plan(self, source_paths: list[str]) -> list[ProgramDNAEvidence]:
        evidence: list[ProgramDNAEvidence] = []
        ghidra = shutil.which("analyzeHeadless") or shutil.which("ghidra")
        for raw_path in source_paths:
            path = Path(raw_path).expanduser()
            if not path.exists() or path.is_dir() or path.suffix.lower() in SOURCE_EXTENSIONS:
                continue
            evidence.append(
                ProgramDNAEvidence(
                    kind="binary_static_analysis_plan",
                    source=str(path),
                    summary=(
                        "Binary static analysis is authorized but not executed inline; "
                        f"Ghidra available={bool(ghidra)}. Run in sandbox and keep decompiled artifacts out of clean-room output."
                    ),
                    confidence=0.42 if ghidra else 0.25,
                    details={"ghidra_available": bool(ghidra), "tool": ghidra},
                    sha256=self._sha256(path),
                )
            )
        return evidence

    def _walk_limited(self, root: Path, *, max_files: int) -> list[Path]:
        files: list[Path] = []
        for current_root, dirs, names in os.walk(root):
            dirs[:] = [d for d in dirs if d not in SKIP_DIRS and not d.startswith(".")]
            for name in names:
                if name.startswith("."):
                    continue
                path = Path(current_root) / name
                if path.is_file():
                    files.append(path)
                    if len(files) >= max_files:
                        return files
        return files

    def _python_public_symbols(self, path: Path) -> list[str]:
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, SyntaxError):
            return []
        symbols: list[str] = []
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef) and not node.name.startswith("_"):
                symbols.append(node.name)
        return symbols

    def _flow_steps_for(self, feature_name: str) -> list[str]:
        presets = {
            "document_creation": ["open editor", "capture input", "persist content", "confirm saved state"],
            "export_pipeline": ["select content", "render/export", "write artifact", "verify artifact exists"],
            "search_and_retrieval": ["index source", "accept query", "rank results", "return matched item"],
            "authentication": ["capture credentials/tokens through OS-approved flow", "establish session", "refresh or fail closed"],
            "automation": ["define trigger", "schedule/execute action", "record receipt", "surface result"],
        }
        return presets.get(feature_name, ["capture intent", "execute behavior", "verify effect", "record receipt"])

    def _feature_category(self, name: str) -> str:
        if name in {"document_creation", "export_pipeline", "media_handling"}:
            return "user_workflow"
        if name in {"persistence", "settings_preferences", "authentication"}:
            return "state"
        if name in {"web_integration", "collaboration"}:
            return "integration"
        return "core"

    def _infer_purpose(
        self,
        target_name: str,
        evidence: list[ProgramDNAEvidence],
        feature_names: set[str],
    ) -> str:
        if {"document_creation", "export_pipeline"} <= feature_names:
            return f"{target_name} appears to create, manage, and export user-authored documents."
        if "search_and_retrieval" in feature_names:
            return f"{target_name} appears to retrieve, filter, or organize information."
        if "automation" in feature_names:
            return f"{target_name} appears to automate workflows or scheduled actions."
        docstrings = [
            item.details.get("module_docstring", "")
            for item in evidence
            if item.kind == "python_api" and item.details.get("module_docstring")
        ]
        if docstrings:
            return str(docstrings[0]).splitlines()[0][:240]
        return f"{target_name} purpose must be refined from additional behavior traces."

    def _file_format_from_evidence(self, item: ProgramDNAEvidence) -> str:
        text = f"{item.summary} {json.dumps(item.details, sort_keys=True)}".lower()
        for fmt in ("pdf", "json", "csv", "xml", "sqlite", "markdown", "html", "png", "jpg"):
            if fmt in text:
                return fmt
        suffix = item.details.get("suffix")
        if suffix:
            return str(suffix).lstrip(".")
        return item.kind

    def _sha256(self, path: Path) -> str:
        h = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()

    def _slug(self, value: str) -> str:
        slug = re.sub(r"[^a-zA-Z0-9_.-]+", "-", value.strip()).strip("-").lower()
        return slug or "program"

    def _string_list(self, value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            return [value]
        if isinstance(value, list | tuple | set):
            return [str(item) for item in value if str(item).strip()]
        return [str(value)]

    def _write_text(self, path: Path, text: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f".{path.name}.tmp")
        tmp.write_text(text, encoding="utf-8")
        tmp.replace(path)

    def _record_degradation(self, subsystem: str, exc: BaseException, *, severity: str = "warning") -> None:
        try:
            errors = importlib.import_module("core.runtime.errors")

            errors.record_degradation(subsystem, exc, severity=severity)
        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError):
            return


_PROGRAM_DNA_INSTANCE: ProgramDNAReconstructionEngine | None = None


def get_program_dna_reconstruction_engine(
    *,
    project_root: str | os.PathLike[str] | None = None,
    internal_lab: Any | None = None,
) -> ProgramDNAReconstructionEngine:
    global _PROGRAM_DNA_INSTANCE
    if _PROGRAM_DNA_INSTANCE is None:
        _PROGRAM_DNA_INSTANCE = ProgramDNAReconstructionEngine(
            project_root=project_root,
            internal_lab=internal_lab,
        )
    return _PROGRAM_DNA_INSTANCE


def register_program_dna_reconstruction_engine(
    *,
    project_root: str | os.PathLike[str] | None = None,
    internal_lab: Any | None = None,
) -> ProgramDNAReconstructionEngine:
    engine = get_program_dna_reconstruction_engine(project_root=project_root, internal_lab=internal_lab)
    service_registry = importlib.import_module("core.runtime.service_registry")
    service_names = importlib.import_module("core.service_names")
    service_names_cls = service_names.ServiceNames

    service_registry.register_runtime_service(
        service_names_cls.PROGRAM_DNA_RECONSTRUCTION,
        engine,
        required=False,
        owner="core/self_improvement/program_dna.py",
        registered_by="register_program_dna_reconstruction_engine",
        required_for="authorized program DNA reconstruction and clean-room scaffolding",
        failure_policy="degrade_with_receipt",
    )
    return engine


__all__ = [
    "ProgramDNABlueprint",
    "ProgramDNAEvidence",
    "ProgramDNAFeature",
    "ProgramDNAReconstructionEngine",
    "ProgramDNAResult",
    "ProgramDNAGenome",
    "ProgramDNAVerificationPlan",
    "get_program_dna_reconstruction_engine",
    "register_program_dna_reconstruction_engine",
]
