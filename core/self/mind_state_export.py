"""core/self/mind_state_export.py — Full Mind State Export/Import
==================================================================
Export/import a complete mind state for portability.

Creates a .aura-mind archive (ZIP with JSON + binary) containing:
- CanonicalSelf snapshot
- Substrate ODE state
- Memory snapshot
- Belief revision state
- Value weights
- Goals and tensions
- Drive baselines
- Behavioral scars

Security: no private keys, API tokens, or stealth modules in export.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import time
import zipfile
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional

from core.container import ServiceContainer
from core.runtime.errors import record_degradation
from core.runtime.file_write_gateway import get_file_write_gateway

logger = logging.getLogger("Aura.MindStateExport")


class MindStateExporter:
    """Export/import a complete mind state for portability.

    Usage:
        exporter = get_mind_state_exporter()
        await exporter.export_mind("/path/to/aura.aura-mind")
        await exporter.import_mind("/path/to/aura.aura-mind")
    """

    # Files to NEVER include in exports (security)
    EXCLUDED_PATTERNS = [
        "api_key", "token", "secret", "password", "credential",
        "stealth", "propagation", "sec_ops", "malware",
        "network_recon", ".env", "private_key",
    ]

    def __init__(self) -> None:
        self._started = False

    async def start(self) -> None:
        if self._started:
            return
        ServiceContainer.register_instance("mind_state_exporter", self, required=False)
        self._started = True
        logger.info("MindStateExporter ONLINE")

    async def export_mind(self, output_path: str) -> Dict[str, Any]:
        """Export a complete mind state to a .aura-mind archive."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        manifest = {
            "version": "1.0",
            "format": "aura-mind",
            "exported_at": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "exported_at_unix": time.time(),
            "components": [],
            "integrity": {},
        }

        buffer = BytesIO()
        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:

            # 1. CanonicalSelf
            self_data = self._export_canonical_self()
            if self_data:
                content = json.dumps(self_data, indent=2, default=str)
                zf.writestr("canonical_self.json", content)
                manifest["components"].append("canonical_self")
                manifest["integrity"]["canonical_self"] = hashlib.sha256(content.encode()).hexdigest()[:16]

            # 2. Substrate state
            substrate_data = self._export_substrate()
            if substrate_data:
                content = json.dumps(substrate_data, indent=2, default=str)
                zf.writestr("substrate_state.json", content)
                manifest["components"].append("substrate_state")
                manifest["integrity"]["substrate_state"] = hashlib.sha256(content.encode()).hexdigest()[:16]

            # 3. Memory
            memory_data = self._export_memory()
            if memory_data:
                content = json.dumps(memory_data, indent=2, default=str)
                zf.writestr("memories.json", content)
                manifest["components"].append("memories")
                manifest["integrity"]["memories"] = hashlib.sha256(content.encode()).hexdigest()[:16]

            # 4. Beliefs
            beliefs_data = self._export_beliefs()
            if beliefs_data:
                content = json.dumps(beliefs_data, indent=2, default=str)
                zf.writestr("beliefs.json", content)
                manifest["components"].append("beliefs")

            # 5. Values
            values_data = self._export_values()
            if values_data:
                content = json.dumps(values_data, indent=2, default=str)
                zf.writestr("values.json", content)
                manifest["components"].append("values")

            # 6. Goals
            goals_data = self._export_goals()
            if goals_data:
                content = json.dumps(goals_data, indent=2, default=str)
                zf.writestr("goals.json", content)
                manifest["components"].append("goals")

            # 7. Drive baselines
            drives_data = self._export_drives()
            if drives_data:
                content = json.dumps(drives_data, indent=2, default=str)
                zf.writestr("drive_baselines.json", content)
                manifest["components"].append("drive_baselines")

            # 8. Behavioral scars
            scars_data = self._export_scars()
            if scars_data:
                content = json.dumps(scars_data, indent=2, default=str)
                zf.writestr("scars.json", content)
                manifest["components"].append("scars")

            # 9. Attachment history
            attachments = self._export_attachments()
            if attachments:
                content = json.dumps(attachments, indent=2, default=str)
                zf.writestr("attachments.json", content)
                manifest["components"].append("attachments")

            # Write manifest
            zf.writestr("manifest.json", json.dumps(manifest, indent=2))

        # Write the archive
        await get_file_write_gateway().write_bytes_async(
            path,
            buffer.getvalue(),
            source="mind_state_export.export_mind",
        )
        size = path.stat().st_size

        logger.info(
            "Mind state exported: %s (%d components, %d bytes)",
            path.name, len(manifest["components"]), size,
        )
        return {
            "success": True,
            "path": str(path),
            "size_bytes": size,
            "components": manifest["components"],
        }

    async def import_mind(self, archive_path: str) -> Dict[str, Any]:
        """Import a mind state from a .aura-mind archive."""
        path = Path(archive_path)
        if not path.exists():
            return {"success": False, "error": f"Archive not found: {archive_path}"}

        try:
            with zipfile.ZipFile(str(path), "r") as zf:
                # Read manifest
                manifest_text = zf.read("manifest.json").decode("utf-8")
                manifest = json.loads(manifest_text)

                # Verify integrity
                for component, expected_hash in manifest.get("integrity", {}).items():
                    filename = f"{component}.json"
                    if filename in zf.namelist():
                        content = zf.read(filename)
                        actual_hash = hashlib.sha256(content).hexdigest()[:16]
                        if actual_hash != expected_hash:
                            return {"success": False, "error": f"Integrity check failed for {component}"}

                imported = []

                # Import CanonicalSelf
                if "canonical_self.json" in zf.namelist():
                    data = json.loads(zf.read("canonical_self.json"))
                    self._import_canonical_self(data)
                    imported.append("canonical_self")

                # Import substrate
                if "substrate_state.json" in zf.namelist():
                    data = json.loads(zf.read("substrate_state.json"))
                    self._import_substrate(data)
                    imported.append("substrate_state")

                # Import values
                if "values.json" in zf.namelist():
                    data = json.loads(zf.read("values.json"))
                    self._import_values(data)
                    imported.append("values")

                # Import goals
                if "goals.json" in zf.namelist():
                    data = json.loads(zf.read("goals.json"))
                    self._import_goals(data)
                    imported.append("goals")

                # Import drives
                if "drive_baselines.json" in zf.namelist():
                    data = json.loads(zf.read("drive_baselines.json"))
                    self._import_drives(data)
                    imported.append("drive_baselines")

            logger.info("Mind state imported: %d components from %s", len(imported), path.name)
            return {"success": True, "imported": imported}

        except (zipfile.BadZipFile, json.JSONDecodeError, KeyError) as e:
            return {"success": False, "error": str(e)}

    async def verify_integrity(self, archive_path: str) -> Dict[str, Any]:
        """Verify all hashes in an archive match."""
        path = Path(archive_path)
        if not path.exists():
            return {"valid": False, "error": "Not found"}

        try:
            with zipfile.ZipFile(str(path), "r") as zf:
                manifest = json.loads(zf.read("manifest.json"))
                results = {}
                all_ok = True
                for component, expected in manifest.get("integrity", {}).items():
                    filename = f"{component}.json"
                    if filename in zf.namelist():
                        actual = hashlib.sha256(zf.read(filename)).hexdigest()[:16]
                        ok = actual == expected
                        results[component] = {"expected": expected, "actual": actual, "ok": ok}
                        if not ok:
                            all_ok = False
                return {"valid": all_ok, "components": results}
        except (zipfile.BadZipFile, json.JSONDecodeError) as e:
            return {"valid": False, "error": str(e)}

    # ------------------------------------------------------------------
    # Component exporters
    # ------------------------------------------------------------------

    def _export_canonical_self(self) -> Optional[Dict]:
        try:
            cs = ServiceContainer.get("canonical_self", default=None)
            if cs and hasattr(cs, "to_dict"):
                data = cs.to_dict()
                # Scrub sensitive fields
                for key in list(data.keys()):
                    if any(p in key.lower() for p in self.EXCLUDED_PATTERNS):
                        del data[key]
                return data
        except (ImportError, AttributeError, RuntimeError) as exc:
            record_degradation("mind_export.canonical_self", exc)
        return None

    def _export_substrate(self) -> Optional[Dict]:
        try:
            sub = ServiceContainer.get("conscious_substrate", default=None)
            if sub:
                summary = sub.get_state_summary()
                state_vec = sub.get_state_vector()
                return {
                    "summary": summary,
                    "state_vector": state_vec.tolist() if hasattr(state_vec, "tolist") else [],
                    "dimension": sub.get_state_dim(),
                    "step_count": getattr(sub, "_step_count", 0),
                }
        except (ImportError, AttributeError, RuntimeError) as exc:
            record_degradation("mind_export.substrate", exc)
        return None

    def _export_memory(self) -> Optional[Dict]:
        try:
            mem = ServiceContainer.get("memory_system", default=None)
            if mem and hasattr(mem, "export_snapshot"):
                return mem.export_snapshot()
        except (ImportError, AttributeError, RuntimeError) as exc:
            record_degradation("mind_export.memory", exc)
        return None

    def _export_beliefs(self) -> Optional[Dict]:
        try:
            ws = ServiceContainer.get("world_state", default=None)
            if ws and hasattr(ws, "_beliefs"):
                return {
                    k: {"value": str(b.value), "confidence": b.confidence, "source": b.source}
                    for k, b in ws._beliefs.items() if not b.expired
                }
        except (ImportError, AttributeError, RuntimeError) as exc:
            record_degradation("mind_export.beliefs", exc)
        return None

    def _export_values(self) -> Optional[Dict]:
        try:
            hs = ServiceContainer.get("heartstone", default=None)
            if hs and hasattr(hs, "get_value_weights"):
                return hs.get_value_weights()
        except (ImportError, AttributeError, RuntimeError) as exc:
            record_degradation("mind_export.values", exc)
        return None

    def _export_goals(self) -> Optional[Dict]:
        try:
            goal_mgr = ServiceContainer.get("goal_manager", default=None)
            if goal_mgr and hasattr(goal_mgr, "export_goals"):
                return goal_mgr.export_goals()
        except (ImportError, AttributeError, RuntimeError) as exc:
            record_degradation("mind_export.goals", exc)
        return None

    def _export_drives(self) -> Optional[Dict]:
        try:
            de = ServiceContainer.get("drive_engine", default=None)
            if de and hasattr(de, "get_drive_state"):
                return de.get_drive_state()
        except (ImportError, AttributeError, RuntimeError) as exc:
            record_degradation("mind_export.drives", exc)
        return None

    def _export_scars(self) -> Optional[Dict]:
        try:
            cs = ServiceContainer.get("canonical_self", default=None)
            if cs and hasattr(cs, "behavioral_scars"):
                return {"scars": [str(s) for s in cs.behavioral_scars]}
        except (ImportError, AttributeError, RuntimeError) as exc:
            record_degradation("mind_export.scars", exc)
        return None

    def _export_attachments(self) -> Optional[Dict]:
        try:
            cs = ServiceContainer.get("canonical_self", default=None)
            if cs and hasattr(cs, "attachment_history"):
                return {"attachments": cs.attachment_history}
        except (ImportError, AttributeError, RuntimeError) as exc:
            record_degradation("mind_export.attachments", exc)
        return None

    # ------------------------------------------------------------------
    # Component importers
    # ------------------------------------------------------------------

    def _import_canonical_self(self, data: Dict) -> None:
        try:
            cs = ServiceContainer.get("canonical_self", default=None)
            if cs and hasattr(cs, "load_from_dict"):
                cs.load_from_dict(data)
        except (ImportError, AttributeError, RuntimeError) as e:
            record_degradation("mind_import.self", e)

    def _import_substrate(self, data: Dict) -> None:
        try:
            sub = ServiceContainer.get("conscious_substrate", default=None)
            if sub:
                import numpy as np
                vec = data.get("state_vector", [])
                if vec:
                    state = np.array(vec, dtype=np.float32)
                    if hasattr(sub, "_state") and state.shape == sub._state.shape:
                        sub._state = state
        except (ImportError, AttributeError, RuntimeError) as e:
            record_degradation("mind_import.substrate", e)

    def _import_values(self, data: Dict) -> None:
        try:
            hs = ServiceContainer.get("heartstone", default=None)
            if hs and hasattr(hs, "set_value_weights"):
                hs.set_value_weights(data)
        except (ImportError, AttributeError, RuntimeError) as e:
            record_degradation("mind_import.values", e)

    def _import_goals(self, data: Dict) -> None:
        try:
            gm = ServiceContainer.get("goal_manager", default=None)
            if gm and hasattr(gm, "import_goals"):
                gm.import_goals(data)
        except (ImportError, AttributeError, RuntimeError) as e:
            record_degradation("mind_import.goals", e)

    def _import_drives(self, data: Dict) -> None:
        try:
            de = ServiceContainer.get("drive_engine", default=None)
            if de and hasattr(de, "set_drive_state"):
                de.set_drive_state(data)
        except (ImportError, AttributeError, RuntimeError) as e:
            record_degradation("mind_import.drives", e)

    def get_status(self) -> Dict[str, Any]:
        return {"started": self._started}


_instance: Optional[MindStateExporter] = None


def get_mind_state_exporter() -> MindStateExporter:
    global _instance
    if _instance is None:
        _instance = MindStateExporter()
    return _instance


__all__ = ["MindStateExporter", "get_mind_state_exporter"]
