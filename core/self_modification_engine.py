"""[DEPRECATED] Legacy Autonomous Self-Modification Engine.
Superseded by core/self_modification/self_modification_engine.py (inside package).
"""
from core.runtime.errors import record_degradation
import warnings

warnings.warn("core/self_modification_engine.py is deprecated. Use core/self_modification package.", DeprecationWarning)

import ast
import difflib
import hashlib
import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("self_modification.engine")

_LEGACY_DIRECT_APPLICATION_ERROR = "legacy_self_modification_engine_direct_application_disabled"

class ModificationState(Enum):
    """Self-modification operational states"""

    IDLE = "idle"
    ANALYZING = "analyzing"
    PROPOSING = "proposing"
    APPLYING = "applying"
    VERIFYING = "verifying"
    ROLLING_BACK = "rolling_back"
    BLOCKED = "blocked"

class SecurityLevel(Enum):
    """Security clearance levels"""

    LOW = "low"      # Minor fixes (typos, formatting)
    MEDIUM = "medium"  # Logic fixes, error handling
    HIGH = "high"    # Core algorithm changes
    CRITICAL = "critical"  # Security/auth changes

@dataclass
class CodeChange:
    """Representation of a code modification"""

    target_file: Path
    target_line: int
    original_code: str
    modified_code: str
    change_type: str  # 'add', 'remove', 'modify'
    description: str
    security_level: SecurityLevel
    checksum: str
    
    def generate_diff(self) -> str:
        """Generate unified diff for this change."""
        diff = difflib.unified_diff(
            self.original_code.splitlines(keepends=True),
            self.modified_code.splitlines(keepends=True),
            fromfile=str(self.target_file),
            tofile=str(self.target_file),
            lineterm=''
        )
        return ''.join(diff)
    
    def calculate_checksum(self) -> str:
        """Calculate SHA-256 checksum of the change."""
        content = f"{self.target_file}:{self.target_line}:{self.modified_code}"
        return hashlib.sha256(content.encode()).hexdigest()

@dataclass
class ModificationProposal:
    """Complete proposal for code modification"""

    id: str
    changes: List[CodeChange]
    justification: str
    confidence: float  # 0.0-1.0
    expected_impact: str
    rollback_plan: Dict[str, Any]
    created_at: float
    creator: str = "self_modification_engine"

class SelfModificationEngine:
    """Secure, auditable self-modification engine with:
    - Multi-level security checks
    - Automatic rollback capabilities
    - Comprehensive audit logging
    - Syntax validation
    - Test verification
    """
    
    def __init__(
        self,
        code_base_path: str,
        backup_dir: str = "backups",
        max_backups: int = 10,
        require_human_approval: bool = True,
        allowed_file_patterns: Optional[List[str]] = None
    ):
        """Initialize the self-modification engine.
        """
        self.code_base = Path(code_base_path).resolve()
        self.backup_dir = Path(backup_dir).resolve()
        self.max_backups = max_backups
        self.require_human_approval = require_human_approval
        self.allowed_patterns = allowed_file_patterns or [r".*\.py$"]
        
        # Security settings
        self.blocked_patterns = [
            r".*__init__\.py$",
            r".*self_modification.*\.py$",  # Cannot modify itself
            r".*kernel\.py$",  # Critical kernel files
        ]
        
        # State tracking
        self.state = ModificationState.IDLE
        self.active_proposal: Optional[ModificationProposal] = None
        self.audit_log: List[Dict[str, Any]] = []
        self.change_history: List[ModificationProposal] = []
        
        # Initialize directories
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("Self-modification engine initialized for: %s", self.code_base)
    
    def analyze_file(self, file_path: Path) -> Dict[str, Any]:
        """Analyze a file for potential issues.
        """
        if not self._is_file_allowed(file_path):
            return {"allowed": False, "reason": "file_not_allowed"}
        
        try:
            content = file_path.read_text(encoding='utf-8')
            
            # Syntax validation
            try:
                ast.parse(content)
                syntax_valid = True
                syntax_error = None
            except SyntaxError as e:
                syntax_valid = False
                syntax_error = str(e)
            
            # Basic metrics
            lines = content.splitlines()
            metrics = {
                "line_count": len(lines),
                "char_count": len(content),
                "import_count": len([l for l in lines if l.strip().startswith('import')]),
                "function_count": len([l for l in lines if l.strip().startswith('def ')]),
                "class_count": len([l for l in lines if l.strip().startswith('class ')]),
            }
            
            return {
                "allowed": True,
                "syntax_valid": syntax_valid,
                "syntax_error": syntax_error,
                "metrics": metrics,
                "checksum": hashlib.sha256(content.encode()).hexdigest()
            }
            
        except (RuntimeError, AttributeError, TypeError, ValueError) as e:
            record_degradation('self_modification_engine', e)
            logger.error("File analysis failed: %s", e)
            return {"allowed": False, "reason": f"analysis_error: {e}"}
    
    def verify_changes(self, changes: List[CodeChange]) -> Dict[str, Any]:
        """Verify changes using the CodeRepairSandbox.
        """
        try:
            # Import here to avoid circular dependencies
            try:
                from ..security.code_sandbox import CodeRepairSandbox
            except ImportError:
                 from autonomy_engine.security.code_sandbox import CodeRepairSandbox

            sandbox = CodeRepairSandbox()
            
            for change in changes:
                # Apply change in memory to get the full new content
                if not change.target_file.exists():
                    # New file
                    full_patched_content = change.modified_code
                else:
                    # Existing file - apply patch in memory
                    file_content = change.target_file.read_text(encoding='utf-8')
                    lines = file_content.splitlines(keepends=True)
                    
                    # This application logic must match _apply_change exactly or we get mismatches
                    if 0 <= change.target_line - 1 < len(lines):
                        if change.change_type == "modify":
                            lines[change.target_line - 1] = change.modified_code + '\n'
                        elif change.change_type == "add":
                            lines.insert(change.target_line - 1, change.modified_code + '\n')
                        elif change.change_type == "remove":
                            lines.pop(change.target_line - 1)
                    
                    full_patched_content = ''.join(lines)
                
                # Verify in sandbox
                result = sandbox.verify_patch(change.target_file, full_patched_content)
                
                if not result["syntax_valid"] or not result["static_check_passed"]:
                    logger.warning("Verification failed for %s: %s", change.target_file, result['error'])
                    return {
                        "success": False,
                        "error": result['error'],
                        "details": result['details']
                    }
                    
            return {"success": True}
            
        except (ImportError, AttributeError, RuntimeError) as e:
            record_degradation('self_modification_engine', e)
            logger.error("Verification process failed: %s", e)
            return {"success": False, "error": str(e)}

    def apply_proposal(
        self,
        proposal: ModificationProposal,
        force: bool = False,
        dry_run: bool = False
    ) -> Dict[str, Any]:
        """Dry-run a legacy modification proposal or block direct application."""
        self.state = ModificationState.APPLYING
        self.active_proposal = proposal
        
        try:
            if dry_run:
                event = {
                    "proposal_id": proposal.id,
                    "dry_run": True,
                    "changes": len(proposal.changes),
                    "timestamp": time.time(),
                }
                self.audit_log.append(event)
                self.active_proposal = None
                self.state = ModificationState.IDLE
                return {"success": True, "dry_run": True}

            if force or not self.require_human_approval:
                self.state = ModificationState.BLOCKED
                self.active_proposal = None
                event = {
                    "proposal_id": proposal.id,
                    "error": _LEGACY_DIRECT_APPLICATION_ERROR,
                    "changes": len(proposal.changes),
                    "timestamp": time.time(),
                }
                self.audit_log.append(event)
                return {
                    "success": False,
                    "error": _LEGACY_DIRECT_APPLICATION_ERROR,
                    "message": (
                        "The deprecated top-level self-modification engine is "
                        "analysis/dry-run only. Source promotion must use the "
                        "core.self_modification SafeSelfModification quarantine "
                        "pipeline with explicit repair-lab or supervised approval."
                    ),
                    "requires": "core.self_modification.safe_modification.SafeSelfModification",
                }

            self.active_proposal = None
            self.state = ModificationState.IDLE
            return {
                "success": False,
                "error": "human_approval_required",
                "message": (
                    "The deprecated top-level engine no longer accepts approval "
                    "for live source mutation. Use dry_run=True for analysis, or "
                    "submit the proposal to the safe self-modification pipeline."
                ),
                "requires": "core.self_modification.safe_modification.SafeSelfModification",
            }
            
        except (RuntimeError, AttributeError, TypeError, ValueError) as e:
            record_degradation('self_modification_engine', e)
            logger.error("Proposal application failed: %s", e, exc_info=True)
            self.active_proposal = None
            self.state = ModificationState.IDLE
            return {"success": False, "error": f"application_exception: {e}"}
            
    def _create_backup(self, proposal: ModificationProposal) -> Optional[Path]:
        """Create backup of files to be modified."""
        backup_id = f"backup_{proposal.id}_{int(time.time())}"
        backup_path = self.backup_dir / backup_id
        backup_path.mkdir(parents=True, exist_ok=True)
        
        try:
            for change in proposal.changes:
                # Backup the entire file
                # Assuming relative paths from code_base
                rel_path = change.target_file.relative_to(self.code_base)
                backup_file = backup_path / rel_path
                backup_file.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(change.target_file, backup_file)
            
            return backup_path
            
        except (OSError, IOError) as e:
            record_degradation('self_modification_engine', e)
            logger.error("Backup creation failed: %s", e)
            return None
    
    def _rollback(self, backup_path: Path) -> bool:
        """Rollback to backup."""
        try:
            for item in backup_path.rglob("*"):
                if item.is_file():
                    rel_path = item.relative_to(backup_path)
                    target = self.code_base / rel_path
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(item, target)
            return True
        except (OSError, IOError) as e:
            record_degradation('self_modification_engine', e)
            logger.error("Rollback failed: %s", e)
            return False

    def _apply_change(self, change: CodeChange, dry_run: bool) -> Dict[str, Any]:
        """Apply a single code change."""
        try:
            if dry_run:
                return {
                    "success": True,
                    "dry_run": True,
                    "file": str(change.target_file),
                    "line": change.target_line
                }

            return {
                "success": False,
                "error": _LEGACY_DIRECT_APPLICATION_ERROR,
                "file": str(change.target_file),
                "line": change.target_line,
            }
            
        except (RuntimeError, AttributeError, TypeError, ValueError) as e:
            record_degradation('self_modification_engine', e)
            logger.error("Change application failed: %s", e)
            return {
                "success": False,
                "error": str(e)
            }

    def _is_file_allowed(self, file_path: Path) -> bool:
        """Check if file can be modified."""
        try:
            file_path.resolve().relative_to(self.code_base)
        except ValueError:
            return False
        
        for pattern in self.blocked_patterns:
            if re.match(pattern, str(file_path)):
                return False
        
        for pattern in self.allowed_patterns:
            if re.match(pattern, str(file_path)):
                return True
        
        return False
