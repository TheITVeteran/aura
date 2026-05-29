"""tests/personhood/test_tool_breadth.py
======================================
Unit tests verifying the open-ended actuators:
  1. CodeExecutionActuator
  2. WebSearchActuator
  3. WebFetchActuator
  4. GitActuator
  5. PackageInstallActuator
  6. ProcessSupervisorActuator
"""

import pytest
from core.actuators.actuator_registry import get_actuator_registry


def test_actuator_registration():
    """Verify that all new actuators are registered in the registry."""
    registry = get_actuator_registry()
    
    assert registry.get_actuator("code_execution") is not None
    assert registry.get_actuator("web_search") is not None
    assert registry.get_actuator("web_fetch") is not None
    assert registry.get_actuator("git_operation") is not None
    assert registry.get_actuator("package_install") is not None
    assert registry.get_actuator("process_supervisor") is not None


def test_code_execution_validation():
    """Verify AST safety validation in CodeExecutionActuator."""
    actuator = get_actuator_registry().get_actuator("code_execution")
    
    # Safe code
    assert actuator.validate_params({"code": "x = 1\ny = 2\nprint(x + y)"}) is True
    
    # Banned module import
    assert actuator.validate_params({"code": "import subprocess\nsubprocess.run(['ls'])"}) is False
    assert actuator.validate_params({"code": "from subprocess import Popen"}) is False
    assert actuator.validate_params({"code": "import os\nos.system('ls')"}) is False
    
    # Banned functions (eval, exec)
    assert actuator.validate_params({"code": "eval('2 + 2')"}) is False
    assert actuator.validate_params({"code": "exec('x = 5')"}) is False
    assert actuator.validate_params({"code": "__import__('os').system('ls')"}) is False
    
    # Network import when network_access is False (default)
    assert actuator.validate_params({"code": "import requests"}) is False
    # Allowed when network_access is True
    assert actuator.validate_params({"code": "import requests", "network_access": True}) is True


def test_web_fetch_validation():
    """Verify domain allowlist validation in WebFetchActuator."""
    actuator = get_actuator_registry().get_actuator("web_fetch")
    
    # Valid allowlisted domains
    assert actuator.validate_params({"url": "https://wikipedia.org/wiki/Artificial_intelligence"}) is True
    assert actuator.validate_params({"url": "https://github.com/youngbryan97/aura"}) is True
    assert actuator.validate_params({"url": "http://python.org"}) is True
    
    # Non-allowlisted domain
    assert actuator.validate_params({"url": "https://malicious-site.com"}) is False
    # Invalid URL format
    assert actuator.validate_params({"url": "not-a-url"}) is False


def test_git_actuator_validation():
    """Verify action validation in GitActuator."""
    actuator = get_actuator_registry().get_actuator("git_operation")
    
    # Valid actions
    assert actuator.validate_params({"action": "status"}) is True
    assert actuator.validate_params({"action": "diff"}) is True
    assert actuator.validate_params({"action": "commit", "message": "test commit", "allow_mutation": True}) is True
    assert actuator.validate_params({"action": "clone", "url": "https://github.com/test/repo", "allow_external_clone": True}) is True
    
    # Invalid action / missing params
    assert actuator.validate_params({"action": "push"}) is False
    assert actuator.validate_params({"action": "commit"}) is False
    assert actuator.validate_params({"action": "commit", "message": "test commit"}) is False
    assert actuator.validate_params({"action": "clone"}) is False
    assert actuator.validate_params({"action": "clone", "url": "https://github.com/test/repo"}) is False


def test_package_install_validation():
    """Verify package name safety validation in PackageInstallActuator."""
    actuator = get_actuator_registry().get_actuator("package_install")
    
    # Valid package names
    assert actuator.validate_params({"package_name": "numpy", "allow_install": True}) is True
    assert actuator.validate_params({"package_name": "pandas>=1.0.0", "allow_install": True}) is True
    assert actuator.validate_params({"package_name": "scikit-learn", "allow_install": True}) is True
    
    # Invalid / dangerous package names (shell injection defense)
    assert actuator.validate_params({"package_name": "numpy; rm -rf /"}) is False
    assert actuator.validate_params({"package_name": "numpy && echo hacked"}) is False
    assert actuator.validate_params({"package_name": ""}) is False
    assert actuator.validate_params({"package_name": "numpy"}) is False


def test_privileged_actuators_refuse_direct_execution():
    """High-impact actuators must be run through ActuatorRegistry authority flow."""
    registry = get_actuator_registry()

    assert registry.get_actuator("code_execution").execute({"code": "print(1)"}).success is False
    assert registry.get_actuator("web_fetch").execute({"url": "https://wikipedia.org"}).success is False
    assert registry.get_actuator("git_operation").execute({"action": "status"}).success is False
    assert registry.get_actuator("package_install").execute({"package_name": "numpy", "allow_install": True}).success is False
    assert registry.get_actuator("process_supervisor").execute({"action": "list"}).success is False
