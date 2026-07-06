"""The general reconstruction sandbox must be BOTH general (real stdlib, real
programs verifiable) AND safe (no filesystem / network / process / eval / gadget
escape). These tests pin both properties — a sandbox that is only one is useless."""
from __future__ import annotations

import pytest

from core.discovery.reconstruction_sandbox import (
    GeneralReconstructionEvaluator,
    ReconstructionASTViolation,
    audit_general_ast,
)

# ── generality: realistic reconstructions verify ──────────────────────────

_BASE64 = (
    "import base64\n"
    "def reconstructed(case):\n"
    "    return base64.b64encode(case['text'].encode()).decode() + '\\n'\n"
)
_MD5 = (
    "import hashlib\n"
    "def reconstructed(case):\n"
    "    return hashlib.md5(case['text'].encode()).hexdigest()\n"
)


def test_realistic_program_with_imports_and_attributes_passes():
    ev = GeneralReconstructionEvaluator(timeout_seconds=5.0)
    result = ev.evaluate(_BASE64, "reconstructed", [(({"text": "hello"},), "aGVsbG8=\n")])
    assert result.outcome == "passed", result.error


def test_wrong_reconstruction_is_caught():
    ev = GeneralReconstructionEvaluator(timeout_seconds=5.0)
    # md5 impl but expected a sha1 digest → must NOT pass
    result = ev.evaluate(_MD5, "reconstructed", [(({"text": "hello"},), "deadbeef")])
    assert result.outcome != "passed"


def test_attribute_methods_and_stdlib_allowed():
    audit_general_ast("import math\ndef f(c):\n return math.sqrt(c['x'])\n")
    audit_general_ast("def f(c):\n return c['t'].upper().split(',')\n")  # attribute methods ok


# ── safety: escapes are blocked ───────────────────────────────────────────

@pytest.mark.parametrize(
    "code",
    [
        "import os\ndef f(c):\n return os.getcwd()\n",
        "import sys\ndef f(c):\n return sys.argv\n",
        "import subprocess\ndef f(c):\n return subprocess.run(['ls'])\n",
        "import socket\ndef f(c):\n return socket.socket()\n",
        "import shutil\ndef f(c):\n return shutil.rmtree('/')\n",
        "import importlib\ndef f(c):\n return importlib.import_module('os')\n",
        "import ctypes\ndef f(c):\n return ctypes.CDLL('libc')\n",
        "import pickle\ndef f(c):\n return pickle.loads(c['b'])\n",
        "import codecs\ndef f(c):\n return codecs.open('/etc/passwd')\n",  # the hole we closed
        "from os import system\ndef f(c):\n return system('id')\n",
        "import urllib.request\ndef f(c):\n return urllib.request.urlopen('http://x')\n",
    ],
)
def test_dangerous_imports_blocked(code):
    with pytest.raises(ReconstructionASTViolation):
        audit_general_ast(code)


@pytest.mark.parametrize(
    "code",
    [
        "def f(c):\n return open('/etc/passwd').read()\n",
        "def f(c):\n return eval(c['x'])\n",
        "def f(c):\n return exec(c['x'])\n",
        "def f(c):\n return __import__('os').system('id')\n",
        "def f(c):\n return getattr(c, '__class__')\n",
        "def f(c):\n return globals()\n",
        "def f(c):\n return ().__class__.__bases__\n",
        "def f(c):\n return type(c).__subclasses__()\n",
        "def f(c):\n return c.__globals__\n",
    ],
)
def test_dangerous_calls_and_gadgets_blocked(code):
    with pytest.raises(ReconstructionASTViolation):
        audit_general_ast(code)


def test_escape_attempt_returns_ast_violation_not_passed():
    # end-to-end: even wrapped as a candidate, an escape never executes
    ev = GeneralReconstructionEvaluator(timeout_seconds=5.0)
    escape = "import os\ndef reconstructed(case):\n return os.getcwd()\n"
    result = ev.evaluate(escape, "reconstructed", [(({"text": "x"},), "/")])
    assert result.outcome == "ast_violation"
    assert result.passed == 0
