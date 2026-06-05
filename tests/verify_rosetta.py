from core.adaptation.rosetta_stone import RosettaStone


def test_rosetta_adapts_windows_shell_verbs_without_partial_rewrites():
    rosetta = RosettaStone()

    assert rosetta.adapt_command("ls -la", target_os="windows") == "dir -la"
    assert rosetta.adapt_command("grep needle file.txt", target_os="windows") == "findstr needle file.txt"
    assert rosetta.adapt_command("classify logs", target_os="windows") == "classify logs"


def test_rosetta_blocks_destructive_and_dynamic_execution_patterns():
    rosetta = RosettaStone()

    destructive = rosetta.analyze_threat("rm -rf /")
    assert destructive["safe"] is False
    assert "Root Deletion Attempt" in destructive["threats"]
    assert "Sandbox Isolation" in destructive["countermeasures"]

    dynamic = rosetta.analyze_threat("eval(\"__import__('os').system('rm -rf /')\")")
    assert dynamic["safe"] is False
    assert "Dynamic Execution / Obfuscation" in dynamic["threats"]


def test_rosetta_allows_plain_safe_command_text():
    rosetta = RosettaStone()

    report = rosetta.analyze_threat("echo hello && pwd")

    assert report == {"safe": True, "threats": [], "countermeasures": []}
