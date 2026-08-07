import json
import sys

from core.body.sensor_registry import BaseSensor
from core.learning.eval_before_promotion import AdapterEvaluator
from core.learning.trace_labeler import TraceLabeler
from core.security.sandbox import LocalCommandSandbox
from core.sleep.world_model_training import WorldModelTrainer
from environments.social_world.social_simulator import VirtualSocialWorld


def test_local_command_sandbox_actually_isolates(tmp_path):
    """It used to pass `cwd` and call itself a sandbox.

    The previous version of this test asserted exactly that: one subprocess
    gateway call carrying `cwd=sandbox_dir`. Both the code and the test were
    self-consistent, and neither was about isolation — the command could read
    any file, write outside the directory with one absolute path, and open a
    socket. What is asserted here instead is the property: the command runs
    under the OS-enforced sandbox, and the receipt says which kind of
    enforcement it got.
    """
    result = LocalCommandSandbox().execute_sandboxed_command(
        [sys.executable, "-c", "print('ok')"], str(tmp_path)
    )

    assert result["exit_code"] == 0
    assert "ok" in result["stdout"]
    assert result["sandboxed"] is True
    if sys.platform == "darwin":
        # A seatbelt profile that denies network and confines writes.
        assert result["kernel_enforced"] is True


def test_local_command_sandbox_refuses_rather_than_running_unsandboxed(
    monkeypatch, tmp_path
):
    """No silent degradation to a plain subprocess.

    Falling back to an unsandboxed run would reinstate the original defect on
    exactly the path where isolation had already failed once.
    """
    monkeypatch.setitem(sys.modules, "security.sandbox", None)

    result = LocalCommandSandbox().execute_sandboxed_command(
        [sys.executable, "-c", "print('should not run')"], str(tmp_path)
    )

    assert result["exit_code"] == -1
    assert result["sandboxed"] is False
    assert "refused" in result["error"]


def test_local_command_sandbox_enforces_its_allowlist(tmp_path):
    """RESTRICTED permits python; it does not permit an arbitrary binary."""
    result = LocalCommandSandbox().execute_sandboxed_command("curl https://x", str(tmp_path))

    assert result["exit_code"] != 0
    assert result["security_violations"]


def test_adapter_promotion_requires_real_evaluation_artifact(tmp_path):
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()

    blocked = AdapterEvaluator().evaluate_candidate(str(adapter_dir))

    assert blocked["can_promote"] is False
    assert blocked["reason"] == "evaluation_report_missing"

    (adapter_dir / "evaluation_report.json").write_text(
        json.dumps(
            {
                "passed_safety": True,
                "regression_passed": True,
                "hidden_eval_passed": True,
                "accuracy_score": 0.86,
                "promotion_threshold": 0.8,
            }
        ),
        encoding="utf-8",
    )

    evaluated = AdapterEvaluator().evaluate_candidate(str(adapter_dir))

    assert evaluated["status"] == "evaluated"
    assert evaluated["can_promote"] is True
    assert evaluated["evidence_path"].endswith("evaluation_report.json")


def test_trace_labeler_rejects_non_observed_success():
    labeler = TraceLabeler()

    labeled = labeler.label_sample({"task": "repair"}, {"status": "simulated", "effect_verified": True})

    assert labeled["success"] is False
    assert labeled["trainable"] is False

    inhibited = labeler.label_sample({"task": "unsafe"}, {"status": "inhibited"})
    assert inhibited["success"] is False
    assert inhibited["inhibited_safely"] is True
    assert inhibited["trainable"] is True


def test_world_model_training_reinforces_observed_edges():
    world_model = {
        "causal_edges": {
            "screen_to_action": {
                "observations": 4,
                "confidence": 0.5,
            }
        }
    }

    WorldModelTrainer().train_world_model(world_model)

    training = world_model["_sleep_world_model_training"]
    edge = world_model["causal_edges"]["screen_to_action"]
    assert training["cycles"] == 1
    assert training["edges_reinforced"] == 1
    assert edge["confidence"] > 0.5
    assert edge["edge_id"] == "screen_to_action"


def test_virtual_social_world_records_outgoing_messages():
    world = VirtualSocialWorld()

    sent = world.send_agent_message("I can take that on.")

    assert sent["sender"] == "Aura"
    assert sent["message"] == "I can take that on."
    assert world.sent_messages() == [sent]


def test_base_sensor_derives_stable_name():
    class AmbientLightSensor(BaseSensor):
        pass

    assert AmbientLightSensor().name == "ambient_light"
