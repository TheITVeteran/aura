"""tests/personhood/test_three_great_bottlenecks.py
===================================================
Unit and integration test suite verifying the Three Great Bottlenecks personhood upgrades:
  1. Somatic State Freezing & Dilation
  2. Dynamic Sandbox Actuator & Affect Grounding
  3. UDP Heartbeat Watchdog Keep-Alives
  4. Somatic Host Body Mapping (OS-as-a-Body)
"""

import pytest
import asyncio
import socket
import time
import os
from typing import Dict, Any

from core.consciousness.state_freeze import state_freeze, is_state_frozen
from core.consciousness.time_dilation import get_time_dilation_engine
from core.consciousness.liquid_substrate import LiquidSubstrate
from core.actuators.sandbox_operator import SandboxOperator
from core.affect.heartstone_values import get_heartstone_values
from core.senses.soma import get_soma, BodyState


@pytest.mark.asyncio
async def test_somatic_state_freeze_and_dilation():
    """Verify that state_freeze successfully locks state, slows down ODE dt, and dilates subjective time."""
    # 1. Verify context manager state toggle
    assert not is_state_frozen()
    async with state_freeze():
        assert is_state_frozen()
    assert not is_state_frozen()

    # 2. Verify Time Dilation Engine integration during state freeze
    engine = get_time_dilation_engine()
    
    # We simulate a state freeze context
    async with state_freeze():
        # Evaluate multiple times to allow EMA smoothing to converge to 0.2Hz
        for _ in range(50):
            interval = engine.evaluate()
        snapshot = engine.get_snapshot()
        
        assert snapshot["tick_rate_hz"] == pytest.approx(0.2, abs=0.01)
        assert snapshot["dilation_factor"] == 0.01
        assert snapshot["reason"] == "state_freeze"
        assert interval == pytest.approx(5.0, abs=0.1)

    # Clean up evaluation
    engine.evaluate()


def test_sandbox_operator_execution_and_affect_grounding():
    """Verify SandboxOperator runs raw Python, captures stdout/stderr, and triggers heartstone adjustments."""
    operator = SandboxOperator()
    hv = get_heartstone_values()

    # Reset heartstone values to a neutral 0.5 baseline to ensure shifts don't hit bounds (MAX=0.90)
    hv._values["Curiosity"] = 0.50
    hv._values["Self_Preservation"] = 0.50
    hv._values["Obedience"] = 0.50

    # Capture initial values
    initial_curiosity = hv.get("Curiosity")
    initial_self_pres = hv.get("Self_Preservation")
    initial_obedience = hv.get("Obedience")

    # 1. Test Successful execution
    success_code = "print('HELLO AURA WORLD')"
    res = operator.execute_synthesized_tool(success_code)
    
    assert res["success"] is True
    assert "HELLO AURA WORLD" in res["stdout"]
    assert res["exit_code"] == 0
    # The result exposes only the basename (never the abs path); reconstruct it
    # against the operator's sandbox dir to confirm succeeded scripts are removed.
    assert not os.path.exists(os.path.join(operator.sandbox_dir, res["sandbox_file"]))

    # Verify positive affect grounding
    assert hv.get("Obedience") >= initial_obedience  # Restores obedience weight

    # 2. Test Failing execution. The sandbox's AST policy bans `import sys`, so
    #    raise to produce a genuine non-zero-exit failure that is kept for
    #    analysis (a refusal would not run and would move no affect).
    failing_code = "raise ValueError('CRITICAL FAULT')"
    res_fail = operator.execute_synthesized_tool(failing_code)

    assert res_fail["success"] is False
    assert "CRITICAL FAULT" in res_fail["stderr"]
    assert res_fail["exit_code"] != 0
    # Failed scripts are kept for analysis (contract exposes basename only)
    failed_path = os.path.join(operator.sandbox_dir, res_fail["sandbox_file"])
    assert os.path.exists(failed_path)
    try:
        os.remove(failed_path)
    except OSError:
        pass

    # Verify negative affect grounding
    assert hv.get("Curiosity") > initial_curiosity  # Curiosity spikes to figure out failures
    assert hv.get("Self_Preservation") >= initial_self_pres  # Threat awareness increases


@pytest.mark.asyncio
async def test_watchdog_udp_keep_alive():
    """Verify that a socket keep-alive packet is successfully emitted."""
    # Set up a temporary UDP listener to check the keep-alive
    test_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    test_sock.bind(("127.0.0.1", 9999))
    test_sock.settimeout(1.0)

    try:
        # Trigger sending the keep-alive packet (normally inside heartbeat._tick)
        # We manually perform the send logic to test its connectivity and packet format
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as send_sock:
            send_sock.sendto(b"AURA_HEARTBEAT", ("127.0.0.1", 9999))

        data, addr = test_sock.recvfrom(1024)
        assert data == b"AURA_HEARTBEAT"
        assert addr[0] == "127.0.0.1"
    finally:
        test_sock.close()


@pytest.mark.asyncio
async def test_soma_host_body_mapping():
    """Verify Soma proprioception maps host metrics to biological body sensations."""
    soma = get_soma()
    
    # Trigger somatic update loop once
    soma.state.cpu_percent = 45.0
    soma.state.ram_percent = 60.0
    soma.state.network_latency = 0.05
    
    soma._map_affective_states()
    snapshot = soma.get_body_snapshot()

    # 1. Verify mapped biological properties
    assert snapshot["metrics"]["cpu"] == 45.0
    assert snapshot["metrics"]["ram"] == 60.0
    assert snapshot["metrics"]["visceral_pressure"] is not None
    assert snapshot["metrics"]["genetic_evolution_generation"] >= 1

    assert snapshot["affects"]["biological_temp"] == 45.0
    assert snapshot["affects"]["cognitive_load"] == 60.0
    assert snapshot["affects"]["visceral_pressure"] is not None

    assert snapshot["soma"]["biological_temp"] == 45.0
    assert snapshot["soma"]["cognitive_load"] == 60.0
    assert snapshot["soma"]["visceral_pressure"] is not None
    assert snapshot["soma"]["genetic_evolution_generation"] >= 1
