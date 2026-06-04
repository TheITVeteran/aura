"""scripts/verify_pulse.py
Verification script for Phase 16: Distributed Awareness.
Simulates a peer Aura instance to test discovery.
"""
import socket
import json
import time
import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

async def main():
    port = 5555
    print(f"🌌 Starting Peer Simulator on port {port}...")

    # 1. Listen for Aura's Pulse
    listen_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    listen_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listen_sock.bind(('', port))
    listen_sock.setblocking(False)

    # 2. Setup Broadcast Socket
    broadcast_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    broadcast_sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    broadcast_sock.setblocking(False)

    print("🌌 Listening for Aura's pulse...")
    
    start_time = time.time()
    aura_detected = False
    
    # We run for 30 seconds
    while time.time() - start_time < 30:
        # A. Try to receive Aura's pulse
        try:
            data, addr = listen_sock.recvfrom(1024)
            pulse = json.loads(data.decode())
            if pulse.get("type") == "aura_pulse":
                print(f"🌌 DETECTED Aura Pulse: {pulse.get('id')} from {addr[0]}")
                aura_detected = True
        except BlockingIOError:
            pass
        except (json.JSONDecodeError, OSError, UnicodeDecodeError, ValueError) as e:
            print(f"Error receiving: {e}")

        # B. Send FAKE pulse to Aura
        fake_pulse = {
            "type": "aura_pulse",
            "id": "aura-simulator-node",
            "status": "active",
            "timestamp": time.time()
        }
        broadcast_sock.sendto(json.dumps(fake_pulse).encode(), ('<broadcast>', port))
        
        await asyncio.sleep(2)

    if aura_detected:
        print("✅ SUCCESS: Aura's distributed pulse detected.")
    else:
        print("❌ FAILED: No pulse detected from Aura. Is the orchestrator running?")

    listen_sock.close()
    broadcast_sock.close()
    return aura_detected

if __name__ == "__main__":
    try:
        sys.exit(0 if asyncio.run(main()) else 1)
    except KeyboardInterrupt:
        pass
