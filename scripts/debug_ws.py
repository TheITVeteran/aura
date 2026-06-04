import asyncio
import json
import os

import websockets
from websockets.exceptions import WebSocketException

MAX_MESSAGES = int(os.getenv("AURA_DEBUG_WS_MAX_MESSAGES", "100"))


async def test_chat():
    uri = "ws://localhost:8000/ws/chat"
    # We might need authentication if enabled
    # But usually localhost is allowed if AURA_ALLOW_LOCALHOST_ONLY=1
    
    print(f"Connecting to {uri}...")
    try:
        async with websockets.connect(uri) as websocket:
            print("Connected!")
            
            msg = {"type": "message", "message": "Hello Aura"}
            await websocket.send(json.dumps(msg))
            print(f"Sent: {msg}")

            for _ in range(MAX_MESSAGES):
                try:
                    response = await asyncio.wait_for(websocket.recv(), timeout=10.0)
                    data = json.loads(response)
                    print(f"Received: {data}")
                    if data.get("type") == "done":
                        print("Stream finished (done received).")
                        break
                    
                    if data.get("type") == "done":
                        break
                    if data.get("type") == "error":
                        print("Error received!")
                        break
                except TimeoutError:
                    print("Timeout waiting for response.")
                    break
                except websockets.exceptions.ConnectionClosed as e:
                    print(f"Connection closed by server: {e.code} {e.reason}")
                    break
                except (json.JSONDecodeError, OSError, RuntimeError, ValueError, WebSocketException) as e:
                    print(f"Error in loop: {type(e)} {e}")
                    break
            else:
                print(f"Stopped after {MAX_MESSAGES} messages without a terminal frame.")
                    
    except (OSError, RuntimeError, WebSocketException) as e:
        print(f"Connection failed: {type(e)} {e}")

if __name__ == "__main__":
    asyncio.run(test_chat())
