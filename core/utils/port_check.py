import socket
import time
import logging
import urllib.request

logger = logging.getLogger("Aura.Utils.PortCheck")

def is_port_open(port: int, host: str = "127.0.0.1") -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(1)
        try:
            s.connect((host, port))
            return True
        except (socket.timeout, ConnectionRefusedError):
            return False

def wait_for_port(port: int, host: str = "127.0.0.1", timeout: float = 30.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if is_port_open(port, host):
            return True
        time.sleep(0.5)
    return False

def is_http_ready(port: int, host: str = "127.0.0.1", path: str = "/api/health") -> bool:
    url = f"http://{host}:{port}{path}"
    try:
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=3) as resp:
            return resp.status < 500
    except Exception:
        return False

def wait_for_http(port: int, host: str = "127.0.0.1", path: str = "/api/health", timeout: float = 30.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if is_http_ready(port, host, path):
            return True
        time.sleep(1.0)
    return False
