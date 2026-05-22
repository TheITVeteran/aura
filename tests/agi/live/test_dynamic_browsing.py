import http.server
import socket
import threading
from pathlib import Path

import pytest

from tools.agi.run_dynamic_browsing_task import run_browsing_task


class LocalHTTPServer:
    """A simple threaded local HTTP server to host dynamic test fixtures."""

    def __init__(self, port: int, root_dir: Path):
        self.port = port
        self.root_dir = root_dir
        self.server = None
        self.thread = None

    def start(self):
        root_dir_str = str(self.root_dir)
        class Handler(http.server.SimpleHTTPRequestHandler):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, directory=root_dir_str, **kwargs)

            def log_message(self, format, *args):
                return None

        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", self.port), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def stop(self):
        if self.server:
            self.server.shutdown()
            self.server.server_close()
        if self.thread:
            self.thread.join(timeout=2.0)


def get_free_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


@pytest.fixture
def local_web_server(tmp_path):
    # Set up index.html and doc.html in a temp directory
    index_content = """
    <html>
        <head><title>Aura Home</title></head>
        <body>
            <h1>Welcome to Aura Main Gate</h1>
            <p>Here is the portal for research.</p>
            <a id="docs-link" href="/doc.html">Aura Docs Portal</a>
        </body>
    </html>
    """
    doc_content = """
    <html>
        <head><title>Aura Documentation</title></head>
        <body>
            <h1>Aura Live Architecture</h1>
            <p>Authentication credentials verification successfully completed.</p>
            <p>Verification Key: AURA-LIVE-AGI-9921</p>
        </body>
    </html>
    """
    (tmp_path / "index.html").write_text(index_content)
    (tmp_path / "doc.html").write_text(doc_content)

    port = get_free_port()
    server = LocalHTTPServer(port, tmp_path)
    server.start()
    
    yield f"http://127.0.0.1:{port}"
    
    server.stop()


@pytest.mark.asyncio
async def test_live_phantom_browser_dynamic_navigation(local_web_server):
    # Run the dynamic browsing task using PhantomBrowser
    res = await run_browsing_task(
        start_url=local_web_server,
        target_link_text="Aura Docs Portal",
        expected_content_keywords=["AURA-LIVE-AGI-9921", "Authentication credentials"],
    )

    assert res["ok"] is True
    assert res["verification"]["AURA-LIVE-AGI-9921"] is True
    assert res["verification"]["Authentication credentials"] is True
    assert "Aura Documentation" in res["content_snippet"]
