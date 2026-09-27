"""Local HTTP ``/metrics`` endpoint for Prometheus. Started only on request."""

from __future__ import annotations

import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from quant_fund.observe.metrics import render_prometheus

_SERVER: ThreadingHTTPServer | None = None
_THREAD: threading.Thread | None = None


class _Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802 - stdlib handler name
        if self.path.split("?", 1)[0] != "/metrics":
            self.send_response(404)
            self.end_headers()
            return
        body = render_prometheus().encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; version=0.0.4; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:
        return


def start_metrics_server(port: int = 9464, host: str = "127.0.0.1") -> int:
    """Bind ``host:port`` and serve ``/metrics``. Returns the actual port."""
    global _SERVER, _THREAD
    if _SERVER is not None:
        return int(_SERVER.server_address[1])
    server = ThreadingHTTPServer((host, port), _Handler)
    thread = threading.Thread(target=server.serve_forever, name="dipcatcher-metrics", daemon=True)
    thread.start()
    _SERVER = server
    _THREAD = thread
    return int(server.server_address[1])


def stop_metrics_server() -> None:
    global _SERVER, _THREAD
    server = _SERVER
    thread = _THREAD
    _SERVER = None
    _THREAD = None
    if server is not None:
        server.shutdown()
        server.server_close()
    if thread is not None:
        thread.join(timeout=2)
