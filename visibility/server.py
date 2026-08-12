"""Small local server for the read-only visibility dashboard."""

from __future__ import annotations

import argparse
import json
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, status_path: Path, **kwargs):
        self.status_path = status_path
        super().__init__(*args, directory=str(Path(__file__).parent), **kwargs)

    def do_GET(self):  # noqa: N802
        if self.path == "/api/status":
            try:
                payload = self.status_path.read_text()
            except FileNotFoundError:
                payload = json.dumps({"project": "azure-mlops", "environment": "dev", "warnings": ["No snapshot found. Run collect_status.py first."], "workflow_runs": [], "training_runs": [], "endpoints": [], "monitoring": {}})
            body = payload.encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        super().do_GET()


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve the read-only MLOps dashboard")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--status", type=Path, default=Path("visibility/status.json"))
    args = parser.parse_args()
    try:
        server = ThreadingHTTPServer(("127.0.0.1", args.port), lambda *a, **kw: Handler(*a, status_path=args.status, **kw))
    except OSError as exc:
        raise SystemExit(f"Could not bind port {args.port}: {exc}. Try --port 8766 or another unused port.") from exc
    print(f"Dashboard available at http://127.0.0.1:{args.port}/dashboard.html")
    server.serve_forever()


if __name__ == "__main__":
    main()
