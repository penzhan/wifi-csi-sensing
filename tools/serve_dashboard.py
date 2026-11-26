#!/usr/bin/env python3
"""Serve the CSI demo locally without caching generated dashboard assets."""

from __future__ import annotations

import argparse
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import http.client
from pathlib import Path
from threading import Lock


INFERENCE_CONNECTION = None
INFERENCE_LOCK = Lock()


class NoCacheHandler(SimpleHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def end_headers(self) -> None:
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        super().end_headers()

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/api/infer":
            self.send_error(404)
            return
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length)
        global INFERENCE_CONNECTION
        with INFERENCE_LOCK:
            for attempt in range(2):
                try:
                    if INFERENCE_CONNECTION is None:
                        INFERENCE_CONNECTION = http.client.HTTPConnection(
                            "127.0.0.1", 8765, timeout=30
                        )
                    INFERENCE_CONNECTION.request(
                        "POST",
                        "/infer",
                        body=body,
                        headers={"Content-Type": "application/json"},
                    )
                    response = INFERENCE_CONNECTION.getresponse()
                    payload = response.read()
                    status = response.status
                    break
                except (ConnectionError, OSError, http.client.HTTPException):
                    INFERENCE_CONNECTION = None
                    if attempt:
                        raise
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    handler = partial(NoCacheHandler, directory=str(args.root.resolve()))
    server = ThreadingHTTPServer((args.host, args.port), handler)
    print(f"Dashboard: http://{args.host}:{args.port}/csi_demo/out/dashboard.html")
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
