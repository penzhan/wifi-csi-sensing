#!/usr/bin/env python3
"""Serve the Pine PyTorch model through a small local HTTP inference API."""

from __future__ import annotations

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
from time import perf_counter
from urllib.parse import urlparse

import numpy as np
import torch

from infer_pine_live import (
    SharedChainPineLeNet,
    load_checkpoint,
    moving_average,
    normalize_per_chain,
    read_capture,
)


class PineInferenceService:
    def __init__(self, checkpoint_path: Path, data_root: Path) -> None:
        checkpoint = load_checkpoint(checkpoint_path)
        self.classes = [str(name) for name in checkpoint["classes"]]
        self.chain_count, self.time_length, self.tone_count = map(
            int, checkpoint["input_shape"]
        )
        config = checkpoint.get("config", {})
        self.fusion = str(config.get("fusion", "concat"))
        self.model = SharedChainPineLeNet(
            self.time_length,
            self.tone_count,
            len(self.classes),
            dropout=float(config.get("dropout", 0.3)),
            fusion=self.fusion,
        )
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
        self.model_name = str(checkpoint.get("model_name", "Pine model"))
        self.data_root = data_root.resolve()
        self.cache: dict[Path, tuple[int, np.ndarray]] = {}

    def capture(self, relative_path: str) -> np.ndarray:
        path = (self.data_root / relative_path).resolve()
        if self.data_root not in path.parents:
            raise ValueError("capture path leaves the configured data root")
        modified = path.stat().st_mtime_ns
        cached = self.cache.get(path)
        if cached is None or cached[0] != modified:
            cached = (modified, read_capture(path))
            self.cache[path] = cached
        return cached[1]

    def infer(self, capture_name: str, end_frame: int) -> dict[str, object]:
        capture = self.capture(capture_name)
        if end_frame < self.time_length:
            raise ValueError(f"need at least {self.time_length} captured frames")
        if end_frame > capture.shape[1]:
            raise ValueError(
                f"requested frame {end_frame}, capture contains {capture.shape[1]}"
            )
        window = capture[:, end_frame - self.time_length:end_frame]
        features = normalize_per_chain(moving_average(window))
        started = perf_counter()
        with torch.inference_mode():
            logits = self.model(torch.from_numpy(features).unsqueeze(0))
            probabilities = torch.softmax(logits, dim=1)[0].cpu().numpy()
        inference_ms = (perf_counter() - started) * 1000.0
        class_index = int(np.argmax(probabilities))
        return {
            "endFrame": end_frame,
            "label": self.classes[class_index].upper(),
            "confidence": round(float(probabilities[class_index]) * 100.0, 4),
            "probabilities": {
                name.upper(): round(float(value) * 100.0, 4)
                for name, value in zip(self.classes, probabilities)
            },
            "inferenceMs": round(inference_ms, 3),
            "model": self.model_name,
        }


def handler_class(service: PineInferenceService) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def cors(self) -> None:
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Max-Age", "3600")

        def send_json(self, status: int, payload: dict[str, object]) -> None:
            encoded = json.dumps(payload, separators=(",", ":")).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.cors()
            self.end_headers()
            self.wfile.write(encoded)

        def do_OPTIONS(self) -> None:  # noqa: N802
            self.send_response(204)
            self.cors()
            self.end_headers()

        def do_GET(self) -> None:  # noqa: N802
            if urlparse(self.path).path != "/health":
                self.send_json(404, {"error": "not found"})
                return
            self.send_json(
                200,
                {
                    "status": "ready",
                    "model": service.model_name,
                    "inputShape": [
                        service.chain_count,
                        service.time_length,
                        service.tone_count,
                    ],
                },
            )

        def do_POST(self) -> None:  # noqa: N802
            if urlparse(self.path).path != "/infer":
                self.send_json(404, {"error": "not found"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                request = json.loads(self.rfile.read(length))
                result = service.infer(
                    str(request["capture"]), int(request["endFrame"])
                )
            except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError) as error:
                self.send_json(400, {"error": str(error)})
                return
            self.send_json(200, result)

        def log_message(self, format_string: str, *args: object) -> None:
            print(f"{self.address_string()} - {format_string % args}", flush=True)

    return Handler


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    service = PineInferenceService(args.checkpoint, args.data_root)
    server = ThreadingHTTPServer((args.host, args.port), handler_class(service))
    print(
        f"Pine inference ready at http://{args.host}:{args.port} "
        f"with {service.model_name}",
        flush=True,
    )
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
