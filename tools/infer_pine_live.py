#!/usr/bin/env python3
"""Run sliding-window Pine inference and export browser-friendly JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path, PosixPath

import numpy as np
import torch
import torch.nn as nn

from pine_data import read_capture


class SharedChainPineLeNet(nn.Module):
    """Shared per-chain encoder used by pine_shared_lenet_concat.pt."""

    def __init__(
        self,
        time_length: int,
        tone_count: int,
        class_count: int,
        dropout: float = 0.3,
        fusion: str = "concat",
    ) -> None:
        super().__init__()
        self.time_length = time_length
        self.tone_count = tone_count
        self.fusion = fusion
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.AvgPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AvgPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.embedding = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 4 * 4, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        classifier_input = 64 if fusion == "mean" else 64 * 4
        self.classifier = nn.Linear(classifier_input, class_count)

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        batch_size, chain_count = data.shape[:2]
        data = data.reshape(
            batch_size * chain_count,
            1,
            self.time_length,
            self.tone_count,
        )
        data = self.embedding(self.encoder(data))
        data = data.reshape(batch_size, chain_count, -1)
        if self.fusion == "mean":
            data = data.mean(dim=1)
        else:
            data = data.reshape(batch_size, -1)
        return self.classifier(data)


def moving_average(data: np.ndarray, width: int = 5) -> np.ndarray:
    left = (width - 1) // 2
    right = width - 1 - left
    padded = np.pad(data, ((0, 0), (left, right), (0, 0)), mode="reflect")
    cumulative = np.concatenate(
        (np.zeros_like(padded[:, :1]), np.cumsum(padded, axis=1, dtype=np.float32)),
        axis=1,
    )
    return ((cumulative[:, width:] - cumulative[:, :-width]) / width).astype(np.float32)


def normalize_per_chain(data: np.ndarray) -> np.ndarray:
    mean = data.mean(axis=(1, 2), keepdims=True, dtype=np.float64).astype(np.float32)
    deviation = data.std(axis=(1, 2), keepdims=True, dtype=np.float64).astype(np.float32)
    deviation[deviation <= np.finfo(np.float32).eps] = 1.0
    return ((data - mean) / deviation).astype(np.float32)


def load_checkpoint(path: Path) -> dict[str, object]:
    try:
        safe_globals = getattr(torch.serialization, "safe_globals", None)
        if safe_globals is None:
            return torch.load(path, map_location="cpu", weights_only=True)
        with safe_globals([PosixPath]):
            return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--period-ms", type=float, default=2.0)
    parser.add_argument("--hop", type=int, default=8)
    args = parser.parse_args()

    checkpoint = load_checkpoint(args.checkpoint)
    classes = [str(name) for name in checkpoint["classes"]]
    chain_count, time_length, tone_count = map(int, checkpoint["input_shape"])
    config = checkpoint.get("config", {})
    fusion = str(config.get("fusion", "concat"))
    dropout = float(config.get("dropout", 0.3))
    model = SharedChainPineLeNet(
        time_length,
        tone_count,
        len(classes),
        dropout=dropout,
        fusion=fusion,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    capture = read_capture(args.input)
    if capture.shape[0] != chain_count or capture.shape[2] != tone_count:
        raise ValueError(
            f"checkpoint expects {(chain_count, tone_count)}, "
            f"capture provides {(capture.shape[0], capture.shape[2])}"
        )
    if capture.shape[1] < time_length:
        raise ValueError(
            f"checkpoint needs {time_length} frames, capture has {capture.shape[1]}"
        )

    predictions = []
    with torch.inference_mode():
        for end_frame in range(time_length, capture.shape[1] + 1, args.hop):
            window = capture[:, end_frame - time_length:end_frame]
            features = normalize_per_chain(moving_average(window))
            logits = model(torch.from_numpy(features).unsqueeze(0))
            probabilities = torch.softmax(logits, dim=1)[0].cpu().numpy()
            class_index = int(np.argmax(probabilities))
            predictions.append(
                {
                    "endFrame": end_frame,
                    "timeSeconds": round(end_frame * args.period_ms / 1000.0, 6),
                    "label": classes[class_index].upper(),
                    "confidence": round(float(probabilities[class_index]) * 100.0, 4),
                    "probabilities": {
                        name.upper(): round(float(value) * 100.0, 4)
                        for name, value in zip(classes, probabilities)
                    },
                }
            )

    result = {
        "model": checkpoint.get("model_name"),
        "checkpoint": args.checkpoint.name,
        "classes": classes,
        "inputShape": [chain_count, time_length, tone_count],
        "fusion": fusion,
        "periodMs": args.period_ms,
        "hopFrames": args.hop,
        "windowFrames": time_length,
        "windowSeconds": time_length * args.period_ms / 1000.0,
        "captureFrames": int(capture.shape[1]),
        "predictions": predictions,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(
        f"wrote {args.output}: {len(predictions)} predictions, "
        f"window={result['windowSeconds']:.3f}s"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
