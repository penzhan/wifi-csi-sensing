#!/usr/bin/env python3
"""Concatenate Pine captures and write frame-accurate golden labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from pine_data import read_capture


def parse_segment(value: str) -> tuple[str, Path]:
    try:
        label, path = value.split("=", 1)
    except ValueError as error:
        raise argparse.ArgumentTypeError("segment must be LABEL=PATH") from error
    return label.upper(), Path(path)


def write_capture(path: Path, capture: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output:
        for chain_index, chain in enumerate(capture):
            output.write(f"Chain {chain_index}:\n")
            np.savetxt(output, chain, fmt="%.8g")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--segment", action="append", type=parse_segment, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--period-ms", type=float, default=2.0)
    args = parser.parse_args()

    arrays = []
    segments = []
    start_frame = 0
    expected_layout = None
    for label, source_path in args.segment:
        capture = read_capture(source_path)
        layout = (capture.shape[0], capture.shape[2])
        if expected_layout is None:
            expected_layout = layout
        elif layout != expected_layout:
            raise ValueError(f"{source_path} layout {layout} differs from {expected_layout}")
        end_frame = start_frame + capture.shape[1]
        arrays.append(capture)
        segments.append(
            {
                "startFrame": start_frame,
                "endFrame": end_frame,
                "startSeconds": start_frame * args.period_ms / 1000.0,
                "endSeconds": end_frame * args.period_ms / 1000.0,
                "label": label,
                "source": source_path.name,
            }
        )
        start_frame = end_frame

    stream = np.concatenate(arrays, axis=1)
    write_capture(args.output, stream)
    manifest = {
        "periodMs": args.period_ms,
        "sampleRate": 1000.0 / args.period_ms,
        "totalFrames": int(stream.shape[1]),
        "chainCount": int(stream.shape[0]),
        "toneCount": int(stream.shape[2]),
        "segments": segments,
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"wrote {args.output}: {stream.shape}")
    print(f"wrote {args.manifest}: {len(segments)} golden-label segments")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
