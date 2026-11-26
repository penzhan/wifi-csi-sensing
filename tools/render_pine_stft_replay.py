#!/usr/bin/env python3
"""Render a progressively revealed STFT GIF from a CSI Demo scene."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np

from plot_pine_stft import DEFAULT_OUTDIR, dashboard_payload


def render_replay(
    scene: dict[str, object],
    output: Path,
    *,
    playback_fps: int,
) -> None:
    first = int(scene["firstDataColumn"])
    times = np.asarray(scene["stftTimes"], dtype=float)
    frequencies = np.asarray(scene["frequencies"], dtype=float)
    spectrum = np.asarray(scene["stftDb"], dtype=float)
    stft = scene["stft"]
    window_columns = int(scene["displayStftWindow"])

    figure, axis = plt.subplots(figsize=(11.5, 5.2), constrained_layout=True)

    def draw(column: int) -> None:
        axis.clear()
        start = max(first, column - window_columns + 1)
        visible_times = times[start:column + 1]
        visible = spectrum[:, start:column + 1]
        axis.pcolormesh(
            visible_times,
            frequencies,
            visible,
            shading="auto",
            cmap="turbo",
            vmin=float(stft["dbMin"]),
            vmax=float(stft["dbMax"]),
        )
        axis.set_xlim(times[first], times[-1])
        axis.set_ylim(frequencies[0], frequencies[-1])
        axis.set_title(f"{scene['label']} — Live CFR STFT · {times[column]:.2f} s")
        axis.set_xlabel("Time (s)")
        axis.set_ylabel("Frequency shift (Hz)")
        axis.grid(color="white", alpha=0.13, linewidth=0.7)

    frames = range(first, len(times))
    animation = FuncAnimation(figure, draw, frames=frames, interval=1000 / playback_fps)
    output.parent.mkdir(parents=True, exist_ok=True)
    animation.save(output, writer=PillowWriter(fps=playback_fps), dpi=120)
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", default="live", help="live, run, walk, pick, lie, fall, stand, or sit")
    parser.add_argument("--fps", type=int, default=12, help="GIF playback frame rate")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    payload = dashboard_payload()
    if args.scene not in payload["scenes"]:
        parser.error(f"unknown scene: {args.scene}")
    scene = payload["scenes"][args.scene]
    output = args.output or DEFAULT_OUTDIR / f"{args.scene}_fused_cfr_stft_replay.gif"
    render_replay(scene, output, playback_fps=max(1, args.fps))
    print(output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
