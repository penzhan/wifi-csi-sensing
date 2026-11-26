#!/usr/bin/env python3
"""Plot static STFT spectra using the exact CSI Demo dashboard pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from dashboard import EXAMPLE_SPECS, build_payload


TOOLS_DIR = Path(__file__).resolve().parent
DEFAULT_OUTDIR = TOOLS_DIR.parent / "out/plots"


def dashboard_payload() -> dict[str, object]:
    paths = {
        key: [Path(path).resolve() for path in spec["paths"]]
        for key, spec in EXAMPLE_SPECS.items()
    }
    return build_payload(paths)


def plot_scene(scene: dict[str, object], output: Path) -> None:
    first = int(scene["firstDataColumn"])
    times = np.asarray(scene["stftTimes"], dtype=float)[first:]
    frequencies = np.asarray(scene["frequencies"], dtype=float)
    spectrum = np.asarray(scene["stftDb"], dtype=float)[:, first:]
    stft = scene["stft"]

    figure, axis = plt.subplots(figsize=(11.5, 5.2), constrained_layout=True)
    image = axis.pcolormesh(
        times,
        frequencies,
        spectrum,
        shading="auto",
        cmap="turbo",
        vmin=float(stft["dbMin"]),
        vmax=float(stft["dbMax"]),
    )
    axis.set_title(f"{scene['label']} — Fused CFR Motion STFT")
    axis.set_xlabel("Time (s)")
    axis.set_ylabel("Frequency shift (Hz)")
    axis.grid(color="white", alpha=0.13, linewidth=0.7)
    colorbar = figure.colorbar(image, ax=axis, pad=0.015)
    colorbar.set_label("Relative power (dB)")

    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scene",
        choices=("all", "live", *EXAMPLE_SPECS),
        default="live",
        help="scene to plot; live uses the cfr_40 capture without exposing its label",
    )
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    args = parser.parse_args()

    payload = dashboard_payload()
    keys = payload["scenes"] if args.scene == "all" else (args.scene,)
    for key in keys:
        scene = payload["scenes"][key]
        output = args.outdir / f"{key}_fused_cfr_stft.png"
        plot_scene(scene, output)
        print(output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
