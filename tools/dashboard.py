#!/usr/bin/env python3
"""Generate a self-contained Wi-Fi CSI/CFR sensing dashboard.

The built-in Pine recordings are presented as toy examples.  Each example
treats two captures as one continuous packet stream:
5-packet smoothing, per-chain normalization, 128-packet Hann STFT with an
8-packet hop, linear-power fusion over tones/chains, and one dB reference
shared by every Pine test capture. Pine contains magnitude only, so phase is
never synthesized.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


CSI_DEMO_ROOT = Path(__file__).resolve().parents[1]
PINE_ROOT = CSI_DEMO_ROOT / "examples/pine"
MODEL_ROOT = CSI_DEMO_ROOT / "models"
EXAMPLE_SPECS = {
    "run": {
        "dataset_key": "pine",
        "dataset_label": "Pine",
        "activity": "Run",
        "prediction": "RUN",
        "paths": (
            PINE_ROOT / "run/cfr_29.txt",
            PINE_ROOT / "run/cfr_37.txt",
        ),
    },
    "walk": {
        "dataset_key": "pine",
        "dataset_label": "Pine",
        "activity": "Walk",
        "prediction": "WALK",
        "paths": (
            PINE_ROOT / "walk/cfr_20.txt",
            PINE_ROOT / "walk/cfr_34.txt",
        ),
    },
    "pick": {
        "dataset_key": "pine",
        "dataset_label": "Pine",
        "activity": "Pick",
        "prediction": "PICK",
        "paths": (
            PINE_ROOT / "pick/cfr_39.txt",
            PINE_ROOT / "pick/cfr_45.txt",
        ),
    },
    "lie": {
        "dataset_key": "pine",
        "dataset_label": "Pine",
        "activity": "Lie",
        "prediction": "LIE",
        "paths": (
            PINE_ROOT / "lie/cfr_37.txt",
            PINE_ROOT / "lie/cfr_46.txt",
        ),
    },
    "fall": {
        "dataset_key": "pine",
        "dataset_label": "Pine",
        "activity": "Fall",
        "prediction": "FALL",
        "paths": (
            PINE_ROOT / "fall/cfr_36.txt",
            PINE_ROOT / "fall/cfr_41.txt",
        ),
    },
    "stand": {
        "dataset_key": "pine",
        "dataset_label": "Pine",
        "activity": "Stand",
        "prediction": "STAND",
        "paths": (
            PINE_ROOT / "stand/cfr_28.txt",
            PINE_ROOT / "stand/cfr_40.txt",
        ),
    },
    "sit": {
        "dataset_key": "pine",
        "dataset_label": "Pine",
        "activity": "Sit",
        "prediction": "SIT",
        "paths": (
            PINE_ROOT / "sit/cfr_38.txt",
            PINE_ROOT / "sit/cfr_49.txt",
        ),
    },
}
DEFAULT_TEMPLATE = Path(__file__).with_name("dashboard.tmpl.html")
DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "out/dashboard.html"

PACKETS = 600
SAMPLE_RATE = 100.0
SMOOTHING = 5
WINDOW_LENGTH = 128
HOP = 8
NFFT = 128
FIRST_TONE = 38
LAST_TONE = 91
DB_MIN = -35.0
DB_MAX = 0.0


def read_pine_magnitude(path: Path) -> dict[int, list[list[float]]]:
    """Read a Pine ``Chain N:`` magnitude capture."""
    rows: dict[int, list[list[float]]] = {}
    chain: int | None = None
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            text = line.strip()
            if not text:
                continue
            if text.startswith("Chain ") and text.endswith(":"):
                chain = int(text[6:-1])
                rows[chain] = []
                continue
            if chain is None:
                raise ValueError(f"data row before chain header at line {line_number}")
            rows[chain].append([float(value) for value in text.split()])
    if not rows:
        raise ValueError(f"no Pine chains found in {path}")
    return rows


def active_array(path: Path, packets: int | None = PACKETS) -> np.ndarray:
    """Return packet x chain x 53-active-tone magnitude."""
    matrices = read_pine_magnitude(path)
    chains = sorted(matrices)
    available = min(len(matrices[index]) for index in chains)
    if packets is not None and available < packets:
        raise ValueError(f"{path} has {available} packets; need {packets}")
    packet_slice = slice(None) if packets is None else slice(0, packets)
    return np.stack(
        [
            np.asarray(matrices[index], dtype=np.float64)[
                packet_slice, FIRST_TONE:LAST_TONE
            ]
            for index in chains
        ],
        axis=1,
    )


def smooth_time(data: np.ndarray, width: int = SMOOTHING) -> np.ndarray:
    """Centered, reflect-padded smoothing along packet time."""
    width = max(1, min(int(width), len(data)))
    left = (width - 1) // 2
    right = width - 1 - left
    padded = np.pad(data, ((left, right), (0, 0), (0, 0)), mode="reflect")
    cumulative = np.concatenate(
        (np.zeros_like(padded[:1]), np.cumsum(padded, axis=0, dtype=np.float64)),
        axis=0,
    )
    return (cumulative[width:] - cumulative[:-width]) / width


def normalize_per_chain(smoothed: np.ndarray) -> np.ndarray:
    """Normalize each chain once over all packets and active tones."""
    mean = np.mean(smoothed, axis=(0, 2), keepdims=True)
    deviation = np.std(smoothed, axis=(0, 2), keepdims=True)
    deviation[deviation <= np.finfo(float).eps] = 1.0
    return (smoothed - mean) / deviation


def fused_stft_power(data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """STFT each chain/tone, then average linear power over tones/chains."""
    starts = np.arange(0, len(data) - WINDOW_LENGTH + 1, HOP)
    window = np.hanning(WINDOW_LENGTH).reshape(-1, 1, 1)
    columns = []
    for start in starts:
        transformed = np.fft.rfft(
            data[start:start + WINDOW_LENGTH] * window,
            n=NFFT,
            axis=0,
        )
        tone_power = np.abs(transformed) ** 2
        chain_power = np.mean(tone_power, axis=2)
        columns.append(np.mean(chain_power, axis=1))
    return np.column_stack(columns), starts + WINDOW_LENGTH / 2


def capture_power_array(data: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Prepare one continuous packet array and calculate its STFT."""
    smoothed = smooth_time(data)
    power, centers = fused_stft_power(normalize_per_chain(smoothed))
    return smoothed, power, centers


def capture_power(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calculate one capture for the shared Pine test reference."""
    return capture_power_array(active_array(path))


def shared_test_reference(test_root: Path) -> float:
    """Return the non-DC peak over all bundled Pine example captures."""
    paths = sorted(test_root.glob("*/*.txt"))
    if not paths:
        raise ValueError(f"no Pine test captures under {test_root}")
    reference = 0.0
    for path in paths:
        _, power, _ = capture_power(path)
        reference = max(reference, float(np.max(power[1:])))
    if reference <= 0:
        raise ValueError("invalid shared STFT reference")
    return reference


def shared_reference_db(power: np.ndarray, reference: float) -> np.ndarray:
    result = 10.0 * np.log10(np.maximum(power, np.finfo(float).tiny) / reference)
    result[0] = DB_MIN
    return np.clip(result, DB_MIN, DB_MAX)


def build_scene(
    input_paths: list[Path],
    *,
    key: str,
    label: str,
    dataset_key: str,
    dataset_label: str,
    activity: str,
    capture_label: str,
    prediction: str,
    source: str,
    reference: float,
    is_live: bool = False,
    sample_rate: float = SAMPLE_RATE,
    packet_limit: int | None = PACKETS,
) -> dict[str, object]:
    if not input_paths:
        raise ValueError("at least one input capture is required")

    # Concatenate packets before smoothing/normalization/STFT.  This makes the
    # second capture continue on the same time axis instead of restarting it.
    stream = np.concatenate(
        [active_array(path, packets=packet_limit) for path in input_paths], axis=0
    )
    smoothed, power, centers = capture_power_array(stream)
    displayed = shared_reference_db(power, reference)
    amplitude = np.sqrt(np.mean(np.square(smoothed), axis=1))
    frequencies = np.fft.rfftfreq(NFFT, d=1.0 / sample_rate)
    # Start the replay with one genuinely empty frame at 0.00 s.  The first
    # calculated STFT column appears on the next playback tick (0.08 s), while
    # its original packet-center index remains available for frame lookup.
    if is_live:
        # Simulate capture from frame zero.  Blank hop columns are emitted
        # until the first complete Hann window becomes available.
        display_time_frames = np.arange(0, len(stream) + 1, HOP, dtype=int)
        blank_count = len(display_time_frames) - displayed.shape[1]
        blank = np.full((displayed.shape[0], blank_count), DB_MIN)
        displayed = np.column_stack((blank, displayed))
        first_data_column = blank_count
        frame_centers = np.minimum(display_time_frames, len(stream) - 1)
    else:
        pre_roll_centers = np.asarray([0], dtype=int)
        blank = np.full((displayed.shape[0], 1), DB_MIN)
        displayed = np.column_stack((blank, displayed))
        display_time_frames = np.arange(len(displayed.T), dtype=int) * HOP
        frame_centers = np.concatenate((pre_roll_centers, centers))
        first_data_column = len(pre_roll_centers)
    motion_metric = np.mean((displayed[1:] - DB_MIN) / (DB_MAX - DB_MIN), axis=0)

    return {
        "key": key,
        "label": label,
        "datasetKey": dataset_key,
        "datasetLabel": dataset_label,
        "activity": activity,
        "captureLabel": capture_label,
        "prediction": prediction,
        "source": source,
        "isLive": is_live,
        "inferenceWindowSeconds": 2.0,
        "sampleRate": sample_rate,
        "chainCount": int(smoothed.shape[1]),
        "toneCount": int(smoothed.shape[2]),
        "frameCount": int(smoothed.shape[0]),
        "firstDataColumn": int(first_data_column),
        "displayFrameWindow": PACKETS,
        "displayStftWindow": (PACKETS - WINDOW_LENGTH) // HOP + 1,
        "bandwidth": "20 MHz",
        "phaseAvailable": False,
        "amplitude": np.round(amplitude, 5).tolist(),
        "stftDb": np.round(displayed, 3).tolist(),
        "stftTimes": np.round(display_time_frames / sample_rate, 3).tolist(),
        "stftCenterFrames": frame_centers.astype(int).tolist(),
        "frequencies": np.round(frequencies, 4).tolist(),
        "motionMetric": np.round(motion_metric, 4).tolist(),
        "stft": {
            "smoothing": SMOOTHING,
            "window": WINDOW_LENGTH,
            "hop": HOP,
            "nfft": NFFT,
            "dbMin": DB_MIN,
            "dbMax": DB_MAX,
            "reference": "all bundled Pine examples",
        },
    }


def build_payload(example_paths: dict[str, list[Path]]) -> dict[str, object]:
    reference = shared_test_reference(PINE_ROOT)
    scenes = {}
    for key, spec in EXAMPLE_SPECS.items():
        paths = example_paths[key]
        capture_names = " → ".join(path.stem for path in paths)
        scenes[key] = build_scene(
            paths,
            key=key,
            label=f"{spec['dataset_label']} · {spec['activity']}",
            dataset_key=spec["dataset_key"],
            dataset_label=spec["dataset_label"],
            activity=spec["activity"],
            capture_label=capture_names,
            prediction=spec["prediction"],
            source=(
                f"{spec['dataset_label']} built-in recording · {capture_names}"
            ),
            reference=reference,
        )

    # A hidden-label Fall recording drives the live-capture simulation.  Model
    # predictions were generated by infer_pine_live.py with the copied PyTorch
    # checkpoint and are revealed only when each window becomes available.
    live_path = CSI_DEMO_ROOT / "examples/live/pine_stand_walk_run_fall.txt"
    live_manifest = json.loads(
        (CSI_DEMO_ROOT / "examples/live/pine_stand_walk_run_fall.json").read_text(
            encoding="utf-8"
        )
    )
    live_scene = build_scene(
        [live_path],
        key="live",
        label="Live Simulation · Pine Capture",
        dataset_key="live",
        dataset_label="Live Simulation",
        activity="Pine Capture",
        capture_label="Simulated CFR stream",
        prediction="WAITING",
        source="Pine live stream · STAND → WALK → RUN → FALL",
        reference=reference,
        is_live=True,
        sample_rate=float(live_manifest["sampleRate"]),
        packet_limit=None,
    )
    live_scene["inferenceWindowFrames"] = 500
    live_scene["inferenceWindowSeconds"] = 1.0
    live_scene["inferenceHopFrames"] = 50
    live_scene["inferenceEndpoint"] = "/api/infer"
    live_scene["inferenceCapture"] = "live/pine_stand_walk_run_fall.txt"
    live_scene["goldenSegments"] = live_manifest["segments"]
    live_scene["modelName"] = "SharedChainPineLeNet"
    live_scene["modelCheckpoint"] = "pine_shared_lenet_concat.pt"
    live_scene["modelInputShape"] = [4, 500, 128]
    scenes["live"] = live_scene

    return {
        "title": "WiFi CSI/CFR · LIVE SENSING",
        "defaultScene": "live",
        "threshold": 0.18,
        "toneIndices": list(range(-26, 27)),
        "scenes": scenes,
    }


def generate(
    example_paths: dict[str, list[Path]],
    template_path: Path,
    output_path: Path,
) -> None:
    payload = build_payload(example_paths)
    template = template_path.read_text(encoding="utf-8")
    marker = "__DASHBOARD_DATA__"
    if marker not in template:
        raise ValueError(f"template does not contain {marker}")
    serialized = json.dumps(payload, separators=(",", ":"), ensure_ascii=False)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(template.replace(marker, serialized), encoding="utf-8")
    print(f"generated: {output_path}")
    for key, paths in example_paths.items():
        scene = payload["scenes"][key]
        print(f"{scene['label']}:")
        for path in paths:
            print(f"  {path}")
        print(
            f"  STFT: {len(scene['frequencies'])} bins x "
            f"{len(scene['stftTimes'])} columns"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        "--run-input",
        dest="run_input",
        type=Path,
        nargs="+",
        default=list(EXAMPLE_SPECS["run"]["paths"]),
        help="Run example captures in chronological order",
    )
    for key, spec in list(EXAMPLE_SPECS.items())[1:]:
        parser.add_argument(
            f"--{key}-input",
            type=Path,
            nargs="+",
            default=list(spec["paths"]),
            help=f"{spec['prediction'].title()} example captures in chronological order",
        )
    parser.add_argument("--template", type=Path, default=DEFAULT_TEMPLATE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    example_paths = {
        key: [path.resolve() for path in getattr(args, f"{key}_input")]
        for key in EXAMPLE_SPECS
    }
    generate(example_paths, args.template.resolve(), args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
