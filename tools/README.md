# CSI Demo tools

The scripts in this directory use one shared processing pipeline:

1. Select Pine's 53 active tones.
2. Apply 5-packet temporal smoothing.
3. Normalize once per chain over time and tones.
4. Apply a 128-packet Hann STFT with an 8-packet hop.
5. Average linear power across tones and chains.
6. Convert to dB using the reference shared by the bundled Pine examples.

## Generate the dashboard

```bash
MPLCONFIGDIR=/tmp/cfr_matplotlib .venv/bin/python csi_demo/tools/dashboard.py
```

## Generate a static STFT image

```bash
MPLCONFIGDIR=/tmp/cfr_matplotlib .venv/bin/python csi_demo/tools/plot_pine_stft.py --scene live
MPLCONFIGDIR=/tmp/cfr_matplotlib .venv/bin/python csi_demo/tools/plot_pine_stft.py --scene all
```

Images are written to `csi_demo/out/plots/` by default.

## Generate a progressive STFT GIF

```bash
MPLCONFIGDIR=/tmp/cfr_matplotlib .venv/bin/python csi_demo/tools/render_pine_stft_replay.py --scene live
```

Use `--scene run`, `walk`, `pick`, `lie`, `fall`, `stand`, or `sit` for a
built-in recorded example.

## Run offline Pine neural-network inference

`infer_pine_live.py` requires PyTorch. It applies the training preprocessing
and can export a prediction timeline for model validation:

```bash
python csi_demo/tools/infer_pine_live.py \
  --checkpoint csi_demo/models/pine_shared_lenet_concat.pt \
  --input csi_demo/examples/pine/fall/cfr_41.txt \
  --output csi_demo/models/pine_fall_cfr41_predictions.json \
  --period-ms 2 \
  --hop 8
```

The checkpoint input is `4 chains × 500 frames × 128 tones`. With the current
2 ms simulation period, inference starts after a 1.0-second capture window.

## Build a labelled live CSI stream

```bash
.venv/bin/python csi_demo/tools/build_pine_live_stream.py \
  --segment STAND=csi_demo/examples/pine/stand/cfr_40.txt \
  --segment WALK=csi_demo/examples/pine/walk/cfr_20.txt \
  --segment WALK=csi_demo/examples/pine/walk/cfr_34.txt \
  --segment RUN=csi_demo/examples/pine/run/cfr_29.txt \
  --segment RUN=csi_demo/examples/pine/run/cfr_37.txt \
  --segment FALL=csi_demo/examples/pine/fall/cfr_41.txt \
  --output csi_demo/examples/live/pine_stand_walk_run_fall.txt \
  --manifest csi_demo/examples/live/pine_stand_walk_run_fall.json \
  --period-ms 2
```

The manifest stores the exact start/end frame and golden label for every
source segment. WALK and RUN use two consecutive captures because they are
periodic activities; STAND and FALL remain single captures. The current stream
contains 3,987 frames and lasts 7.974 s.

## Run the live Dashboard with server inference

For low-latency local PyTorch inference:

```bash
bash csi_demo/tools/run_local_dashboard.sh
```

The plotting and inference tools share the repository `.venv`.

For inference on `hu-penzhan-lv` through an SSH tunnel:

```bash
bash csi_demo/tools/run_live_dashboard.sh
```

Then open:

```text
http://127.0.0.1:8000/csi_demo/out/dashboard.html
```

The launcher starts an SSH tunnel to `hu-penzhan-lv`. The Dashboard does not
contain precomputed labels or confidence values: after 500 simulated frames it
shows `PREDICTING`, calls the server, waits for PyTorch, and then displays the
returned result. Prediction requests use a 500-frame window and a 50-frame
hop, corresponding to a 100 ms update interval at 500 FPS.
