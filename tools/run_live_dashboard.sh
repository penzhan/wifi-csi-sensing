#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
TOOLS_DIR="$ROOT_DIR/csi_demo/tools"
REMOTE_HOST="penzhan@hu-penzhan-lv"
INFERENCE_PORT=8765
DASHBOARD_PORT=8000

scp -F /dev/null \
  "$TOOLS_DIR/pine_data.py" \
  "$TOOLS_DIR/infer_pine_live.py" \
  "$TOOLS_DIR/serve_pine_inference.py" \
  "$TOOLS_DIR/start_remote_pine_inference.sh" \
  "$REMOTE_HOST:/tmp/"

ssh -F /dev/null "$REMOTE_HOST" \
  "mkdir -p /local/mnt/workspace/penzhan/projects/WiFi-CSI-Sensing-Benchmark/Data/Pine/60/live"
scp -F /dev/null \
  "$ROOT_DIR/csi_demo/examples/live/pine_stand_walk_run_fall.txt" \
  "$REMOTE_HOST:/local/mnt/workspace/penzhan/projects/WiFi-CSI-Sensing-Benchmark/Data/Pine/60/live/pine_stand_walk_run_fall.txt"

ssh -F /dev/null "$REMOTE_HOST" \
  "bash /tmp/start_remote_pine_inference.sh $INFERENCE_PORT"

ssh -F /dev/null -N \
  -L "$INFERENCE_PORT:127.0.0.1:$INFERENCE_PORT" \
  "$REMOTE_HOST" &
TUNNEL_PID=$!
trap 'kill "$TUNNEL_PID" 2>/dev/null || true' EXIT INT TERM

echo "Dashboard: http://127.0.0.1:$DASHBOARD_PORT/csi_demo/out/dashboard.html"
echo "Inference: http://127.0.0.1:$INFERENCE_PORT/health"
python3 "$TOOLS_DIR/serve_dashboard.py" \
  --root "$ROOT_DIR" \
  --host 127.0.0.1 \
  --port "$DASHBOARD_PORT"
