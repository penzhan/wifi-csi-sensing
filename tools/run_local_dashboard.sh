#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
TOOLS_DIR="$ROOT_DIR/csi_demo/tools"
PYTHON="$ROOT_DIR/.venv/bin/python"
INFERENCE_PORT=8765
DASHBOARD_PORT=8000

"$PYTHON" "$TOOLS_DIR/serve_pine_inference.py" \
  --checkpoint "$ROOT_DIR/csi_demo/models/pine_shared_lenet_concat.pt" \
  --data-root "$ROOT_DIR/csi_demo/examples" \
  --host 127.0.0.1 \
  --port "$INFERENCE_PORT" &
INFERENCE_PID=$!

cleanup() {
  kill "$INFERENCE_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

for _ in $(seq 1 30); do
  if "$PYTHON" -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:${INFERENCE_PORT}/health', timeout=2)" >/dev/null 2>&1; then
    break
  fi
  if ! kill -0 "$INFERENCE_PID" 2>/dev/null; then
    echo "Local inference service exited before becoming ready" >&2
    exit 1
  fi
  sleep 1
done

echo "Dashboard (local): http://127.0.0.1:$DASHBOARD_PORT/dashboard.html"
echo "Dashboard (LAN):   http://<host-ip>:$DASHBOARD_PORT/dashboard.html"
echo "Inference: local PyTorch CPU · http://127.0.0.1:$INFERENCE_PORT/health"
"$PYTHON" "$TOOLS_DIR/serve_dashboard.py" \
  --root "$ROOT_DIR/csi_demo/out" \
  --host 0.0.0.0 \
  --port "$DASHBOARD_PORT"
