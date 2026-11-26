#!/usr/bin/env bash
set -euo pipefail

CSI_DEMO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
TOOLS_DIR="$CSI_DEMO_ROOT/tools"
if [[ -x "$CSI_DEMO_ROOT/.venv/bin/python" ]]; then
  PYTHON="$CSI_DEMO_ROOT/.venv/bin/python"
else
  PYTHON="$CSI_DEMO_ROOT/../.venv/bin/python"
fi
INFERENCE_PORT=8765
DASHBOARD_PORT=8000

"$PYTHON" "$TOOLS_DIR/serve_pine_inference.py" \
  --checkpoint "$CSI_DEMO_ROOT/models/pine_shared_lenet_concat.pt" \
  --data-root "$CSI_DEMO_ROOT/examples" \
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
  --root "$CSI_DEMO_ROOT/out" \
  --host 0.0.0.0 \
  --port "$DASHBOARD_PORT"
