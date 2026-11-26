#!/usr/bin/env bash
set -euo pipefail

REMOTE_ROOT="/local/mnt/workspace/penzhan/projects/WiFi-CSI-Sensing-Benchmark"
PORT="${1:-8765}"

if python3 -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:${PORT}/health', timeout=2)" >/dev/null 2>&1; then
  exit 0
fi

cd "$REMOTE_ROOT"
nohup cfr/bin/python /tmp/serve_pine_inference.py \
  --checkpoint checkpoints/pine_shared_lenet_concat.pt \
  --data-root Data/Pine/60 \
  --host 127.0.0.1 \
  --port "$PORT" \
  >/tmp/pine_inference_server.log 2>&1 </dev/null &

for _ in $(seq 1 30); do
  if python3 -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:${PORT}/health', timeout=2)" >/dev/null 2>&1; then
    exit 0
  fi
  sleep 1
done

echo "Pine inference service did not become ready" >&2
tail -20 /tmp/pine_inference_server.log >&2 || true
exit 1
