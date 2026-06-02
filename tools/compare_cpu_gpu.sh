#!/usr/bin/env bash
# Side-by-side qualitative validation: launch the CPU and CUDA builds at the
# same resolution, both unpaused, so the fluid can be compared visually.
#
# Usage:
#   bash tools/compare_cpu_gpu.sh [RES]      # default RES=100
#   RES=200 bash tools/compare_cpu_gpu.sh
#
# Both windows are 900x700. If `wmctrl` is installed they are auto-placed left
# (CPU) and right (CUDA); otherwise drag them apart manually.
set -e
cd "$(dirname "$0")/.."

RES=${RES:-${1:-100}}

# Build if needed.
make flip flip_cuda

# On Optimus/PRIME laptops force the CUDA build's GL onto the NVIDIA GPU.
PRIME=""
if command -v prime-run >/dev/null 2>&1; then PRIME="prime-run"; fi

echo "[compare] res=$RES  prime=${PRIME:-none}"
echo "[compare] launching CPU (left) and CUDA (right) — both autostart, no-vsync"

./flip --no-vsync --autostart --res "$RES" &
CPU_PID=$!

$PRIME ./flip_cuda --no-vsync --autostart --res "$RES" &
GPU_PID=$!

# Place windows side by side once they exist.
if command -v wmctrl >/dev/null 2>&1; then
    sleep 2
    wmctrl -r "C++ CPU sim"  -e 0,40,80,900,700  2>/dev/null || true
    wmctrl -r "CUDA GPU sim" -e 0,980,80,900,700 2>/dev/null || true
else
    echo "[compare] (install 'wmctrl' to auto-arrange windows; drag them apart for now)"
fi

echo "[compare] both running. Close a window or Ctrl-C to stop."
wait "$CPU_PID" "$GPU_PID"
