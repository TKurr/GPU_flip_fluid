#!/usr/bin/env bash
# Bonus §4.4 — Nsight profiling helper for flip_cuda.
#
# Produces:
#   1. Nsight Systems timeline of one resolution (kernel order + GPU gaps).
#   2. Nsight Compute report for the slow kernel (k_jacobiRB) and a comparison
#      kernel (k_p2g), with the metrics the assignment asks for.
#
# Requires an NVIDIA GPU + Nsight tools (nsys, ncu) on PATH and an X display
# (the app opens a GL window). Run from the repo root:  bash tools/profile_nsight.sh
#
# Notes:
#   * ncu REPLAYS each kernel many times — keep --frames tiny (it's slow).
#   * ncu usually needs GPU perf-counter access: run as root or set
#     /proc/sys/kernel/perf_event_paranoid, or pass --target-processes.
set -e
cd "$(dirname "$0")/.."

RES=${RES:-200}
APP=./flip_cuda
OUTDIR=nsight
mkdir -p "$OUTDIR"

if [ ! -x "$APP" ]; then
    echo "building $APP ..."; make flip_cuda
fi

# ── 1. Nsight Systems timeline (a few steady-state frames) ──
if command -v nsys >/dev/null 2>&1; then
    echo "=== nsys: timeline @ res $RES ==="
    nsys profile --force-overwrite true \
        -o "$OUTDIR/flip_cuda_timeline_res${RES}" \
        --stats=true \
        "$APP" --bench --only-res "$RES" --warmup 10 --frames 30
else
    echo "nsys not found — skipping timeline"
fi

# ── 2. Nsight Compute per-kernel metrics ──
# Metric set covers: compute throughput, DRAM throughput, achieved occupancy,
# branch efficiency, and global load/store coalescing (sectors per request).
METRICS="sm__throughput.avg.pct_of_peak_sustained_elapsed,\
dram__throughput.avg.pct_of_peak_sustained_elapsed,\
smsp__warps_active.avg.pct_of_peak_sustained_active,\
smsp__sass_average_branch_targets_threads_uniform.pct,\
l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.ratio,\
l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_st.ratio"

if command -v ncu >/dev/null 2>&1; then
    echo "=== ncu: k_jacobiRB (slow) + k_p2g (comparison) @ res $RES ==="
    ncu --force-overwrite \
        -o "$OUTDIR/flip_cuda_ncu_res${RES}" \
        --target-processes all \
        --kernel-name "regex:k_jacobiRB|k_p2g" \
        --launch-count 4 \
        --metrics "$METRICS" \
        "$APP" --bench --only-res "$RES" --warmup 3 --frames 2
    echo "open the report with:  ncu-ui $OUTDIR/flip_cuda_ncu_res${RES}.ncu-rep"
else
    echo "ncu not found — skipping kernel metrics"
fi

echo "done. reports in $OUTDIR/"
