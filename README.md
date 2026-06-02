# GPU_flip_fluid

FLIP (Fluid-Implicit-Particle) fluid simulator with two backends:

- **`flip`** — single-thread C++ reference (CPU sim, OpenGL render).
- **`flip_cuda`** — CUDA port (all simulation stages run as GPU kernels).

Both share the same scene setup, controls, and OpenGL/X11 rendering, and both
have matching per-stage instrumentation so CPU vs GPU can be compared fairly.

## Build

Requires a C++17 compiler, OpenGL/GLX/X11 dev headers, and (for the CUDA build)
the CUDA toolkit + an NVIDIA GPU.

```bash
make flip          # CPU build (no GPU needed)
make flip_cuda     # CUDA build (needs nvcc + NVIDIA GPU)
make               # both
make flip_validate # headless CPU-vs-CUDA numerical validation
```

> The CUDA build cannot run on machines without an NVIDIA GPU. Use a CUDA-capable
> PC or Google Colab.

## Run (interactive)

```bash
./flip [--no-vsync]
./flip_cuda [--no-vsync] [--interop]
```

Controls: **LMB drag** = move obstacle, **SPACE/P** = pause, **G** = grid,
**R** = reset, **Q/Esc** = quit. `flip_cuda` also prints timing on **T**.

`--interop` (CUDA only) renders particles straight from a CUDA-mapped OpenGL VBO
(`cudaGraphicsMapResources`), avoiding the per-frame device→host copy.

> Interop needs the **OpenGL context to run on the same NVIDIA GPU** as CUDA.
> On a laptop with switchable graphics (Optimus/PRIME) GLX defaults to the
> iGPU (or llvmpipe), so interop registration fails and it falls back to D2H.
> Force GL onto the NVIDIA GPU:
> ```bash
> prime-run ./flip_cuda --interop
> # or, without the prime-run wrapper:
> __NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia ./flip_cuda --interop
> ```
> Check which GPU drives GL with `glxinfo | grep "OpenGL renderer"`.

## Benchmark mode

Runs the assignment benchmark matrix automatically, then exits:

```bash
./flip      --bench [--warmup 60] [--frames 600] [--csv bench_cpu.csv]
./flip_cuda --bench [--interop]   [--csv bench_cuda.csv]
```

- Sweeps grid resolutions **{50, 100, 150, 200}** (or one via `--only-res N`).
- Discards `--warmup` frames (default 60), averages the next `--frames` (default 600).
- Fixed config: gravity ON, separateParticles ON, compensateDrift ON,
  flipRatio 0.9, obstacle static at (3.0, 2.0), vsync forced off.
- Prints system info + `numPressureIters`/`numSubSteps`, and writes a CSV with
  per-stage timings T1..T10 and T_total (one row per resolution).

## Numerical validation (CPU vs CUDA)

Runs both backends from an identical initial state and reports max/RMS error
per field — proof the port is correct despite reordered ops (Gauss-Seidel →
red-black, sequential scatter → atomic). Headless, no OpenGL.

```bash
make flip_validate
./flip_validate --res 100 --frames 5
```

## Analysis & profiling tools

```bash
# Plots from the benchmark CSVs (needs: pip install matplotlib)
python3 tools/plot_bench.py --cpu bench_cpu.csv --cuda bench_cuda.csv --outdir plots
#   -> plots/total_vs_res.png, speedup.png, stages_cpu.png, stages_cuda.png

# Bonus §4.4 — Nsight timeline + per-kernel metrics (needs nsys/ncu + NVIDIA GPU)
bash tools/profile_nsight.sh          # defaults to res 200
RES=150 bash tools/profile_nsight.sh  # other resolution
```

### Nsight profiling (bonus §4.4) — step by step

1. **Check the tools** are on PATH:
   ```bash
   which nsys ncu
   ```
   If missing, install them (Arch: `sudo pacman -S nsight-systems nsight-compute`)
   or use the copies bundled with the CUDA toolkit (often `/opt/cuda/bin`).

2. **Allow GPU performance counters** (needed by `ncu` only). The NVIDIA driver
   restricts counters to admins by default, so either run `ncu` with `sudo`, or
   open access for the session:
   ```bash
   sudo sh -c 'echo 0 > /proc/sys/kernel/perf_event_paranoid'
   ```
   An `ERR_NVGPUCTRPERM` error means this step is required.

3. **Run** (the app opens a GL window — needs an active display):
   ```bash
   make flip_cuda
   bash tools/profile_nsight.sh            # res 200
   RES=150 bash tools/profile_nsight.sh    # another resolution
   sudo RES=200 bash tools/profile_nsight.sh   # if ncu needs root
   ```
   - `nsys` profiles `--only-res 200 --warmup 10 --frames 30` →
     `nsight/flip_cuda_timeline_res200.nsys-rep`
   - `ncu` profiles `k_jacobiRB` (slow) + `k_p2g` (comparison),
     `--launch-count 4`, tiny frame count (ncu replays each kernel, so it is
     slow) → `nsight/flip_cuda_ncu_res200.ncu-rep`. Metrics captured: compute
     throughput, DRAM throughput, occupancy, branch efficiency, global ld/st
     coalescing.

4. **View results**:
   ```bash
   nsys-ui nsight/flip_cuda_timeline_res200.nsys-rep
   ncu-ui  nsight/flip_cuda_ncu_res200.ncu-rep
   # text summary instead of GUI:
   ncu --import nsight/flip_cuda_ncu_res200.ncu-rep --page details | less
   ```
   Compare `sm__throughput` vs `dram__throughput` per kernel to conclude whether
   it is **compute-bound** or **memory-bound**.

## Timing stages

| Code | Stage |
|------|-------|
| T1_integrate | integrateParticles |
| T2_pushApart | spatial hash (count + parallel scan + scatter) + separation |
| T3_collisions | handleParticleCollisions |
| T4_p2g | transferVelocities(toGrid) = savePrev + classify + p2g + normalize + restoreSolid |
| T5_density | updateParticleDensity (+ rest density) |
| T6_pressure | solveIncompressibility (CPU: Gauss-Seidel; CUDA: red-black GS) |
| T7_g2p | transferVelocities(fromGrid) |
| T8_colors | updateParticleColors + updateCellColors |
| T9_render | drawGrid + drawParticles + drawObstacle + swapBuffers |
| T10_transfer | CUDA only: D2H copy, or interop map/pack/unmap (`--interop`) |
| T_total | wall-clock for the whole frame |
