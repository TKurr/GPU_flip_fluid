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
make flip        # CPU build (no GPU needed)
make flip_cuda   # CUDA build (needs nvcc + NVIDIA GPU)
make             # both
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

## Analysis & profiling tools

```bash
# Plots from the benchmark CSVs (needs: pip install matplotlib)
python3 tools/plot_bench.py --cpu bench_cpu.csv --cuda bench_cuda.csv --outdir plots
#   -> plots/total_vs_res.png, speedup.png, stages_cpu.png, stages_cuda.png

# Bonus §4.4 — Nsight timeline + per-kernel metrics (needs nsys/ncu + NVIDIA GPU)
bash tools/profile_nsight.sh          # defaults to res 200
RES=150 bash tools/profile_nsight.sh  # other resolution
```

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
