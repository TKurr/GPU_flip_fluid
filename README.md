# GPU_flip_fluid

Simulator fluida FLIP dengan dua backend:

- **`flip`** — referensi CPU single-thread.
- **`flip_cuda`** — port CUDA (semua tahap simulasi jalan sebagai kernel GPU).

Keduanya pakai scene, kontrol, dan render OpenGL yang sama, plus instrumentasi
timing per-tahap yang identik supaya CPU vs GPU bisa dibandingkan adil.

## Build

Butuh compiler C++17, header OpenGL/GLX/X11, dan (untuk CUDA) CUDA toolkit + GPU
NVIDIA.

```bash
make flip          # CPU
make flip_cuda     # CUDA (butuh nvcc + GPU NVIDIA)
make flip_validate # validasi numerik CPU vs CUDA (headless)
make               # flip + flip_cuda
```

## Run

```bash
./flip
./flip_cuda                 # tambah --interop untuk render via VBO tanpa copy D2H
```

Kontrol: **LMB drag** = geser obstacle · **SPACE/P** = pause · **G** = grid ·
**R** = reset · **Q/Esc** = keluar · **T** = print timing (CUDA).
Flag: `--autostart` (langsung jalan), `--res N` (set resolusi awal).

> **Interop (`--interop`)** butuh konteks OpenGL di GPU NVIDIA yang sama. Di
> laptop Optimus jalankan dengan `prime-run ./flip_cuda --interop`, kalau tidak
> ia otomatis fallback ke copy D2H.

## Demo (urutan presentasi)

1. **Sim jalan** — `./flip_cuda`, tekan **SPACE**. Tunjukkan fluida + FPS naik.
2. **CPU vs GPU berdampingan** — `bash tools/compare_cpu_gpu.sh 100`. Bentuk
   fluida sama; window CPU jauh lebih berat saat resolusi dinaikkan.
3. **Benchmark** — `./flip_cuda --bench` (lalu `./flip --bench`). Tunjukkan tabel
   timing per tahap + `numPressureIters`. Buka plot speedup (lihat di bawah).
4. **Validasi numerik** — `./flip_validate --res 100 --frames 5`. Error RMS kecil
   → hasil GPU setara CPU.
5. **(Bonus) Interop** — `prime-run ./flip_cuda --interop`, tunjukkan T10 turun.
6. **(Bonus) Nsight** — buka report di `docs/` (timeline + analisa kernel).

## Benchmark

```bash
./flip      --bench --csv bench_cpu.csv
./flip_cuda --bench --csv bench_cuda.csv     # tambah --interop bila perlu
```

Sweep resolusi {50,100,150,200} (atau satu via `--only-res N`), buang 60 frame
warmup, rata-rata 600 frame, vsync off, obstacle statis (3,2). Output: info
hardware + tabel timing T1..T10/T_total + CSV.

Plot dari CSV (butuh `pip install matplotlib`):

```bash
python3 tools/plot_bench.py --cpu bench_cpu.csv --cuda bench_cuda.csv --outdir plots
# -> total_vs_res.png, speedup.png, stages_cpu.png, stages_cuda.png
```

## Validasi numerik

```bash
./flip_validate --res 100 --frames 5
```

Jalankan CPU & CUDA dari kondisi awal identik, laporkan error max/RMS per field.
Error kecil = port benar meski ordering operasi berubah (Gauss-Seidel → red-black,
scatter → atomic).

## Nsight (bonus profiling)

```bash
# izin counter bila kena ERR_NVGPUCTRPERM:
sudo sh -c 'echo 0 > /proc/sys/kernel/perf_event_paranoid'

bash tools/profile_nsight.sh           # res 200 (RES=150 untuk resolusi lain)
nsys-ui docs/*.nsys-rep                 # timeline
ncu-ui  docs/*.ncu-rep                  # metrik kernel
```

Bandingkan **Compute Throughput** vs **Memory Throughput** per kernel untuk
menyimpulkan compute-bound / memory-bound.

## Tahap timing

| Kode | Tahap |
|------|-------|
| T1_integrate | integrateParticles |
| T2_pushApart | spatial hash (count + scan + scatter) + separasi |
| T3_collisions | handleParticleCollisions |
| T4_p2g | transferVelocities→grid (savePrev + classify + p2g + normalize + restoreSolid) |
| T5_density | updateParticleDensity |
| T6_pressure | solveIncompressibility (CPU: Gauss-Seidel · CUDA: red-black) |
| T7_g2p | transferVelocities→partikel |
| T8_colors | updateParticleColors + updateCellColors |
| T9_render | drawGrid + drawParticles + drawObstacle + swap |
| T10_transfer | CUDA: copy D2H, atau map/pack/unmap (`--interop`) |
| T_total | wall-clock satu frame |
