# Checklist Bukti Laporan — FLIP CPU→CUDA

Daftar perintah yang dijalankan + apa yang di-screenshot untuk tiap bagian
laporan (§4). Jalankan dari root repo di mesin ber-GPU NVIDIA (archlinux).
Centang saat selesai.

> Tip screenshot: rapikan terminal (font cukup besar, sertakan seluruh output
> yang relevan). Simpan semua gambar ke folder `bukti/`.

---

## 0. Persiapan
```bash
make all
nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv   # cek GPU
```
- [ ] Build sukses (tidak ada error)

---

## 1. Spesifikasi hardware  →  Laporan §"Hardware"
Header info hardware tercetak di awal tiap `--bench`. Ambil dari run di bawah.
- [ ] 📸 **SS-01**: blok `=== System Info (CUDA build) ===` (CPU, RAM, GPU, compute
      capability, bus, peak DRAM bandwidth, versi CUDA/driver)

---

## 2. Sanity check (opsional, biar yakin jalan)
```bash
./flip      --bench --warmup 5 --frames 20 --csv test_cpu.csv
./flip_cuda --bench --warmup 5 --frames 20 --csv test_cuda.csv
rm -f test_cpu.csv test_cuda.csv
```
- [ ] Output timing masuk akal (tidak NaN/0 semua)

---

## 3. Benchmark resmi  →  Laporan §"Tabel benchmark" (§4.2)
```bash
./flip      --bench --csv bench_cpu.csv      # ~3-5 menit
./flip_cuda --bench --csv bench_cuda.csv     # lebih cepat
```
- [ ] 📸 **SS-02**: tabel timing console CPU (ke-4 resolusi, terlihat
      `numPressureIters`/`numSubSteps`)
- [ ] 📸 **SS-03**: tabel timing console CUDA (ke-4 resolusi)
- [ ] 📄 simpan `bench_cpu.csv` dan `bench_cuda.csv` (lampiran data mentah)

---

## 4. Grafik analisis  →  Laporan §4.3 (skalabilitas & speedup)
```bash
pip install matplotlib    # kalau belum
python3 tools/plot_bench.py --cpu bench_cpu.csv --cuda bench_cuda.csv --outdir plots
```
- [ ] 📸 **SS-04**: `plots/total_vs_res.png` (T_total vs resolusi, log-scale)
- [ ] 📸 **SS-05**: `plots/speedup.png` (speedup CPU/CUDA per resolusi)
- [ ] 📸 **SS-06**: `plots/stages_cpu.png` (breakdown per-tahap CPU)
- [ ] 📸 **SS-07**: `plots/stages_cuda.png` (breakdown per-tahap CUDA)

---

## 5. Verifikasi visual (port CUDA terlihat benar)  →  Laporan §"Kebenaran"
```bash
./flip          # tekan SPACE, biarkan beberapa detik
./flip_cuda     # tekan SPACE, bandingkan tampilannya
```
- [ ] 📸 **SS-08**: window `flip` (CPU) saat fluida bergerak
- [ ] 📸 **SS-09**: window `flip_cuda` (GPU) — bentuk fluida mirip CPU,
      tidak meledak/NaN

---

## 6. Bonus B1 — CUDA-OpenGL interop  →  Laporan §"Bonus B1"
Bandingkan T10 tanpa vs dengan interop (di resolusi sama, mis. 200):
```bash
./flip_cuda --bench --only-res 200 --csv bench_cuda_d2h.csv
./flip_cuda --bench --only-res 200 --interop --csv bench_cuda_interop.csv
```
- [ ] 📸 **SS-10**: dua output console berdampingan — tunjukkan **T10_transfer
      turun drastis** dengan `--interop` (dan tulisan "interop enabled")

---

## 7. Bonus B2 — Nsight profiling  →  Laporan §4.4
Lihat README bagian "Nsight profiling" untuk izin counter (`perf_event_paranoid`).
```bash
bash tools/profile_nsight.sh        # res 200 (pakai sudo bila ERR_NVGPUCTRPERM)
nsys-ui nsight/flip_cuda_timeline_res200.nsys-rep
ncu-ui  nsight/flip_cuda_ncu_res200.ncu-rep
```
- [ ] 📸 **SS-11**: timeline Nsight Systems 1 frame @res 200 (urutan kernel +
      gap idle GPU)
- [ ] 📸 **SS-12**: Nsight Compute — kernel **lambat** (`k_jacobiRB`): metrik
      `sm__throughput`, `dram__throughput`, occupancy, branch efficiency, coalescing
- [ ] 📸 **SS-13**: Nsight Compute — kernel **pembanding** (`k_p2g`): metrik sama
- [ ] catat kesimpulan tiap kernel: **compute-bound** atau **memory-bound**

---

## 8. (Belum ada tool) Validasi numerik CPU↔CUDA  →  Laporan §"Kebenaran"
Tool dump-state + error L2/max **belum dibuat**. Kalau dibuat nanti, hasilnya:
- [ ] 📸 **SS-14**: tabel error L2/max per field (u, v, posisi) setelah N frame

---

## Ringkasan pemetaan ke laporan
| Bagian laporan | Bukti |
|---|---|
| Hardware | SS-01 |
| Tabel benchmark (§4.2) | SS-02, SS-03, CSV |
| Analisis skalabilitas/speedup (§4.3) | SS-04, SS-05, SS-06, SS-07 |
| Kebenaran numerik | SS-08, SS-09 (visual), SS-14 (kuantitatif, jika ada) |
| Bonus B1 interop | SS-10 |
| Bonus B2 Nsight (§4.4) | SS-11, SS-12, SS-13 |
