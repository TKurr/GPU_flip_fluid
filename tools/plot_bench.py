#!/usr/bin/env python3
"""Plot FLIP fluid CPU-vs-CUDA benchmark results (§4.3).

Reads the CSVs produced by `./flip --bench` and `./flip_cuda --bench` and emits:
  1. T_total vs resolution (log-scale Y) for CPU and CUDA  -> total_vs_res.png
  2. Speedup (CPU T_total / CUDA T_total) vs resolution    -> speedup.png
  3. Per-stage breakdown (stacked bars) for each build      -> stages_<ver>.png

Usage:
  python3 tools/plot_bench.py [--cpu bench_cpu.csv] [--cuda bench_cuda.csv]
                              [--outdir plots]

Only matplotlib + the stdlib are required:  pip install matplotlib
"""
import argparse
import csv
import os
import sys

STAGES = ["T1_integrate", "T2_pushApart", "T3_collisions", "T4_p2g",
          "T5_density", "T6_pressure", "T7_g2p", "T8_colors",
          "T9_render", "T10_transfer"]


def load(path):
    """Return list of row dicts (floats where possible), or None if missing."""
    if not path or not os.path.exists(path):
        return None
    rows = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            conv = {}
            for k, v in row.items():
                try:
                    conv[k] = float(v)
                except (ValueError, TypeError):
                    conv[k] = v
            rows.append(conv)
    rows.sort(key=lambda r: r["res"])
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cpu", default="bench_cpu.csv")
    ap.add_argument("--cuda", default="bench_cuda.csv")
    ap.add_argument("--outdir", default="plots")
    args = ap.parse_args()

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        sys.exit("matplotlib not installed:  pip install matplotlib")

    cpu = load(args.cpu)
    cuda = load(args.cuda)
    if not cpu and not cuda:
        sys.exit(f"no data found ({args.cpu}, {args.cuda}) — run --bench first")

    os.makedirs(args.outdir, exist_ok=True)

    # ── Plot 1: T_total vs resolution, log-scale Y ──
    plt.figure(figsize=(7, 5))
    if cpu:
        plt.plot([r["res"] for r in cpu], [r["T_total"] for r in cpu],
                 "o-", label="CPU")
    if cuda:
        plt.plot([r["res"] for r in cuda], [r["T_total"] for r in cuda],
                 "s-", label="CUDA")
    plt.yscale("log")
    plt.xlabel("grid resolution")
    plt.ylabel("T_total per frame (ms, log scale)")
    plt.title("Frame time vs resolution")
    plt.grid(True, which="both", ls=":", alpha=0.5)
    plt.legend()
    out = os.path.join(args.outdir, "total_vs_res.png")
    plt.tight_layout(); plt.savefig(out, dpi=130); plt.close()
    print("wrote", out)

    # ── Plot 2: speedup vs resolution ──
    if cpu and cuda:
        cmap = {r["res"]: r["T_total"] for r in cuda}
        res = [r["res"] for r in cpu if r["res"] in cmap]
        spd = [r["T_total"] / cmap[r["res"]] for r in cpu if r["res"] in cmap]
        if res:
            plt.figure(figsize=(7, 5))
            plt.plot(res, spd, "d-", color="tab:green")
            for x, y in zip(res, spd):
                plt.annotate(f"{y:.1f}x", (x, y),
                             textcoords="offset points", xytext=(0, 8),
                             ha="center")
            plt.xlabel("grid resolution")
            plt.ylabel("speedup (CPU T_total / CUDA T_total)")
            plt.title("CUDA speedup over CPU")
            plt.grid(True, ls=":", alpha=0.5)
            out = os.path.join(args.outdir, "speedup.png")
            plt.tight_layout(); plt.savefig(out, dpi=130); plt.close()
            print("wrote", out)

    # ── Plot 3: per-stage stacked breakdown ──
    for ver, rows in (("cpu", cpu), ("cuda", cuda)):
        if not rows:
            continue
        res = [r["res"] for r in rows]
        present = [s for s in STAGES if any(s in r for r in rows)]
        bottoms = [0.0] * len(rows)
        plt.figure(figsize=(8, 5))
        x = range(len(res))
        for s in present:
            vals = [r.get(s, 0.0) for r in rows]
            plt.bar(x, vals, bottom=bottoms, label=s)
            bottoms = [b + v for b, v in zip(bottoms, vals)]
        plt.xticks(list(x), [str(int(r)) for r in res])
        plt.xlabel("grid resolution")
        plt.ylabel("per-frame time (ms)")
        plt.title(f"Per-stage breakdown ({ver.upper()})")
        plt.legend(fontsize=7, ncol=2)
        out = os.path.join(args.outdir, f"stages_{ver}.png")
        plt.tight_layout(); plt.savefig(out, dpi=130); plt.close()
        print("wrote", out)


if __name__ == "__main__":
    main()
