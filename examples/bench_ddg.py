#!/usr/bin/env python
"""
Benchmark DDG dual mesh computation across dimensions, methods, and
refinement levels.

Usage:
    python examples/bench_ddg.py              # print table
    python examples/bench_ddg.py --plot       # also show matplotlib plots
    python examples/bench_ddg.py --plot --save bench_ddg.png
"""
from __future__ import annotations

import argparse
import time
from itertools import product as iterproduct

import numpy as np

from hyperct._complex import Complex
from hyperct._backend import get_backend
from hyperct.ddg import compute_vd, d_area, e_star, mean_curvature

# ── helpers ──────────────────────────────────────────────────────────

def _build_complex(dim: int, gens: int):
    """Build a complex with *gens* split-refine cycles.

    Each cycle calls refine_all() then split_generation(), which roughly
    doubles the vertex count per cycle in 1D, quadruples in 2D, etc.

    gens=0 means just triangulate() + refine_all() (the base mesh).
    """
    HC = Complex(dim)
    HC.triangulate()
    HC.refine_all()
    for _ in range(gens):
        HC.split_generation()
        HC.refine_all()
    if dim >= 2:
        dV = HC.boundary()
        for v in dV:
            v.boundary = True
    return HC


def _timeit(fn, repeats: int = 3):
    """Return (best_time_seconds, last_result)."""
    best = float("inf")
    result = None
    for _ in range(repeats):
        t0 = time.perf_counter()
        result = fn()
        elapsed = time.perf_counter() - t0
        best = min(best, elapsed)
    return best, result


def _fmt_time(t):
    """Format seconds as a human-readable string."""
    if t < 1e-3:
        return f"{t * 1e6:7.1f} us"
    if t < 1.0:
        return f"{t * 1e3:7.2f} ms"
    return f"{t:7.2f}  s"


# ── benchmark routines ───────────────────────────────────────────────

def bench_compute_vd(dim, method, gens, repeats):
    """Benchmark compute_vd; returns dict with timings and sizes."""
    HC = _build_complex(dim, gens)
    n_primal = len(HC.V)

    t, _ = _timeit(
        lambda: compute_vd(HC, method=method, global_merge=True), repeats
    )
    n_dual = len(HC.Vd)
    return dict(
        dim=dim, method=method, gens=gens,
        n_primal=n_primal, n_dual=n_dual, time_s=t,
    )


def bench_operators(dim, gens, repeats):
    """Benchmark d_area and e_star on a barycentric dual."""
    HC = _build_complex(dim, gens)
    compute_vd(HC, method="barycentric")
    verts = list(HC.V)

    # d_area
    t_darea, _ = _timeit(
        lambda: sum(d_area(v) for v in verts), repeats
    )

    # e_star (first edge per vertex, limited sample to keep it quick)
    sample = verts[:min(20, len(verts))]

    def _run_estar():
        total = 0.0
        for v1 in sample:
            for v2 in v1.nn:
                r = e_star(v1, v2, HC, dim=dim)
                total += float(np.sum(r)) if isinstance(r, np.ndarray) else float(r)
                break
        return total

    t_estar, _ = _timeit(_run_estar, repeats)

    return dict(
        dim=dim, gens=gens, n_primal=len(verts),
        t_darea=t_darea, t_estar=t_estar,
    )


def bench_curvature(gens, repeats):
    """Benchmark mean_curvature on a 3D complex."""
    HC = _build_complex(3, gens)
    verts = [v for v in HC.V if len(v.nn) >= 3]
    if not verts:
        return None

    def _run():
        for v in verts:
            try:
                mean_curvature(v)
            except Exception:
                pass

    t, _ = _timeit(_run, repeats)
    return dict(dim=3, gens=gens, n_verts=len(verts), time_s=t)


def bench_backend_comparison(dim, method, gens, repeats):
    """Compare sequential (no backend) vs batch (numpy backend) paths."""
    HC_seq = _build_complex(dim, gens)
    n_primal = len(HC_seq.V)

    # Sequential path (default)
    t_seq, _ = _timeit(
        lambda: compute_vd(HC_seq, method=method, global_merge=True), repeats
    )
    n_dual_seq = len(HC_seq.Vd)

    # Batch path (numpy backend)
    HC_batch = _build_complex(dim, gens)
    backend = get_backend("numpy")
    t_batch, _ = _timeit(
        lambda: compute_vd(
            HC_batch, method=method, global_merge=True, backend=backend
        ),
        repeats,
    )
    n_dual_batch = len(HC_batch.Vd)

    speedup = t_seq / t_batch if t_batch > 0 else float("inf")
    return dict(
        dim=dim, method=method, gens=gens, n_primal=n_primal,
        n_dual_seq=n_dual_seq, n_dual_batch=n_dual_batch,
        t_seq=t_seq, t_batch=t_batch, speedup=speedup,
    )


def bench_global_merge(dim, method, gens, repeats):
    """Compare compute_vd with and without global merge."""
    HC_on = _build_complex(dim, gens)
    t_on, _ = _timeit(
        lambda: compute_vd(HC_on, method=method, global_merge=True), repeats
    )
    n_on = len(HC_on.Vd)

    HC_off = _build_complex(dim, gens)
    t_off, _ = _timeit(
        lambda: compute_vd(HC_off, method=method, global_merge=False), repeats
    )
    n_off = len(HC_off.Vd)

    return dict(
        dim=dim, method=method, gens=gens,
        n_primal=len(HC_on.V),
        t_merge_on=t_on, n_merge_on=n_on,
        t_merge_off=t_off, n_merge_off=n_off,
    )


def bench_gpu_vs_cpu(dim, method, gens, repeats):
    """Compare numpy vs torch (CPU) vs torch (GPU) backends."""
    results = {}
    backends_to_try = [
        ("numpy", lambda: get_backend("numpy")),
        ("torch", lambda: get_backend("torch")),
    ]

    n_primal = 0
    for bname, get_be in backends_to_try:
        try:
            be = get_be()
        except (ImportError, Exception):
            continue

        HC = _build_complex(dim, gens)
        n_primal = len(HC.V)
        t, _ = _timeit(
            lambda: compute_vd(
                HC, method=method, global_merge=True, backend=be
            ),
            repeats,
        )
        device = ""
        if bname == "torch":
            device = " (CUDA)" if be.has_cuda else " (CPU)"
        results[bname + device] = dict(time_s=t, n_dual=len(HC.Vd))

    return dict(
        dim=dim, method=method, gens=gens,
        n_primal=n_primal, backends=results,
    )


# ── main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Benchmark DDG dual mesh")
    parser.add_argument("--plot", action="store_true", help="Show matplotlib plots")
    parser.add_argument("--save", type=str, default="", help="Save plot to file")
    parser.add_argument("--repeats", type=int, default=3, help="Timing repeats")
    args = parser.parse_args()

    methods = ["barycentric", "circumcentric"]
    repeats = args.repeats

    # Per-dimension generation limits (3D grows fast → fewer gens)
    dim_gens = {1: [0, 1, 2, 3, 4], 2: [0, 1, 2, 3], 3: [0, 1, 2]}

    # ── 1. compute_vd benchmarks ─────────────────────────────────────
    print("=" * 72)
    print("  compute_vd  benchmarks")
    print("=" * 72)
    hdr = f"{'dim':>3}  {'method':<14} {'gens':>4} {'|V|':>7} {'|Vd|':>7} {'time':>10}"
    print(hdr)
    print("-" * len(hdr))

    vd_results = []
    for dim in [1, 2, 3]:
        for method in methods:
            for gens in dim_gens[dim]:
                row = bench_compute_vd(dim, method, gens, repeats)
                vd_results.append(row)
                print(
                    f"{row['dim']:>3}  {row['method']:<14} {row['gens']:>4} "
                    f"{row['n_primal']:>7} {row['n_dual']:>7} "
                    f"{_fmt_time(row['time_s'])}"
                )
        print()

    # ── 2. operator benchmarks ───────────────────────────────────────
    print("=" * 72)
    print("  operator benchmarks  (d_area, e_star on barycentric dual)")
    print("=" * 72)
    hdr2 = f"{'dim':>3}  {'gens':>4} {'|V|':>7} {'d_area':>10} {'e_star':>10}"
    print(hdr2)
    print("-" * len(hdr2))

    op_results = []
    for dim in [1, 2, 3]:
        for gens in dim_gens[dim]:
            row = bench_operators(dim, gens, repeats)
            op_results.append(row)
            print(
                f"{row['dim']:>3}  {row['gens']:>4} {row['n_primal']:>7} "
                f"{_fmt_time(row['t_darea'])} {_fmt_time(row['t_estar'])}"
            )
        print()

    # ── 3. curvature benchmarks ──────────────────────────────────────
    print("=" * 72)
    print("  mean_curvature benchmarks  (3D)")
    print("=" * 72)
    hdr3 = f"{'gens':>4} {'n_verts':>8} {'time':>10}"
    print(hdr3)
    print("-" * len(hdr3))

    curv_results = []
    for gens in dim_gens[3]:
        row = bench_curvature(gens, repeats)
        if row:
            curv_results.append(row)
            print(f"{row['gens']:>4} {row['n_verts']:>8} {_fmt_time(row['time_s'])}")

    # ── 4. backend comparison (sequential vs batch) ───────────────
    print()
    print("=" * 72)
    print("  sequential vs batch (numpy backend)  comparison")
    print("=" * 72)
    hdr4 = (
        f"{'dim':>3}  {'method':<14} {'gens':>4} {'|V|':>7} "
        f"{'seq':>10} {'batch':>10} {'speedup':>8} {'|Vd| ok':>7}"
    )
    print(hdr4)
    print("-" * len(hdr4))

    be_results = []
    for dim in [2, 3]:
        for method in methods:
            for gens in dim_gens[dim]:
                row = bench_backend_comparison(dim, method, gens, repeats)
                be_results.append(row)
                ok = "Y" if row["n_dual_seq"] == row["n_dual_batch"] else "N"
                print(
                    f"{row['dim']:>3}  {row['method']:<14} {row['gens']:>4} "
                    f"{row['n_primal']:>7} "
                    f"{_fmt_time(row['t_seq'])} {_fmt_time(row['t_batch'])} "
                    f"{row['speedup']:>7.2f}x {ok:>7}"
                )
        print()

    # ── 5. global merge impact ────────────────────────────────────
    print("=" * 72)
    print("  global merge impact  (merge_all on vs off)")
    print("=" * 72)
    hdr5 = (
        f"{'dim':>3}  {'method':<14} {'gens':>4} {'|V|':>7} "
        f"{'merge on':>10} {'|Vd|':>6} {'merge off':>10} {'|Vd|':>6}"
    )
    print(hdr5)
    print("-" * len(hdr5))

    gm_results = []
    for dim in [2, 3]:
        for method in methods:
            for gens in dim_gens[dim][:3]:  # limit to keep it quick
                row = bench_global_merge(dim, method, gens, repeats)
                gm_results.append(row)
                print(
                    f"{row['dim']:>3}  {row['method']:<14} {row['gens']:>4} "
                    f"{row['n_primal']:>7} "
                    f"{_fmt_time(row['t_merge_on'])} {row['n_merge_on']:>6} "
                    f"{_fmt_time(row['t_merge_off'])} {row['n_merge_off']:>6}"
                )
        print()

    # ── 6. GPU vs CPU backends ────────────────────────────────────
    print("=" * 72)
    print("  GPU vs CPU backend comparison")
    print("=" * 72)

    gpu_results = []
    for dim in [2, 3]:
        for method in methods:
            for gens in dim_gens[dim][:3]:
                row = bench_gpu_vs_cpu(dim, method, gens, repeats)
                gpu_results.append(row)
                parts = [
                    f"{dim}D {method:<14} gen={gens} |V|={row['n_primal']:>5}"
                ]
                for bname, bdata in row["backends"].items():
                    parts.append(
                        f"  {bname}: {_fmt_time(bdata['time_s'])} "
                        f"|Vd|={bdata['n_dual']}"
                    )
                print("  ".join(parts[:1]))
                for p in parts[1:]:
                    print(f"    {p}")
        print()

    # ── 7. plots ──────────────────────────────────────────────────
    if args.plot or args.save:
        _plot(vd_results, op_results, curv_results, be_results, args.save)


# ── plotting ─────────────────────────────────────────────────────────

def _plot(vd_results, op_results, curv_results, be_results, save_path):
    import matplotlib
    if save_path:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(17, 9))
    fig.suptitle("DDG Dual Mesh Benchmarks", fontsize=14, fontweight="bold")

    colours = {
        (1, "barycentric"):   "tab:blue",
        (1, "circumcentric"): "tab:cyan",
        (2, "barycentric"):   "tab:orange",
        (2, "circumcentric"): "tab:red",
        (3, "barycentric"):   "tab:green",
        (3, "circumcentric"): "tab:olive",
    }

    # ── panel 0: compute_vd time vs |V| ──────────────────────────────
    ax = axes[0, 0]
    groups = {}
    for r in vd_results:
        key = (r["dim"], r["method"])
        groups.setdefault(key, []).append(r)
    for key, rows in sorted(groups.items()):
        rows.sort(key=lambda r: r["n_primal"])
        xs = [r["n_primal"] for r in rows]
        ys = [r["time_s"] * 1e3 for r in rows]
        c = colours.get(key, "grey")
        ax.plot(xs, ys, "o-", color=c, markersize=5,
                label=f"{key[0]}D {key[1][:5]}")
    ax.set_xlabel("|V| (primal vertices)")
    ax.set_ylabel("time (ms)")
    ax.set_title("compute_vd  time vs mesh size")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)

    # ── panel 1: |Vd| vs |V| ────────────────────────────────────────
    ax = axes[0, 1]
    for key, rows in sorted(groups.items()):
        rows.sort(key=lambda r: r["n_primal"])
        xs = [r["n_primal"] for r in rows]
        ys = [r["n_dual"] for r in rows]
        c = colours.get(key, "grey")
        ax.plot(xs, ys, "o-", color=c, markersize=5,
                label=f"{key[0]}D {key[1][:5]}")
    ax.set_xlabel("|V| (primal vertices)")
    ax.set_ylabel("|Vd| (dual vertices)")
    ax.set_title("dual mesh size vs primal mesh size")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)

    # ── panel 2: operator times ──────────────────────────────────────
    ax = axes[1, 0]
    dim_colours = {1: "tab:blue", 2: "tab:orange", 3: "tab:green"}
    op_groups = {}
    for r in op_results:
        op_groups.setdefault(r["dim"], []).append(r)
    for d in sorted(op_groups):
        rows = sorted(op_groups[d], key=lambda r: r["n_primal"])
        xs = [r["n_primal"] for r in rows]
        c = dim_colours[d]
        ax.plot(xs, [r["t_darea"] * 1e3 for r in rows], "o-", color=c,
                markersize=5, label=f"{d}D d_area")
        ax.plot(xs, [r["t_estar"] * 1e3 for r in rows], "^--", color=c,
                markersize=5, label=f"{d}D e_star")
    ax.set_xlabel("|V| (primal vertices)")
    ax.set_ylabel("time (ms)")
    ax.set_title("operator timings")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)

    # ── panel 3: curvature ───────────────────────────────────────────
    ax = axes[1, 1]
    if curv_results:
        nverts = [r["n_verts"] for r in curv_results]
        times = [r["time_s"] * 1e3 for r in curv_results]
        ax.plot(nverts, times, "o-", color="tab:purple", markersize=6)
    ax.set_xlabel("n_verts (3D)")
    ax.set_ylabel("time (ms)")
    ax.set_title("mean_curvature  (3D)")
    ax.grid(True, alpha=0.3)

    # ── panel 4: backend comparison (sequential vs batch speedup) ────
    ax = axes[0, 2]
    if be_results:
        be_groups = {}
        for r in be_results:
            key = (r["dim"], r["method"])
            be_groups.setdefault(key, []).append(r)
        for key, rows in sorted(be_groups.items()):
            rows.sort(key=lambda r: r["n_primal"])
            xs = [r["n_primal"] for r in rows]
            ys = [r["speedup"] for r in rows]
            c = colours.get(key, "grey")
            ax.plot(xs, ys, "o-", color=c, markersize=5,
                    label=f"{key[0]}D {key[1][:5]}")
        ax.axhline(1.0, color="black", linestyle="--", alpha=0.5, linewidth=0.8)
    ax.set_xlabel("|V| (primal vertices)")
    ax.set_ylabel("speedup (seq / batch)")
    ax.set_title("batch vs sequential speedup")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)

    # ── panel 5: backend absolute times ──────────────────────────────
    ax = axes[1, 2]
    if be_results:
        for key, rows in sorted(be_groups.items()):
            rows.sort(key=lambda r: r["n_primal"])
            xs = [r["n_primal"] for r in rows]
            c = colours.get(key, "grey")
            ax.plot(xs, [r["t_seq"] * 1e3 for r in rows], "o-", color=c,
                    markersize=5, label=f"{key[0]}D {key[1][:5]} seq")
            ax.plot(xs, [r["t_batch"] * 1e3 for r in rows], "^--", color=c,
                    markersize=5, alpha=0.6,
                    label=f"{key[0]}D {key[1][:5]} batch")
    ax.set_xlabel("|V| (primal vertices)")
    ax.set_ylabel("time (ms)")
    ax.set_title("sequential vs batch absolute times")
    ax.legend(fontsize=6, ncol=2)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"\nPlot saved to {save_path}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
