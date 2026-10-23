#!/usr/bin/env python3
"""
Transport report: how much of PriX's speedup over CrypTen depends on the transport?

Input: one benchmark JSON with four runs made with --comm-stats (ideally in one session):
    crypten    CrypTen (PyTorch), torch.distributed/gloo over loopback TCP
    prix       PriX with its default peer-to-peer TCP transport (--prix-transport tcp)
    prix_gloo  PriX with its 2-party all_reduce routed over gloo, like CrypTen (--prix-transport gloo)
    prix_shm   PriX over shared memory, a single-host shortcut (--prix-transport shm)

    python benchmarks/transport_ablation.py results.json [--fig docs/figures/transport_ablation]
"""

import argparse
import json
import math
import os

import numpy as np

MODES = ["crypten", "poly", "shaft", "curl", "fission"]
MODE_LABELS = {"crypten": "CrypTen approx.", "poly": "Piecewise poly", "shaft": "SHAFT",
               "curl": "Curl LUTs", "fission": "Fission"}
RUNS = ("crypten", "prix", "prix_gloo", "prix_shm")


def geomean(xs):
    xs = [x for x in xs if x > 0]
    return math.exp(sum(math.log(x) for x in xs) / len(xs))


def rows(data, section):
    out = []
    for mode in MODES:
        if not all(mode in data[fw] for fw in RUNS):
            continue
        for key in data["prix"][mode][section]:
            r = dict(mode=mode, key=key)
            for fw in RUNS:
                r[fw] = data[fw][mode][section][key]
                comm = data[fw][mode].get("comm", {}).get(section, {}).get(key)
                if comm:
                    r[fw + "_rounds"], r[fw + "_bytes"] = comm["rounds"], comm["bytes"]
            out.append(r)
    return out


def summarize(rs):
    """Geomean speedups over CrypTen for each PriX transport, plus TCP's ratios to the others."""
    return dict(
        tcp=geomean([r["crypten"] / r["prix"] for r in rs]),
        gloo=geomean([r["crypten"] / r["prix_gloo"] for r in rs]),
        shm=geomean([r["crypten"] / r["prix_shm"] for r in rs]),
        tcp_vs_gloo=geomean([r["prix_gloo"] / r["prix"] for r in rs]),
        shm_vs_tcp=geomean([r["prix"] / r["prix_shm"] for r in rs]),
    )


def print_report(data):
    for section, title in (("llm", "Transformer blocks"), ("micro", "Primitives")):
        rs = rows(data, section)
        print(f"\n### {title}: PriX speedup over CrypTen by PriX transport (geometric means)")
        print("| Mode | PriX over TCP (default) | PriX over gloo | PriX over shared memory | "
              "TCP vs. gloo (PriX) | Shared memory vs. TCP (PriX) |")
        print("|---|---:|---:|---:|---:|---:|")
        for mode in MODES + ["all"]:
            sel = rs if mode == "all" else [r for r in rs if r["mode"] == mode]
            if not sel:
                continue
            s = summarize(sel)
            label = "**All modes**" if mode == "all" else MODE_LABELS[mode]
            print(f"| {label} | {s['tcp']:.2f}x | {s['gloo']:.2f}x | {s['shm']:.2f}x | "
                  f"{s['tcp_vs_gloo']:.2f}x | {s['shm_vs_tcp']:.2f}x |")

        print(f"\n### {title}: rounds and MB sent (CrypTen / PriX)")
        print("| Mode | Workload | Rounds | MB sent |")
        print("|---|---|---:|---:|")
        for r in rs:
            if "crypten_rounds" in r and "prix_rounds" in r:
                print(f"| {MODE_LABELS[r['mode']]} | {r['key']} | {r['crypten_rounds']:.0f} / "
                      f"{r['prix_rounds']:.0f} | {r['crypten_bytes'] / 1e6:.2f} / {r['prix_bytes'] / 1e6:.2f} |")


def plot(data, out_base, model="GPT2 (1-Block, seq=16)"):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.titleweight": "bold"})
    modes = [m for m in MODES if all(m in data[fw] for fw in RUNS)]
    series = [("crypten", "CrypTen (gloo/TCP)", "#9aa5b1"), ("prix_gloo", "PriX (gloo/TCP)", "#bfdbfe"),
              ("prix", "PriX (own TCP, default)", "#2563eb"), ("prix_shm", "PriX (shared memory)", "#1e3a8a")]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.0), gridspec_kw={"width_ratios": [1.35, 1]})

    ax = axes[0]
    x = np.arange(len(modes))
    w = 0.2
    for i, (fw, label, color) in enumerate(series):
        ax.bar(x + (i - 1.5) * w, [data[fw][m]["llm"][model] for m in modes], w, color=color, label=label)
    for xi, m in zip(x, modes):
        c, p = data["crypten"][m]["llm"][model], data["prix"][m]["llm"][model]
        ax.annotate(f"{c / p:.1f}×", (xi, c), xytext=(0, 4), textcoords="offset points",
                    ha="center", fontsize=8.5, fontweight="bold", color="#1d4ed8")
    ax.set_xticks(x, [MODE_LABELS[m] for m in modes])
    ax.set_ylabel("Latency (ms, lower is better)")
    ax.set_title("GPT-2 block on CPU\n(labels: PriX speedup with its default TCP transport)", fontsize=10.5)
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    ax.set_ylim(0, max(data["crypten"][m]["llm"][model] for m in modes) * 1.25)
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)

    ax = axes[1]
    labels, vals = [], {k: [] for k in ("gloo", "tcp", "shm")}
    for section, name in (("llm", "Transformer\nblocks"), ("micro", "Primitives")):
        s = summarize(rows(data, section))
        labels.append(name)
        for k in vals:
            vals[k].append(s[k])
    xx = np.arange(len(labels))
    for i, (k, label, color) in enumerate((("gloo", "PriX over gloo", "#bfdbfe"),
                                           ("tcp", "PriX over own TCP (default)", "#2563eb"),
                                           ("shm", "PriX over shared memory", "#1e3a8a"))):
        bars = ax.bar(xx + (i - 1) * 0.26, vals[k], 0.26, color=color, label=label)
        for b in bars:
            ax.annotate(f"{b.get_height():.2f}×", (b.get_x() + b.get_width() / 2, b.get_height()),
                        xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8.5, fontweight="bold")
    ax.axhline(1.0, color="#6b7280", lw=1, ls="--")
    ax.set_xticks(xx, labels)
    ax.set_ylabel("PriX speedup over CrypTen (geomean, ×)")
    ax.set_title("Geometric-mean speedup, all modes,\nby PriX transport", fontsize=10.5)
    ax.legend(frameon=False, loc="upper left", fontsize=8.5)
    ax.set_ylim(0, max(max(v) for v in vals.values()) * 1.45)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{out_base}.{ext}", dpi=200 if ext == "png" else None, bbox_inches="tight")
    paper_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "paper", "figures")
    if os.path.isdir(paper_dir):
        fig.savefig(os.path.join(paper_dir, os.path.basename(out_base) + ".pdf"), bbox_inches="tight")
    print(f"wrote {out_base}.{{png,pdf}}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results")
    ap.add_argument("--fig", default=None, help="output path (without extension) for the figure")
    args = ap.parse_args()
    with open(args.results) as f:
        data = json.load(f)
    missing = [fw for fw in RUNS if fw not in data]
    if missing:
        raise SystemExit(f"results file is missing runs: {missing}")
    print_report(data)
    if args.fig:
        plot(data, args.fig)


if __name__ == "__main__":
    main()
