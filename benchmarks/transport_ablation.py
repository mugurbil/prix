#!/usr/bin/env python3
"""
Transport ablation report: how much of PriX's speedup over CrypTen comes from the local
transport (shared memory vs. gloo/TCP) and how much from everything else (compiled kernels,
limb matmul, batching)?

Input: one benchmark JSON containing three runs made with --comm-stats:
    crypten    CrypTen (PyTorch), gloo over loopback TCP
    prix_gloo  PriX with its 2-party all_reduce routed over gloo (--prix-transport gloo)
    prix       PriX with its default shared-memory transport

    python benchmarks/transport_ablation.py results.json [--fig docs/figures/transport_ablation]
"""

import argparse
import json
import math
import os

import numpy as np

MODES = ["crypten", "poly", "shaft", "lut", "fission"]
MODE_LABELS = {"crypten": "CrypTen approx.", "poly": "Piecewise poly", "shaft": "SHAFT",
               "lut": "Curl LUTs", "fission": "Fission"}


def geomean(xs):
    xs = [x for x in xs if x > 0]
    return math.exp(sum(math.log(x) for x in xs) / len(xs))


def rows(data, section):
    out = []
    for mode in MODES:
        if not all(mode in data[fw] for fw in ("crypten", "prix_gloo", "prix")):
            continue
        for key in data["prix"][mode][section]:
            c = data["crypten"][mode][section][key]
            g = data["prix_gloo"][mode][section][key]
            p = data["prix"][mode][section][key]
            cc = data["crypten"][mode]["comm"][section][key]
            gc = data["prix_gloo"][mode]["comm"][section][key]
            pc = data["prix"][mode]["comm"][section][key]
            out.append(dict(mode=mode, key=key, c=c, g=g, p=p, c_comm=cc["comm_ms"], g_comm=gc["comm_ms"],
                            p_comm=pc["comm_ms"], c_rounds=cc["rounds"], p_rounds=pc["rounds"],
                            c_bytes=cc["bytes"], p_bytes=pc["bytes"]))
    return out


def summarize(rs):
    total = geomean([r["c"] / r["p"] for r in rs])
    same = geomean([r["c"] / r["g"] for r in rs])
    transport = geomean([r["g"] / r["p"] for r in rs])
    share = math.log(transport) / math.log(total) if total > 1 else float("nan")
    return total, same, transport, share


def print_report(data):
    for section, title in (("llm", "Transformer blocks"), ("micro", "Primitives")):
        rs = rows(data, section)
        print(f"\n### {title}: speedup decomposition (geometric means)")
        print("| Mode | Overall (CrypTen / PriX-shm) | Same transport (CrypTen / PriX-gloo) | "
              "Transport alone (PriX-gloo / PriX-shm) | Share of speedup from transport |")
        print("|---|---:|---:|---:|---:|")
        for mode in MODES + ["all"]:
            sel = rs if mode == "all" else [r for r in rs if r["mode"] == mode]
            if not sel:
                continue
            t, s, tr, sh = summarize(sel)
            label = "**All modes**" if mode == "all" else MODE_LABELS[mode]
            print(f"| {label} | {t:.2f}x | {s:.2f}x | {tr:.2f}x | {100 * sh:.0f}% |")

        print(f"\n### {title}: time spent in communication calls (median per run, ms; % of wall time)")
        print("| Mode | Workload | CrypTen | PriX-gloo | PriX-shm | Rounds (CrypTen / PriX) | MB sent (CrypTen / PriX) |")
        print("|---|---|---:|---:|---:|---:|---:|")
        for r in rs:
            def cell(comm, tot):
                return f"{comm:.1f} ({100 * comm / tot:.0f}%)"
            print(f"| {MODE_LABELS[r['mode']]} | {r['key']} | {cell(r['c_comm'], r['c'])} | "
                  f"{cell(r['g_comm'], r['g'])} | {cell(r['p_comm'], r['p'])} | "
                  f"{r['c_rounds']:.0f} / {r['p_rounds']:.0f} | {r['c_bytes'] / 1e6:.2f} / {r['p_bytes'] / 1e6:.2f} |")


def plot(data, out_base, model="GPT2 (1-Block, seq=16)"):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.titleweight": "bold"})
    modes = [m for m in MODES if all(m in data[fw] for fw in ("crypten", "prix_gloo", "prix"))]
    series = [("crypten", "CrypTen (gloo/TCP)", "#9aa5b1"), ("prix_gloo", "PriX (gloo/TCP)", "#93c5fd"),
              ("prix", "PriX (shared memory)", "#2563eb")]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.0), gridspec_kw={"width_ratios": [1.35, 1]})

    ax = axes[0]
    x = np.arange(len(modes))
    w = 0.26
    for i, (fw, label, color) in enumerate(series):
        tot = np.array([data[fw][m]["llm"][model] for m in modes])
        ax.bar(x + (i - 1) * w, tot, w, color=color, label=label)
    for xi, m in zip(x, modes):
        c, g, p = (data[fw][m]["llm"][model] for fw in ("crypten", "prix_gloo", "prix"))
        ax.annotate(f"{c / g:.1f}× → {c / p:.1f}×", (xi, c), xytext=(0, 4), textcoords="offset points",
                    ha="center", fontsize=8.5, fontweight="bold", color="#1d4ed8")
    ax.set_xticks(x, [MODE_LABELS[m] for m in modes])
    ax.set_ylabel("Latency (ms, lower is better)")
    ax.set_title("GPT-2 block on CPU\n(labels: PriX speedup with gloo → with shared memory)", fontsize=10.5)
    ax.legend(frameon=False, fontsize=8.5, loc="upper right")
    ax.set_ylim(0, max(data["crypten"][m]["llm"][model] for m in modes) * 1.22)
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)

    ax = axes[1]
    labels, overall, same = [], [], []
    for section, name in (("llm", "Transformer\nblocks"), ("micro", "Primitives")):
        t, s, _, _ = summarize(rows(data, section))
        labels.append(name)
        overall.append(t)
        same.append(s)
    xx = np.arange(len(labels))
    b1 = ax.bar(xx - 0.18, same, 0.36, color="#93c5fd", label="Same transport (gloo/TCP)")
    b2 = ax.bar(xx + 0.18, overall, 0.36, color="#2563eb", label="As shipped (shared memory)")
    for bars in (b1, b2):
        for b in bars:
            ax.annotate(f"{b.get_height():.2f}×", (b.get_x() + b.get_width() / 2, b.get_height()),
                        xytext=(0, 3), textcoords="offset points", ha="center", fontweight="bold")
    ax.axhline(1.0, color="#6b7280", lw=1, ls="--")
    ax.set_xticks(xx, labels)
    ax.set_ylabel("PriX speedup over CrypTen (geomean, ×)")
    ax.set_title("Geometric-mean speedup, all modes\n(same transport vs. as shipped)", fontsize=10.5)
    ax.legend(frameon=False, loc="upper left")
    ax.set_ylim(0, max(overall) * 1.45)
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
    ap.add_argument("--fig", default=None, help="output path (without extension) for the ablation figure")
    args = ap.parse_args()
    with open(args.results) as f:
        data = json.load(f)
    missing = [fw for fw in ("crypten", "prix_gloo", "prix") if fw not in data]
    if missing:
        raise SystemExit(f"results file is missing runs: {missing}")
    print_report(data)
    if args.fig:
        plot(data, args.fig)


if __name__ == "__main__":
    main()
