#!/usr/bin/env python3
"""
Render benchmark charts for the README and paper from benchmark JSON results.

    python benchmarks/plot_results.py                     # uses benchmarks/results/*.json
    python benchmarks/plot_results.py --cpu my_cpu.json --gpu my_gpu.json --out docs/figures

The JSON files are the `--out` files written by benchmark_prix_vs_crypten.py.
Each figure is written as PNG (README) and PDF to --out, plus a PDF copy in paper/figures/.
"""

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LogNorm  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

MODES = [
    ("crypten", "CrypTen\napprox."),
    ("poly", "Piecewise\npoly"),
    ("shaft", "SHAFT"),
    ("curl", "Curl\nLUTs"),
    ("fission", "Fission"),
]
MODELS = [
    ("BertTiny (2-Block, seq=16)", "BertTiny"),
    ("BertBase (1-Block, seq=16)", "BertBase"),
    ("GPT2 (1-Block, seq=16)", "GPT2"),
    ("Llama (1-Block, d=768, seq=16)", "Llama"),
    ("ModernBert (1-Block, seq=16)", "ModernBert"),
]
OPS = [
    ("Private MatMul (128x128)", "MatMul"),
    ("Private Mul (128x128)", "Mul"),
    ("Comparison / ReLU (32x64)", "ReLU"),
    ("Exp (32x64)", "Exp"),
    ("Reciprocal (32x64)", "Reciprocal"),
    ("InvSqrt (32x64)", "InvSqrt"),
    ("GeLU (32x64)", "GeLU"),
    ("SiLU (32x64)", "SiLU"),
    ("Softmax (4x32x32)", "Softmax"),
    ("LayerNorm (4x32x64)", "LayerNorm"),
]
SHORT_MODES = ["CrypTen", "Poly", "SHAFT", "Curl", "Fission"]
PLATFORMS = [("cpu", "CPU · AMD EPYC 7B12"), ("gpu", "GPU · NVIDIA Tesla T4")]

CRYPTEN_COLOR = "#9aa5b1"
PRIX_COLOR = "#2563eb"

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titleweight": "bold",
        "axes.titlesize": 11,
        "figure.dpi": 150,
        "savefig.bbox": "tight",
    }
)


def _get(data, fw, mode, section, key):
    return data[fw][mode][section][key]


def _speedups(data, section, rows):
    return np.array(
        [
            [_get(data, "crypten", m, section, k) / _get(data, "prix", m, section, k) for m, _ in MODES]
            for k, _ in rows
        ]
    )


PAPER_FIG_DIR = os.path.join(ROOT, "paper", "figures")


def _save(fig, out_dir, name):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out_dir, f"{name}.{ext}"), dpi=200 if ext == "png" else None)
    # The paper directory is self-contained (for arXiv), so it gets its own PDF copy.
    if os.path.isdir(PAPER_FIG_DIR):
        fig.savefig(os.path.join(PAPER_FIG_DIR, f"{name}.pdf"))
    plt.close(fig)
    print(f"wrote {os.path.join(out_dir, name)}.{{png,pdf}}")


def plot_llm_latency(results, out_dir, model_key="GPT2 (1-Block, seq=16)", model_label="GPT-2 block"):
    """Paired CrypTen/PriX latency bars per mode for one Transformer block, CPU and GPU."""
    fig, axes = plt.subplots(1, len(results), figsize=(5.4 * len(results), 3.6), squeeze=False)
    x = np.arange(len(MODES))
    w = 0.38
    for ax, (plat, title) in zip(axes[0], [p for p in PLATFORMS if p[0] in results]):
        data = results[plat]
        c = [_get(data, "crypten", m, "llm", model_key) for m, _ in MODES]
        p = [_get(data, "prix", m, "llm", model_key) for m, _ in MODES]
        ax.bar(x - w / 2, c, w, color=CRYPTEN_COLOR, label="CrypTen (PyTorch)")
        ax.bar(x + w / 2, p, w, color=PRIX_COLOR, label="PriX (JAX)")
        for xi, cv, pv in zip(x, c, p):
            ax.annotate(
                f"{cv / pv:.1f}×",
                (xi, max(cv, pv)),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                fontsize=9,
                fontweight="bold",
                color=PRIX_COLOR,
            )
        ax.set_xticks(x, [label for _, label in MODES])
        ax.set_ylabel("Latency (ms, lower is better)")
        ax.set_title(f"{title}: {model_label}")
        ax.set_ylim(0, max(c) * 1.18)
        ax.grid(axis="y", alpha=0.3)
        ax.set_axisbelow(True)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle(
        "End-to-end private inference latency (2 parties, seq=16) and PriX speedup",
        fontsize=12,
        fontweight="bold",
        y=1.09,
    )
    _save(fig, out_dir, "llm_latency")


def _heatmap(ax, values, row_labels, title, norm, show_rows=True, cmap="YlGnBu"):
    im = ax.imshow(values, cmap=cmap, norm=norm, aspect="auto")
    ax.set_xticks(range(len(MODES)), SHORT_MODES)
    ax.set_yticks(range(len(row_labels)), row_labels if show_rows else [""] * len(row_labels))
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks(np.arange(len(MODES) + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(row_labels) + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="white", linewidth=2)
    ax.tick_params(which="minor", length=0)
    lo, hi = np.log(norm.vmin), np.log(norm.vmax)
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            v = values[i, j]
            dark = (np.log(v) - lo) / (hi - lo) > 0.55
            ax.text(j, i, f"{v:.1f}×", ha="center", va="center", fontsize=9,
                    color="white" if dark else "#1f2937", fontweight="bold")
    ax.set_title(title)
    return im


def plot_speedup_heatmaps(results, out_dir, section, rows, name, suptitle, height):
    plats = [p for p in PLATFORMS if p[0] in results]
    vals = {plat: _speedups(results[plat], section, rows) for plat, _ in plats}
    allv = np.concatenate([v.ravel() for v in vals.values()])
    norm = LogNorm(vmin=max(1.0, allv.min()), vmax=allv.max())
    fig, axes = plt.subplots(1, len(plats), figsize=(4.4 * len(plats) + 1.2, height), squeeze=False,
                             gridspec_kw={"wspace": 0.06})
    for i, (ax, (plat, title)) in enumerate(zip(axes[0], plats)):
        im = _heatmap(ax, vals[plat], [label for _, label in rows], title, norm, show_rows=(i == 0))
    cbar = fig.colorbar(im, ax=axes[0].tolist(), shrink=0.85, pad=0.02)
    cbar.set_label("PriX speedup over CrypTen (×, log scale)")
    ticks = [t for t in (1, 1.5, 2, 3, 5, 10, 20) if norm.vmin <= t <= norm.vmax]
    cbar.ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    cbar.set_ticks(ticks, labels=[f"{t:g}×" for t in ticks])
    fig.suptitle(suptitle, fontsize=12, fontweight="bold", y=1.02)
    _save(fig, out_dir, name)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cpu", default=os.path.join(HERE, "results", "cpu_amd_epyc_7b12.json"))
    ap.add_argument("--gpu", default=os.path.join(HERE, "results", "gpu_tesla_t4.json"))
    ap.add_argument("--out", default=os.path.join(ROOT, "docs", "figures"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    results = {}
    for plat, path in (("cpu", args.cpu), ("gpu", args.gpu)):
        if path and os.path.exists(path):
            with open(path) as f:
                results[plat] = json.load(f)
    if not results:
        raise SystemExit("no benchmark JSON found")

    plot_llm_latency(results, args.out)
    plot_speedup_heatmaps(
        results, args.out, "llm", MODELS, "llm_speedup",
        "PriX speedup on Transformer blocks, by approximation mode", height=3.2,
    )
    plot_speedup_heatmaps(
        results, args.out, "micro", OPS, "primitive_speedup",
        "PriX speedup on MPC primitives and nonlinearities", height=4.8,
    )


if __name__ == "__main__":
    main()
