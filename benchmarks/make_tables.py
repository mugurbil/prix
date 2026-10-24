#!/usr/bin/env python3
"""
Generate every number-bearing table of the paper, README and docs from `benchmarks/results/*.json`.

    python benchmarks/make_tables.py            # write paper/tables/*.tex, README/docs blocks, docs/provenance.md
    python benchmarks/make_tables.py --check    # exit 1 if any generated output differs from what is checked in

Inputs (all optional; a missing file just skips its tables):
    results/cpu_<host>.json           benchmark_prix_vs_crypten.py (+ --prix-transport gloo/shm runs)   -> CPU tables, transport table
    results/gpu_<name>.json           same, --device cuda (Colab)                                        -> GPU tables
    results/accuracy_sweep.json       accuracy_sweep.py                                                   -> accuracy table, docs ranges
    results/fss_modes_cpu.json        benchmark_fss_modes.py                                              -> FSS primitives / key-size tables
    results/fss_trunc_ablation_cpu.json  benchmark_fss_modes.py --ablation                                -> truncation ablation table
Outputs:
    paper/tables/*.tex                tabular bodies, \\input{} by paper/main.tex
    paper/tables/numbers.tex          \\newcommand macros for every number quoted in prose
    README.md, docs/benchmarks.md     blocks between <!-- generated:NAME:begin --> / <!-- generated:NAME:end -->
    docs/provenance.md                table -> script -> file -> commit map
"""

import argparse
import glob
import json
import math
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, ".."))
RES = os.path.join(HERE, "results")
TABLES = os.path.join(ROOT, "paper", "tables")

MODES5 = ["crypten", "poly", "shaft", "curl", "fission"]
MODE_TITLES = {"crypten": "CrypTen Approx.", "poly": "Piecewise Poly", "shaft": "SHAFT (Fourier+ODE)",
               "curl": "Curl Wavelet LUTs", "fission": "Fission Evaluator"}
MODE_SHORT = {"crypten": "CrypTen approx.", "poly": "Piecewise poly", "shaft": "SHAFT", "curl": "Curl LUTs", "fission": "Fission"}
LLM_ROWS = [("BertTiny (2-Block, seq=16)", "BertTiny (2-Blk)"), ("BertBase (1-Block, seq=16)", "BertBase (1-Blk)"),
            ("GPT2 (1-Block, seq=16)", "GPT2 (1-Blk)"), ("Llama (1-Block, d=768, seq=16)", "Llama (1-Blk)"),
            ("ModernBert (1-Block, seq=16)", "ModernBert (1-Blk)")]
OP_ROWS = [("Private MatMul (128x128)", r"Private MatMul ($128\times 128$)"), ("Private Mul (128x128)", r"Private Mul ($128\times 128$)"),
           ("Comparison / ReLU (32x64)", r"Comparison / ReLU ($32\times 64$)"), ("Exp (32x64)", r"Exp ($32\times 64$)"),
           ("Reciprocal (32x64)", r"Reciprocal ($32\times 64$)"), ("InvSqrt (32x64)", r"InvSqrt ($32\times 64$)"),
           ("GeLU (32x64)", r"GeLU ($32\times 64$)"), ("SiLU (32x64)", r"SiLU ($32\times 64$)"),
           ("Softmax (4x32x32)", r"Softmax ($4\times 32\times 32$)"), ("LayerNorm (4x32x64)", r"LayerNorm ($4\times 32\times 64$)")]
FSS_COLS = [("default", "PriX default", r'\texttt{mode="default"}'), ("wave_det", "Wave", r'\texttt{lut\_indexing="wave"}'),
            ("wave_sigma_tr", "Wave + SIGMA TR", r'\texttt{lut\_indexing="sigma"}'), ("wave_prob", "Wave-Prob", r'\texttt{lut\_indexing="wave\_prob"}'),
            ("sigma", "SIGMA", r'\texttt{mode="sigma"}'), ("sigma_ext", "SIGMA + ext. LN", r'\texttt{mode="sigma"}, wide $1/\sqrt{x}$'),
            ("curl", "Curl", r'\texttt{mode="curl"}'), ("shaft", "SHAFT", r'\texttt{mode="shaft"}')]
ABL_COLS = [("wave_det_egk", r"\texttt{wave\_det\_egk}", r'\texttt{lut\_indexing="wave", truncation="egk"}'),
            ("wave_det_crypten", r"\texttt{wave\_det\_crypten}", r'\texttt{lut\_indexing="wave", truncation="crypten"}'),
            ("wave_prob_egk", r"\texttt{wave\_prob\_egk}", r'\texttt{lut\_indexing="wave\_prob", truncation="egk"}'),
            ("wave_prob_crypten", r"\texttt{wave\_prob\_crypten}", r'\texttt{lut\_indexing="wave\_prob", truncation="crypten"}')]
ACC_ROWS = [("exp(neg)", r"$\exp$, $[-8, 0]$"), ("exp(mixed)", r"$\exp$, $[-4, 2]$"), ("log", r"$\log$, $[0.1, 50]$"),
            ("reciprocal", r"$1/x$, $[0.1, 50]$"), ("inv_sqrt", r"$1/\sqrt{x}$, $[0.05, 50]$"), ("sigmoid", r"Sigmoid, $[-8, 8]$"),
            ("tanh", r"Tanh, $[-6, 6]$"), ("erf", r"Erf, $[-4, 4]$"), ("gelu", r"GeLU, $[-6, 6]$"), ("silu", r"SiLU, $[-8, 8]$"),
            ("softmax", r"Softmax ($L=32$)"), ("layernorm", r"LayerNorm ($d=64$)")]

MACROS = {}
PROVENANCE = []


# ----------------------------------------------------------------------------- helpers
def load(name):
    p = os.path.join(RES, name)
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)


def env_of(d):
    return (d or {}).get("_env") or (d or {}).get("metadata") or {}


def geomean(xs):
    xs = [x for x in xs if x and x > 0]
    return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else float("nan")


def sci(x):
    """Short scientific/decimal format used in the accuracy and FSS tables."""
    if x == 0:
        return "0"
    if x >= 1e4:
        return f"{x:.1e}".replace("e+0", "e").replace("e+", "e")
    if x >= 0.01:
        return f"{x:.3g}" if x < 1 else f"{x:.2f}".rstrip("0").rstrip(".")
    m, e = f"{x:.0e}".split("e")
    return f"{m}e-{int(e[1:])}"


def texsci(x):
    m, e = f"{x:.2e}".split("e")
    return f"{m}\\text{{e}}{int(e)}"


def macro(name, value):
    name = re.sub(r"[^A-Za-z]", "", name)
    MACROS[name] = value


def bold_best(vals, fmts, lower=True):
    best = (min if lower else max)(v for v in vals if v is not None)
    return [(f"\\textbf{{{fm}}}" if v == best else fm) for v, fm in zip(vals, fmts)]


def tabular(colspec, body):
    return "\\begin{tabular}{" + colspec + "}\n\\toprule\n" + body + "\\bottomrule\n\\end{tabular}\n"


def body(lines):
    """Rows for \\input{} inside a tabular: the last row's `\\\\` is supplied by main.tex (TeX's end-of-file
    token would otherwise sit between `\\\\` and `\\bottomrule` and trigger 'Misplaced \\noalign')."""
    text = "\n".join(lines)
    return (text[: -len(r" \\")] if text.endswith(r" \\") else text) + "\n"


def write(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    old = open(path).read() if os.path.exists(path) else None
    if old != text:
        open(path, "w").write(text)
    return old != text


def replace_block(path, name, body):
    s = open(path).read()
    b, e = f"<!-- generated:{name}:begin -->", f"<!-- generated:{name}:end -->"
    if b not in s or e not in s:
        return False
    new = s[: s.index(b) + len(b)] + "\n" + body.strip("\n") + "\n" + s[s.index(e):]
    return write(path, new)


# ----------------------------------------------------------------------------- CrypTen comparison
def crypten_tables(d, tag, device_label):
    """tab:<tag>_llm and tab:<tag>_ops bodies + speedup macros."""
    if d is None or "crypten" not in d or "prix" not in d:
        return
    env = env_of(d)
    PROVENANCE.append((f"tab:{tag}_llm, tab:{tag}_ops, fig:llm_*", "benchmark_prix_vs_crypten.py", f"results/{tag_file[tag]}", env.get("prix_commit") or env.get("git_commit"), env.get("timestamp")))
    sp_llm, sp_ops = [], []
    # LLM table
    lines = []
    for key, label in LLM_ROWS:
        cells = []
        for m in MODES5:
            ct, px = d["crypten"][m]["llm"][key], d["prix"][m]["llm"][key]
            sp = ct / px
            sp_llm.append(sp)
            cells.append(f"{ct:.2f} & \\textbf{{{px:.2f} ({sp:.2f}$\\times$)}}")
        lines.append(f"{label:<19}& " + " & ".join(cells) + r" \\")
    write(os.path.join(TABLES, f"{tag}_llm.tex"), body(lines))
    # ops table
    lines = []
    for key, label in OP_ROWS:
        cells = []
        for m in MODES5:
            ct, px = d["crypten"][m]["micro"][key], d["prix"][m]["micro"][key]
            sp = ct / px
            sp_ops.append(sp)
            cells.append(f"{ct:.2f} / \\textbf{{{px:.2f} ({sp:.2f}$\\times$)}}")
        lines.append(f"{label:<36}& " + " & ".join(cells) + r" \\")
    write(os.path.join(TABLES, f"{tag}_ops.tex"), body(lines))
    macro(f"{tag}LlmSpeedupMin", f"{min(sp_llm):.2f}"); macro(f"{tag}LlmSpeedupMax", f"{max(sp_llm):.2f}")
    macro(f"{tag}LlmSpeedupGeomean", f"{geomean(sp_llm):.2f}")
    macro(f"{tag}OpsSpeedupMin", f"{min(sp_ops):.1f}"); macro(f"{tag}OpsSpeedupMax", f"{max(sp_ops):.1f}")
    macro(f"{tag}AllFaster", "yes" if min(sp_llm + sp_ops) > 1 else "no")
    return sp_llm, sp_ops


def transport_table(cpu, gpu):
    """tab:ablation body: geomean speedup per mode for TCP / gloo / shm, blocks and primitives."""
    sys.path.insert(0, HERE)
    from transport_ablation import rows, summarize  # noqa: E402

    lines = []
    for d, label in ((cpu, "CPU"), (gpu, "GPU")):
        if d is None or not all(k in d for k in ("crypten", "prix", "prix_gloo", "prix_shm")):
            continue
        env = env_of(d)
        host = env.get("cpu_model") if label == "CPU" else env.get("gpu_model")
        lines.append(rf"\emph{{{label}{' (' + host + ')' if host else ''}}} & & & & & & \\")  # plain row: \multicolumn cannot follow \input
        for m in MODES5:
            cells = []
            for section in ("llm", "micro"):
                rs = [r for r in rows(d, section) if r["mode"] == m]
                s = summarize(rs)
                cells += [f"{s['tcp']:.2f}$\\times$", f"{s['gloo']:.2f}$\\times$", f"{s['shm']:.2f}$\\times$"]
            lines.append(f"{MODE_SHORT[m]:<17}& " + " & ".join(cells) + r" \\")
        cells = []
        for section in ("llm", "micro"):
            s = summarize(rows(d, section))
            cells += [f"\\textbf{{{s['tcp']:.2f}$\\times$}}", f"\\textbf{{{s['gloo']:.2f}$\\times$}}", f"\\textbf{{{s['shm']:.2f}$\\times$}}"]
            macro(f"{label}{'Llm' if section == 'llm' else 'Ops'}TcpGeomean", f"{s['tcp']:.2f}")
            macro(f"{label}{'Llm' if section == 'llm' else 'Ops'}GlooGeomean", f"{s['gloo']:.2f}")
            macro(f"{label}{'Llm' if section == 'llm' else 'Ops'}ShmGeomean", f"{s['shm']:.2f}")
            macro(f"{label}{'Llm' if section == 'llm' else 'Ops'}TcpVsGloo", f"{s['tcp_vs_gloo']:.2f}")
            macro(f"{label}{'Llm' if section == 'llm' else 'Ops'}ShmVsTcp", f"{s['shm_vs_tcp']:.2f}")
        lines.append(rf"\textbf{{All modes ({label})}} & " + " & ".join(cells) + r" \\")
        lines.append(r"\midrule")
    if lines and lines[-1] == r"\midrule":
        lines.pop()
    if lines:
        write(os.path.join(TABLES, "transport.tex"), body(lines))


# ----------------------------------------------------------------------------- accuracy
def accuracy_table(a):
    if a is None:
        return
    env = env_of(a)
    PROVENANCE.append(("tab:accuracy, docs ranges", "accuracy_sweep.py", "results/accuracy_sweep.json", env.get("git_commit"), env.get("timestamp")))
    modes = [m for m in ["default", "wave", "sigma", "curl", "poly", "shaft", "crypten", "fission"] if m in a]

    def cell(st):
        med, worst = st["median_abs"], st["worst_run_max_abs"]
        f_med = sci(med)
        f_worst = sci(worst)
        if worst > 1:
            f_worst = f"\\textbf{{{f_worst}}}$^\\dagger$"
        elif worst > 0.1:
            f_worst = f"\\textbf{{{f_worst}}}"
        if med > 0.1:
            f_med = f"\\textbf{{{f_med}}}"
        return f"{f_med} / {f_worst}"

    header = r"\textbf{Function (inputs)} & " + " & ".join(f"\\texttt{{{m}}}" for m in modes) + r" \\"
    lines = [header, r"\midrule"]
    for key, label in ACC_ROWS:
        cells = [cell(a[m][key]) if key in a[m] else "--" for m in modes]
        lines.append(f"{label:<26}& " + " & ".join(cells) + r" \\")
    write(os.path.join(TABLES, "accuracy.tex"), body(lines))
    # markdown for docs
    md = ["| Function (inputs) | " + " | ".join(f"`{m}`" for m in modes) + " |", "|---|" + "---:|" * len(modes)]
    for key, label in ACC_ROWS:
        lab = label.replace("$", "").replace("\\", "")
        md.append(f"| {lab} | " + " | ".join((f"{sci(a[m][key]['median_abs'])} / {sci(a[m][key]['worst_run_max_abs'])}" if key in a[m] else "--") for m in modes) + " |")
    md.append("")
    md.append("Each cell is median absolute error / worst-case absolute error over the test grid (10 runs); worst-case values above 1 indicate an input-range violation, not approximation error.")
    return "\n".join(md)


# ----------------------------------------------------------------------------- FSS tables
def fss_tables(d):
    if d is None:
        return None
    env = env_of(d)
    PROVENANCE.append(("tab:wave_primitives, tab:keysize, README table", "benchmark_fss_modes.py", "results/fss_modes_cpu.json", env.get("git_commit"), env.get("timestamp")))
    cols = [c for c in FSS_COLS if c[0] in d]
    keys = [c[0] for c in cols]

    def r3(label, metric_rows):
        out = [f"\\multirow{{{len(metric_rows)}}}{{*}}{{{label}}}"]
        for name, cells in metric_rows:
            out.append(f" & {name} & " + " & ".join(cells) + r" \\")
        out.append(r"\midrule")
        return out

    # primitives (8 functions): rounds / online B, key, time / MAE
    lines = []
    for prim, label in [("gelu", r"$\operatorname{GeLU}$"), ("silu", r"$\operatorname{SiLU}$"), ("sigmoid", r"$\operatorname{Sigmoid}$"),
                        ("tanh", r"$\operatorname{Tanh}$"), ("erf", r"$\operatorname{Erf}$"), ("exp", r"$\operatorname{nexp}$"),
                        ("reciprocal", r"$1/x$"), ("inv_sqrt", r"$1/\sqrt{x}$")]:
        P = [d[k]["primitives"].get(prim) for k in keys]
        if any(p is None for p in P):
            continue
        rds = [p["rounds"] for p in P]; ob = [p["online_bytes_per_elem"] for p in P]
        kb = [p["key_bytes_per_elem_l128"] for p in P]; tm = [p["time_ms"] for p in P]; mae = [p["mae"] for p in P]
        rows_ = [("Rounds / Online (B)", bold_best(rds, [f"{r} / {b:.1f}" for r, b in zip(rds, ob)])),
                 (r"Key (B, $\lambda{=}128$)", bold_best(kb, [f"{v:,.0f}" for v in kb])),
                 ("Time / MAE", [f"${t:.1f}\\,\\text{{ms}}$ / ${texsci(m)}$" for t, m in zip(tm, mae)])]
        lines += r3(label, rows_)
    lines = lines[:-1]
    header = (r"\textbf{Primitive} & \textbf{Metric} & " + " & ".join(f"\\textbf{{{c[1]}}}" for c in cols) + r" \\" + "\n"
              + " & & " + " & ".join(f"({c[2]})" for c in cols) + r" \\" + "\n" + r"\midrule" + "\n")
    write(os.path.join(TABLES, "fss_primitives.tex"), tabular("@{}ll" + "c" * len(cols) + "@{}", header + "\n".join(lines) + "\n"))

    # key-size / end-to-end table
    lines = []
    for sub, label in [("gelu_128x3072", r"$\operatorname{GeLU}$ $(1,128,3072)$"), ("softmax_12x128x128", r"$\operatorname{Softmax}$ $(1,12,128,128)$"),
                       ("layernorm_128x768", r"$\operatorname{LayerNorm}$ $(1,128,768)$")]:
        P = [d[k]["sublayers"].get(sub) for k in keys]
        if any(p is None for p in P):
            continue
        rds = [p["rounds"] for p in P]; ob = [p["online_bytes_1p"] / 1e6 for p in P]
        kb = [p["key_bytes_1p_l128"] / 1e9 for p in P]; tm = [p["time_s"] for p in P]; mae = [p["mae"] for p in P]
        lines += r3(label, [("Rounds / Online (MB)", bold_best(rds, [f"{r} / {b:.1f}" for r, b in zip(rds, ob)])),
                            (r"Key (GB, $\lambda{=}128$)", bold_best(kb, [f"{v:.3f}" for v in kb])),
                            ("Time / MAE", [f"${t:.2f}\\,\\text{{s}}$ / ${texsci(m)}$" for t, m in zip(tm, mae)])])
    for mod, label in [("bert_base", r"\texttt{BERT-base} (12L)"), ("bert_large", r"\texttt{BERT-large} (24L)"),
                       ("gpt2_64", r"\texttt{GPT-2} (12L, $L{=}64$)"), ("gpt2_128", r"\texttt{GPT-2} (12L, $L{=}128$)")]:
        P = [d[k]["models"].get(mod) for k in keys]
        if any(p is None for p in P):
            continue
        rds = [p["total_rounds"] for p in P]; ob = [p["total_online_gb_1p"] for p in P]
        kb = [p["total_key_gb_1p_l128"] for p in P]; tm = [p["total_time_s"] for p in P]; mae = [p.get("mae") for p in P]
        rows_ = [("Rounds / Online 1P (GB)", bold_best(rds, [f"{r:,} / {b:.3f}" for r, b in zip(rds, ob)])),
                 (r"Key 1P (GB, $\lambda{=}128$)", bold_best(kb, [f"{v:.2f}" for v in kb])),
                 ("Time", bold_best(tm, [f"${v:.1f}\\,\\text{{s}}$" for v in tm]))]
        if all(m is not None for m in mae):
            rows_.append(("Block MAE", bold_best(mae, [f"${v:.4f}$" for v in mae])))
        lines += r3(label, rows_)
        if mod == "bert_base":
            for k, p in zip(keys, P):
                K = k.replace("_", " ").title().replace(" ", "")
                macro(f"BertBase{K}Rounds", f"{p['total_rounds']:,}"); macro(f"BertBase{K}OnlineGB", f"{p['total_online_gb_1p']:.2f}")
                macro(f"BertBase{K}KeyGB", f"{p['total_key_gb_1p_l128']:.1f}"); macro(f"BertBase{K}TimeS", f"{p['total_time_s']:.1f}")
                if p.get("mae") is not None:
                    macro(f"BertBase{K}MAE", f"{p['mae']:.4f}")
    lines = lines[:-1]
    header = (r"\textbf{Benchmark} & \textbf{Metric} & " + " & ".join(f"\\textbf{{{c[1]}}}" for c in cols) + r" \\" + "\n"
              + " & & " + " & ".join(f"({c[2]})" for c in cols) + r" \\" + "\n" + r"\midrule" + "\n")
    write(os.path.join(TABLES, "fss_keysize.tex"), tabular("@{}ll" + "c" * len(cols) + "@{}", header + "\n".join(lines) + "\n"))
    # derived macros (default vs sigma, etc.)
    if "default" in d and "sigma" in d:
        a, b = d["default"]["models"]["bert_base"], d["sigma"]["models"]["bert_base"]
        macro("DefaultVsSigmaRoundsPct", f"{100 * (1 - a['total_rounds'] / b['total_rounds']):.0f}")
        macro("DefaultVsSigmaOnlinePct", f"{100 * (1 - a['total_online_gb_1p'] / b['total_online_gb_1p']):.0f}")
        macro("DefaultVsSigmaKeyPct", f"{100 * (1 - a['total_key_gb_1p_l128'] / b['total_key_gb_1p_l128']):.0f}")
        macro("DefaultVsSigmaTimeX", f"{b['total_time_s'] / a['total_time_s']:.1f}")
        if a.get("mae") and b.get("mae"):
            macro("DefaultVsSigmaMaeX", f"{b['mae'] / a['mae']:.1f}")
    if "sigma" in d:
        g = d["sigma"]["primitives"]["gelu"]["key_bytes_per_elem_l128"]
        macro("SigmaGeluKeyBytes", f"{g:,.0f}"); macro("SigmaGeluKeyMatchPct", f"{100 * abs(g / 1430 - 1):.0f}")
    # README block
    desc = {"default": "Wave bior(5,3) FSS tables + SIGMA Truncate-Reduce + `[EGK+20]`", "wave_det": "2-round Wave (Grotto DPF + DCF)",
            "wave_sigma_tr": "Wave tables, SIGMA Truncate-Reduce indexing (3 rounds)", "wave_prob": "Wave tables, probabilistic indexing",
            "sigma": "SIGMA 0th-order Haar FSS LUTs", "sigma_ext": "SIGMA with the default preset's LayerNorm domain",
            "curl": "Curl wavelet LUTs (one-hot) + `[EGK+20]`", "shaft": "SHAFT Fourier activations + ODE Softmax"}
    cfgname = {"default": "`default`", "wave_det": "`wave`", "wave_sigma_tr": "`wave`, `lut_indexing=\"sigma\"`", "wave_prob": "`wave`, `lut_indexing=\"wave_prob\"`",
               "sigma": "`sigma`", "sigma_ext": "`sigma` + wide `inv_sqrt`", "curl": "`curl`", "shaft": "`shaft`"}
    P = {k: d[k]["models"].get("bert_base") for k in keys}
    P = {k: p for k, p in P.items() if p}
    best = {m: (min if m != "x" else max)(p[m] for p in P.values()) for m in ("total_rounds", "total_online_gb_1p", "total_key_gb_1p_l128", "total_time_s", "mae")}
    def c(k, m, fmt):
        v = P[k][m]; t = fmt.format(v)
        return f"**{t}**" if v == best[m] else t
    md = ["| `mode` | Nonlinearities | Rounds | Online | Key (λ=128) | CPU time | Block MAE |", "|---|---|---:|---:|---:|---:|---:|"]
    for k in keys:
        if k in P:
            md.append(f"| {cfgname[k]} | {desc[k]} | {c(k,'total_rounds','{:,}')} | {c(k,'total_online_gb_1p','{:.2f} GB')} | {c(k,'total_key_gb_1p_l128','{:.1f} GB')} | {c(k,'total_time_s','{:.1f} s')} | {c(k,'mae','{:.4f}')} |")
    md.append("| `crypten` / `poly` / `fission` | CrypTen Newton/limit · PUMA polynomials · hybrid evaluator | see [benchmarks](docs/benchmarks.md) | | | | |")
    return "\n".join(md)


def ablation_table(d):
    if d is None:
        return
    env = env_of(d)
    PROVENANCE.append(("tab:wave_trunc_ablation", "benchmark_fss_modes.py --ablation", "results/fss_trunc_ablation_cpu.json", env.get("git_commit"), env.get("timestamp")))
    keys = [c[0] for c in ABL_COLS if c[0] in d]
    lines = []
    for sub, label in [("gelu_128x3072", r"$\operatorname{GeLU}$ $(1, 128, 3072)$"), ("softmax_12x128x128", r"$\operatorname{Softmax}$ $(1, 12, 128, 128)$"),
                       ("layernorm_128x768", r"$\operatorname{LayerNorm}$ $(1, 128, 768)$")]:
        P = [d[k]["sublayers"][sub] for k in keys]
        rds = [p["rounds"] for p in P]; mb = [2 * p["online_bytes_1p"] / 1e6 for p in P]
        tm = [p["time_s"] for p in P]; mae = [p["mae"] for p in P]; mx = [p["max_err"] for p in P]
        lines.append(f"\\multirow{{2}}{{*}}{{{label}}}")
        lines.append(" & Rounds / 2P Comm & " + " & ".join(bold_best(rds, [f"{r} rds / ${b:.2f}\\,\\text{{MB}}$" for r, b in zip(rds, mb)])) + r" \\")
        lines.append(" & Time / MAE / Max & " + " & ".join(f"${t:.3f}\\,\\text{{s}}$ / ${texsci(a)}$ / ${texsci(b)}$" for t, a, b in zip(tm, mae, mx)) + r" \\")
        lines.append(r"\midrule")
    for mod, label in [("bert_base", r"\texttt{BERT-base} (12L, $L=128$)"), ("bert_large", r"\texttt{BERT-large} (24L, $L=128$)"), ("gpt2_64", r"\texttt{GPT-2} (12L, $L=64$)")]:
        P = [d[k]["models"].get(mod) for k in keys]
        if any(p is None for p in P):
            continue
        rds = [p["total_rounds"] for p in P]; gb = [p["total_online_gb_2p"] for p in P]
        tm = [p["total_time_s"] for p in P]; mae = [p["mae"] for p in P]; mx = [p["max_err"] for p in P]
        lines.append(f"\\multirow{{3}}{{*}}{{{label}}}")
        lines.append(" & Rounds / 2P Comm & " + " & ".join(bold_best(rds, [f"{r:,} rds / ${b:.3f}\\,\\text{{GB}}$" for r, b in zip(rds, gb)])) + r" \\")
        lines.append(" & Total Time & " + " & ".join(bold_best(tm, [f"${v:.2f}\\,\\text{{s}}$" for v in tm])) + r" \\")
        lines.append(" & Block MAE / Max & " + " & ".join(bold_best(mae, [f"${a:.4f}$ / ${b:.4f}$" for a, b in zip(mae, mx)])) + r" \\")
        lines.append(r"\midrule")
    lines = lines[:-1]
    cols = [c for c in ABL_COLS if c[0] in d]
    header = (r"\textbf{Benchmark} & \textbf{Metric} & " + " & ".join(f"\\textbf{{{c[1]}}}" for c in cols) + r" \\" + "\n"
              + " & & " + " & ".join(f"({c[2]})" for c in cols) + r" \\" + "\n" + r"\midrule" + "\n")
    write(os.path.join(TABLES, "fss_trunc_ablation.tex"), tabular("@{}ll" + "c" * len(cols) + "@{}", header + "\n".join(lines) + "\n"))


# ----------------------------------------------------------------------------- outputs
def write_macros():
    lines = ["% Generated by benchmarks/make_tables.py -- do not edit by hand.", "% Every number quoted in prose comes from benchmarks/results/*.json through these macros."]
    for k in sorted(MACROS):
        lines.append(f"\\newcommand{{\\{k}}}{{{MACROS[k]}}}")
    write(os.path.join(TABLES, "numbers.tex"), "\n".join(lines) + "\n")


def write_provenance():
    lines = ["# Provenance of every reported number", "", "Generated by `python benchmarks/make_tables.py`. Each table in the paper / README / docs is produced from a results file written by the listed script at the listed commit; re-run the script, then the generator, to refresh.", "",
             "| Table(s) | Script | Results file | PriX commit | Measured |", "|---|---|---|---|---|"]
    for t, s, f, c, ts in PROVENANCE:
        lines.append(f"| {t} | `benchmarks/{s}` | `benchmarks/{f}` | `{c or '?'}` | {ts or '?'} |")
    lines += ["", "GPU results are produced on Google Colab with `examples/prix_vs_curl_gpu_colab.ipynb` and `examples/prix_fss_wave_sigma_gpu_colab.ipynb` and copied into `benchmarks/results/`."]
    write(os.path.join(ROOT, "docs", "provenance.md"), "\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cpu", default=None, help="CrypTen-comparison CPU results JSON (default: newest results/cpu_*.json)")
    ap.add_argument("--gpu", default=None, help="CrypTen-comparison GPU results JSON (default: newest results/gpu_*.json)")
    ap.add_argument("--check", action="store_true", help="fail if any generated file would change")
    args = ap.parse_args()
    global tag_file
    cpu_path = args.cpu or (sorted(glob.glob(os.path.join(RES, "cpu_*.json")), key=os.path.getmtime) or [None])[-1]
    gpu_path = args.gpu or (sorted(glob.glob(os.path.join(RES, "gpu_*.json")), key=os.path.getmtime) or [None])[-1]
    tag_file = {"cpu": os.path.basename(cpu_path) if cpu_path else "?", "gpu": os.path.basename(gpu_path) if gpu_path else "?"}
    cpu = json.load(open(cpu_path)) if cpu_path else None
    gpu = json.load(open(gpu_path)) if gpu_path else None

    if args.check:
        import subprocess
        before = subprocess.check_output(["git", "-C", ROOT, "status", "--porcelain", "paper/tables", "README.md", "docs"], text=True)

    crypten_tables(cpu, "cpu", "CPU")
    crypten_tables(gpu, "gpu", "GPU")
    transport_table(cpu, gpu)
    acc_md = accuracy_table(load("accuracy_sweep.json"))
    fss_md = fss_tables(load("fss_modes_cpu.json"))
    ablation_table(load("fss_trunc_ablation_cpu.json"))
    write_macros()
    write_provenance()
    if fss_md:
        replace_block(os.path.join(ROOT, "README.md"), "fss_bert_base", fss_md)
        replace_block(os.path.join(ROOT, "docs", "benchmarks.md"), "fss_bert_base", "\n".join(fss_md.splitlines()[:-1]))
    if acc_md:
        replace_block(os.path.join(ROOT, "docs", "modes_and_configuration.md"), "accuracy_ranges", acc_md)
    print(f"wrote {len(os.listdir(TABLES))} files to paper/tables, {len(MACROS)} macros, {len(PROVENANCE)} provenance rows")
    if args.check:
        after = subprocess.check_output(["git", "-C", ROOT, "status", "--porcelain", "paper/tables", "README.md", "docs"], text=True)
        if before != after:
            print("ERROR: generated outputs differ from the checked-in versions:\n" + after)
            sys.exit(1)


if __name__ == "__main__":
    main()
