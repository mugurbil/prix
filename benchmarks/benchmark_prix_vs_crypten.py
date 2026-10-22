#!/usr/bin/env python3
"""
Head-to-head benchmark: CrypTen (PyTorch) vs PriX (JAX), on CPU or GPU.

The PyTorch baseline is the CrypTen-based reference implementation from
https://github.com/jimouris/curl (a CrypTen fork that adds CuRL wavelet LUTs and
Fission evaluators), augmented with SHAFT (https://github.com/andeskyl/SHAFT,
NDSS 2025). Both frameworks are benchmarked in four approximation modes:

  crypten  - CrypTen's original approximations (Newton-Raphson, limit exp,
             Taylor erf, ...) with CrypTen truncation  (configs/approximations.yaml)
  shaft    - SHAFT constant-round Fourier-series activations & ODE softmax
                                                       (configs/shaft.yaml)
  lut      - CuRL wavelet look-up tables                (configs/default.yaml)
  fission  - Fission hybrid plaintext evaluator         (configs/fission.yaml)

Each mode's YAML is deep-merged on top of the framework's default.yaml (so every
key exists) and the provider is forced to TFP for both frameworks.
"""

import argparse
import json
import math
import multiprocessing as mp
import os
import subprocess
import sys
import tempfile
import time

import numpy as np

MODES = [
    # (key, title, config file, number of Fission evaluators)
    ("crypten", "CrypTen approximations", "approximations.yaml", 0),
    ("shaft", "SHAFT (Fourier + ODE)", "shaft.yaml", 0),
    ("lut", "CuRL wavelet LUTs", "default.yaml", 0),
    ("fission", "Fission evaluator", "fission.yaml", 1),
]
MODE_KEYS = [m[0] for m in MODES]

FRAMEWORK_LABELS = {"crypten": "CrypTen (PyTorch)", "prix": "PriX (JAX)"}


class AttrDict(dict):
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key)

    def __setattr__(self, key, value):
        self[key] = value


def _bench_fn(fn, warmup, repeats, sync_fn=None, barrier=None):
    for _ in range(warmup):
        out = fn()
        if sync_fn is not None:
            sync_fn(out)
    times = []
    for _ in range(repeats):
        if barrier is not None:
            barrier()
        t0 = time.perf_counter()
        out = fn()
        if sync_fn is not None:
            sync_fn(out)
        times.append((time.perf_counter() - t0) * 1000.0)  # ms
    return float(np.median(times))


def _save_part(path, data):
    with open(path, "w") as f:
        json.dump(data, f)


def _load_part(path):
    with open(path, "r") as f:
        return json.load(f)


def _deep_merge(base, override):
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _merged_config(root, cfg_name, tmp_dir, tag, fallback_root=None):
    """default.yaml <- mode yaml, provider forced to TFP; written to tmp_dir."""
    import yaml

    with open(os.path.join(root, "configs", "default.yaml")) as f:
        base = yaml.safe_load(f)
    mode_cfg_path = os.path.join(root, "configs", cfg_name)
    if not os.path.exists(mode_cfg_path) and fallback_root is not None:
        mode_cfg_path = os.path.join(fallback_root, "configs", cfg_name)
    with open(mode_cfg_path) as f:
        mode = yaml.safe_load(f)
    merged = _deep_merge(base, mode)
    merged.setdefault("mpc", {})["provider"] = "TFP"
    merged.setdefault("functions", {}).setdefault("exp", {}).setdefault("all_neg", False)
    path = os.path.join(tmp_dir, f"_{tag}_{cfg_name}")
    with open(path, "w") as f:
        yaml.safe_dump(merged, f)
    return path


def _patch_crypten_repo(crypten_root):
    """Apply Python 3.12+/modern JAX/PyTorch compatibility fixes to a cloned jimouris/curl checkout."""
    for rel in ["curl/mpc/primitives/beaver.py", "curl/mpc/primitives/plaintext.py"]:
        p = os.path.join(crypten_root, rel)
        if os.path.exists(p):
            txt = open(p).read()
            txt_new = txt.replace("from jax.lib import xla_bridge\n", "").replace(
                "xla_bridge.get_backend().platform\n", ""
            )
            if txt_new != txt:
                open(p, "w").write(txt_new)

    onnx_p = os.path.join(crypten_root, "curl/nn/onnx_converter.py")
    if os.path.exists(onnx_p):
        txt = open(onnx_p).read()
        target = "    from torch.onnx._internal.registration import registry  # noqa"
        replacement = (
            "    try:\n"
            "        from torch.onnx._internal.registration import registry  # noqa\n"
            "    except ImportError:\n"
            "        registry = None"
        )
        if target in txt and "except ImportError:\n        registry = None" not in txt:
            open(onnx_p, "w").write(txt.replace(target, replacement))

    llama_p = os.path.join(crypten_root, "examples/llms/models/llama.py")
    if os.path.exists(llama_p):
        txt = open(llama_p).read()
        if "self.embed_dim, self.head_dim = 2048, 32" in txt:
            open(llama_p, "w").write(
                txt.replace(
                    "self.embed_dim, self.head_dim = 2048, 32",
                    "self.embed_dim, self.head_dim = embed_dim, head_dim",
                )
            )

    init_p = os.path.join(crypten_root, "curl/__init__.py")
    if os.path.exists(init_p):
        txt = open(init_p).read()
        if 'logging.getLogger("jax._src.xla_bridge")' not in txt:
            txt = 'import logging\nlogging.getLogger("jax._src.xla_bridge").setLevel(logging.CRITICAL)\n' + txt
        target = '            generators[key][device] = torch.Generator(device=torch.device(device))'
        replacement = (
            '            gen = torch.Generator(device=torch.device(device))\n'
            '            generators[key][device] = gen\n'
            '            generators[key][torch.device("cuda")] = gen\n'
            '            generators[key][torch.device("cuda", 0)] = gen'
        )
        if target in txt and 'generators[key][torch.device("cuda", 0)]' not in txt:
            txt = txt.replace(target, replacement)
        open(init_p, "w").write(txt)

    for cfg_rel in ["configs/default.yaml", "configs/fission.yaml"]:
        cfg_p = os.path.join(crypten_root, cfg_rel)
        if os.path.exists(cfg_p):
            txt = open(cfg_p).read()
            if 'provider: "TTP"' in txt:
                open(cfg_p, "w").write(txt.replace('provider: "TTP"', 'provider: "TFP"'))

    launcher_p = os.path.join(crypten_root, "examples/multiprocess_launcher.py")
    if os.path.exists(launcher_p):
        txt = open(launcher_p).read()
        target = "        if curl.mpc.ttp_required():"
        replacement = (
            "        if cfg_file is not None:\n"
            "            curl.cfg.load_config(cfg_file)\n"
            "        if curl.mpc.ttp_required():"
        )
        if target in txt and "curl.cfg.load_config(cfg_file)" not in txt:
            open(launcher_p, "w").write(txt.replace(target, replacement))

    mod_p = os.path.join(crypten_root, "curl/nn/module.py")
    if os.path.exists(mod_p):
        txt = open(mod_p).read()
        txt_new = txt.replace(r"\text{in\_features}", r"\\text{in\\_features}").replace(
            r"\text{out\_features}", r"\\text{out\\_features}"
        )
        if txt_new != txt:
            open(mod_p, "w").write(txt_new)


# ============================================================================
# Shared workload definitions (identical inputs/shapes for both frameworks)
# ============================================================================
def _micro_inputs():
    rng = np.random.RandomState(42)
    return dict(
        a=rng.uniform(-2.0, 2.0, size=(128, 128)).astype(np.float32),
        b=rng.uniform(-2.0, 2.0, size=(128, 128)).astype(np.float32),
        pos=rng.uniform(0.5, 4.0, size=(32, 64)).astype(np.float32),
        act=rng.uniform(-2.0, 2.0, size=(32, 64)).astype(np.float32),
        attn=rng.uniform(-2.0, 2.0, size=(4, 32, 32)).astype(np.float32),
        ln=rng.uniform(-2.0, 2.0, size=(4, 32, 64)).astype(np.float32),
    )


def _micro_ops(t):
    return [
        ("Private MatMul (128x128)", lambda: t["a"] @ t["b"]),
        ("Private Mul (128x128)", lambda: t["a"] * t["b"]),
        ("Comparison / ReLU (32x64)", lambda: t["act"].relu()),
        ("Exp (32x64)", lambda: t["act"].exp()),
        ("Reciprocal (32x64)", lambda: t["pos"].reciprocal()),
        ("InvSqrt (32x64)", lambda: t["pos"].inv_sqrt()),
        ("GeLU (32x64)", lambda: t["act"].gelu()),
        ("SiLU (32x64)", lambda: t["act"].silu()),
        ("Softmax (4x32x32)", lambda: t["attn"].softmax(dim=-1)),
        ("LayerNorm (4x32x64)", lambda: t["ln"].layernorm(1, 0)),
    ]


def _llm_specs():
    from examples.llms.models.bert import BertTiny, BertModel
    from examples.llms.models.gpt2 import GPT
    from examples.llms.models.llama import Llama
    from examples.llms.models.modern_bert import ModernBertModel

    return [
        ("BertTiny (2-Block, seq=16)", lambda: BertTiny(seq_len=16, full=False), (1, 16, 128)),
        (
            "BertBase (1-Block, seq=16)",
            lambda: BertModel(vocab_size=30522, emb_size=768, seq_len=16, n_heads=12, n_layers=1, full=False),
            (1, 16, 768),
        ),
        (
            "GPT2 (1-Block, seq=16)",
            lambda: GPT(embed_dim=768, num_heads=12, num_blocks=1, vocab_size=50257, seq_len=16, full=False),
            (1, 16, 768),
        ),
        (
            "Llama (1-Block, d=768, seq=16)",
            lambda: Llama(
                embed_dim=768, head_dim=12, n_layers=1, n_kv_heads=4, vocab_size=32000,
                multiple_of=256, ffn_dim_multiplier=1.0, seq_len=16, full=False, cache=False,
            ),
            (1, 16, 768),
        ),
        (
            "ModernBert (1-Block, seq=16)",
            lambda: ModernBertModel(
                vocab_size=50368, hidden_size=768, intermediate_size=1152, seq_len=16,
                n_heads=12, n_layers=1, full=False,
            ),
            (1, 16, 768),
        ),
    ]


def _run_workload(a, fw, mod, comm, to_private, sync, no_grad):
    """Runs micro + LLM benchmarks inside one MPC party (config already loaded)."""
    rank = comm.get().get_rank()
    barrier = comm.get().barrier
    tag = f"[{FRAMEWORK_LABELS[fw]} | {a['mode']} | {a['device']}]"
    out = {"micro": {}, "llm": {}}

    with no_grad():
        mod_luts = getattr(mod, "LookupTables", None) or mod.common.functions.approximations.LookupTables
        mod_luts(device=a["device"])

        t = {k: to_private(v) for k, v in _micro_inputs().items()}
        for name, fn in _micro_ops(t):
            ms = _bench_fn(fn, a["warmup"], a["repeats"], sync, barrier)
            if rank == 0:
                print(f"{tag} {name}: {ms:.2f} ms", flush=True)
                out["micro"][name] = ms

        for label, builder, shape in _llm_specs():
            m = builder().to(a["device"]).encrypt(src=0)
            m.eval()
            x_np = np.random.RandomState(123).uniform(-1.0, 1.0, size=shape).astype(np.float32)
            x = to_private(x_np)
            ms = _bench_fn(lambda: m(x), a["llm_warmup"], a["llm_repeats"], sync, barrier)
            if rank == 0:
                print(f"{tag} {label}: {ms:.2f} ms", flush=True)
                out["llm"][label] = ms

    if rank == 0:
        _save_part(a["out_path"], out)


# ============================================================================
# PriX
# ============================================================================
def _prix_worker(a):
    import prix
    import prix.communicator as comm

    prix.cfg.load_config(a["config_path"])
    device = a["device"]

    def sync(x):
        x = x[0] if isinstance(x, tuple) else x
        if hasattr(x, "share"):
            x.share.block_until_ready()

    _run_workload(
        a, "prix", prix, comm,
        to_private=lambda v: prix.cryptensor(v, device=device),
        sync=sync, no_grad=prix.no_grad,
    )


def run_prix(prix_root, device, tmp_dir, modes, bench_args):
    sys.path.insert(0, prix_root)
    from prix.launcher import MultiProcessLauncher

    results = {}
    for key, _title, cfg_name, n_eval in MODES:
        if key not in modes:
            continue
        cfg_path = _merged_config(prix_root, cfg_name, tmp_dir, "prix")
        part = os.path.join(tmp_dir, f"_prix_{key}.json")
        args = AttrDict(config_path=cfg_path, out_path=part, device=device, mode=key, **bench_args)
        launcher = MultiProcessLauncher(2, n_eval, _prix_worker, args, cfg_file=cfg_path)
        launcher.start()
        launcher.join()
        launcher.terminate()
        results[key] = _load_part(part)
    return results


# ============================================================================
# CrypTen (PyTorch, jimouris/curl + andeskyl/SHAFT)
# ============================================================================
def _install_shaft_into_curl():
    """Register SHAFT (NDSS 2025) Fourier-series activations and ODE softmax on CrypTen/CuRL."""
    import torch
    import curl
    from curl.config import cfg
    from curl.cryptensor import CrypTensor
    from curl.common.functions import approximations as approx_mod
    from curl.mpc.primitives import ArithmeticSharedTensor
    from curl.mpc.provider.provider import TupleProvider
    from curl.mpc.provider.tfp_provider import TrustedFirstParty

    if getattr(approx_mod, "_shaft_installed", False):
        return

    if "generate_trig_triple" not in TupleProvider.TRACEABLE_FUNCTIONS:
        TupleProvider.TRACEABLE_FUNCTIONS.append("generate_trig_triple")

    def generate_trig_triple(self, size, period, terms, device=None):
        t = torch.rand(size, device=device) * period
        k = [i * 2 * math.pi / period for i in range(1, terms + 1)]
        t_k = torch.stack([i * t for i in k])
        u, v = torch.sin(t_k), torch.cos(t_k)
        t = ArithmeticSharedTensor(t, precision=None, src=0)
        u = ArithmeticSharedTensor(u, precision=None, src=0)
        v = ArithmeticSharedTensor(v, precision=None, src=0)
        return t, u, v

    TrustedFirstParty.generate_trig_triple = generate_trig_triple

    def _fourier_series(self, terms, period, alpha=None, beta_cos=None, beta_sin=None):
        device = self.device
        beta_sin = beta_sin.view([-1 if i == 0 else 1 for i in range(self.dim() + 1)]).to(device)
        k = [i * 2 * math.pi / period for i in range(1, terms + 1)]
        provider = curl.mpc.get_default_provider()
        t, u, v = provider.generate_trig_triple(self.size(), period, terms, device=device)
        delta = (self - t).get_plain_text()
        delta_k = torch.stack([i * delta for i in k])
        p, q = torch.sin(delta_k).to(device), torch.cos(delta_k).to(device)
        res = ((v * p + u * q) * beta_sin).sum(dim=0)
        if alpha is not None:
            res = res + alpha
        return res

    orig_gelu = approx_mod.gelu
    orig_silu = approx_mod.silu
    orig_softmax = approx_mod.softmax

    def shaft_gelu(self):
        if cfg.functions.gelu.method == "fourier":
            period = getattr(cfg.functions.gelu, "fs_period", 8)
            width = period / 2
            terms = getattr(cfg.functions.gelu, "fs_terms", 8)
            relu_x = self.relu()
            abs_x = 2 * relu_x - self
            do_fs = abs_x < width
            beta_sin = torch.tensor(
                [-0.0818, -0.0809, -0.0424, -0.0176, -0.0079, -0.0043, -0.0026, -0.0017],
                device=self.device,
            )
            return relu_x + do_fs * _fourier_series(abs_x, terms, period, beta_sin=beta_sin)
        return orig_gelu(self)

    def shaft_silu(self):
        if cfg.functions.silu.method == "fourier":
            period = getattr(cfg.functions.silu, "fs_period", 16)
            width = period / 2
            terms = getattr(cfg.functions.silu, "fs_terms", 12)
            relu_x = self.relu()
            abs_x = 2 * relu_x - self
            do_fs = abs_x < width
            beta_sin = torch.tensor(
                [
                    -0.1299, -0.1220, -0.0743, -0.0394, -0.0216, -0.0118,
                    -0.0074, -0.0044, -0.0033, -0.0021, -0.0018, -0.0011,
                ],
                device=self.device,
            )
            return relu_x + do_fs * _fourier_series(abs_x, terms, period, beta_sin=beta_sin)
        return orig_silu(self)

    def shaft_softmax(self, dim, **kwargs):
        if cfg.functions.softmax.method == "ode":
            if self.dim() == 0:
                return self.new(torch.ones_like(self.data))
            if self.size(dim) == 1:
                return self.new(torch.ones_like(self.data))
            iter_num = int(getattr(cfg.functions.softmax, "ode_iter_num", 16))
            clip = bool(getattr(cfg.functions.softmax, "ode_clip", True))
            upper = float(getattr(cfg.functions.softmax, "ode_ub", 12))
            lower = float(getattr(cfg.functions.softmax, "ode_lb", -4))
            logits = self.clone()
            if clip:
                diff = curl.cat([logits - upper, lower - logits]).relu().split(logits.size(0))
                logits = logits + diff[1] - diff[0]
            x = logits / iter_num
            g = logits.new(torch.ones_like(logits.data) / logits.size(dim), device=logits.device)
            for _ in range(iter_num):
                g = g + (x - g.mul(x).sum(dim=dim, keepdim=True)) * g
            return g
        return orig_softmax(self, dim, **kwargs)

    approx_mod.gelu = CrypTensor.gelu = shaft_gelu
    approx_mod.silu = CrypTensor.silu = shaft_silu
    approx_mod.softmax = CrypTensor.softmax = shaft_softmax
    approx_mod._shaft_installed = True


def _crypten_worker(a):
    import torch
    import curl
    import curl.communicator as comm
    from curl.config import cfg

    _install_shaft_into_curl()
    cfg.load_config(a["config_path"])
    device = a["device"]
    cuda = str(device).startswith("cuda")

    _run_workload(
        a, "crypten", curl, comm,
        to_private=lambda v: curl.cryptensor(torch.from_numpy(v).to(device), device=device),
        sync=(lambda _: torch.cuda.synchronize()) if cuda else None,
        no_grad=curl.no_grad,
    )


def run_crypten(crypten_root, device, tmp_dir, modes, bench_args, prix_root=None):
    _patch_crypten_repo(crypten_root)
    sys.path.insert(0, crypten_root)
    orig_set_start = mp.set_start_method
    mp.set_start_method = lambda method, force=True: orig_set_start(method, force=True)
    from curl.config import cfg
    from examples.multiprocess_launcher import MultiProcessLauncher

    results = {}
    for key, _title, cfg_name, n_eval in MODES:
        if key not in modes:
            continue
        cfg_path = _merged_config(crypten_root, cfg_name, tmp_dir, "crypten", fallback_root=prix_root)
        cfg.load_config(cfg_path)
        part = os.path.join(tmp_dir, f"_crypten_{key}.json")
        args = AttrDict(
            config_path=cfg_path, out_path=part, device=device, mode=key, multi_gpu=False, **bench_args
        )
        launcher = MultiProcessLauncher(2, n_eval, _crypten_worker, args, cfg_file=cfg_path)
        launcher.start()
        launcher.join()
        launcher.terminate()
        results[key] = _load_part(part)
    return results


# ============================================================================
# Report
# ============================================================================
def _fmt(v):
    return "—" if v is None or v != v else f"{v:.2f}"


def print_report(data):
    c_all, p_all = data.get("crypten", {}), data.get("prix", {})

    print("\n### Private LLM inference: all modes (ms)")
    hdr = ["Model"]
    for key, title, _, _ in MODES:
        hdr += [f"CrypTen · {title}", f"PriX · {title}"]
    print("| " + " | ".join(hdr) + " |")
    print("|---|" + "---:|" * (len(hdr) - 1))
    labels = []
    for key in MODE_KEYS:
        for k in list(p_all.get(key, {}).get("llm", {})) + list(c_all.get(key, {}).get("llm", {})):
            if k not in labels:
                labels.append(k)
    for k in labels:
        row = [k]
        for key in MODE_KEYS:
            row += [
                _fmt(c_all.get(key, {}).get("llm", {}).get(k)),
                _fmt(p_all.get(key, {}).get("llm", {}).get(k)),
            ]
        print("| " + " | ".join(row) + " |")

    for key, title, _, _ in MODES:
        for section, sec_title in [("llm", "Private LLM inference"), ("micro", "MPC primitives & nonlinearities")]:
            c_sec = c_all.get(key, {}).get(section, {})
            p_sec = p_all.get(key, {}).get(section, {})
            if not c_sec and not p_sec:
                continue
            print(f"\n### {sec_title}: {title} (ms)")
            print("| Benchmark | CrypTen (PyTorch, ms) | PriX (JAX, ms) | Speedup (CrypTen / PriX) |")
            print("|---|---:|---:|---:|")
            for k in dict.fromkeys(list(p_sec) + list(c_sec)):
                c_val, p_val = c_sec.get(k), p_sec.get(k)
                speedup = f"**{c_val / p_val:.2f}x**" if c_val and p_val else "—"
                print(f"| {k} | {_fmt(c_val)} | {_fmt(p_val)} | {speedup} |")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark PriX (JAX) vs CrypTen (PyTorch)")
    parser.add_argument(
        "--framework", choices=["all", "prix", "crypten", "curl", "report"], default="all",
        help="'curl' is accepted as an alias for 'crypten'",
    )
    parser.add_argument("--device", default="cpu", help="Device to run on: 'cpu' or 'cuda'")
    parser.add_argument("--modes", default=",".join(MODE_KEYS), help=f"Comma-separated subset of {MODE_KEYS}")
    parser.add_argument("--prix-root", default=os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    parser.add_argument(
        "--crypten-root", "--curl-root", dest="crypten_root", default="/content/curl",
        help="Path to a jimouris/curl checkout (CrypTen-based PyTorch baseline)",
    )
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--llm-warmup", type=int, default=1)
    parser.add_argument("--llm-repeats", type=int, default=3)
    parser.add_argument("--out", default=os.path.join(tempfile.gettempdir(), "prix_vs_crypten_bench.json"))
    args = parser.parse_args()
    if args.framework == "curl":
        args.framework = "crypten"
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    unknown = set(modes) - set(MODE_KEYS)
    if unknown:
        parser.error(f"unknown modes {sorted(unknown)}; choose from {MODE_KEYS}")

    if args.framework == "all":
        cmd_base = [
            sys.executable, os.path.abspath(__file__),
            "--device", args.device, "--modes", ",".join(modes),
            "--prix-root", args.prix_root, "--crypten-root", args.crypten_root,
            "--warmup", str(args.warmup), "--repeats", str(args.repeats),
            "--llm-warmup", str(args.llm_warmup), "--llm-repeats", str(args.llm_repeats),
            "--out", args.out,
        ]
        subprocess.run(cmd_base + ["--framework", "prix"], check=True)
        subprocess.run(cmd_base + ["--framework", "crypten"], check=True)
        subprocess.run(cmd_base + ["--framework", "report"], check=True)
        sys.exit(0)

    tmp_dir = os.path.dirname(os.path.abspath(args.out))
    data = {}
    if os.path.exists(args.out):
        with open(args.out, "r") as f:
            data = json.load(f)
    bench_args = dict(
        warmup=args.warmup, repeats=args.repeats, llm_warmup=args.llm_warmup, llm_repeats=args.llm_repeats
    )

    if args.framework in ("prix", "crypten"):
        if args.framework == "prix":
            res = run_prix(args.prix_root, args.device, tmp_dir, modes, bench_args)
        else:
            res = run_crypten(
                args.crypten_root, args.device, tmp_dir, modes, bench_args, prix_root=args.prix_root
            )
        data.setdefault(args.framework, {}).update(res)
        with open(args.out, "w") as f:
            json.dump(data, f, indent=2)
    elif args.framework == "report":
        print_report(data)
