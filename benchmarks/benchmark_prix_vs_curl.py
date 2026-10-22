#!/usr/bin/env python3
"""
Head-to-head benchmark comparing CuRL (PyTorch/CrypTen) vs PriX (JAX) on CPU or GPU.
"""

import argparse
import json
import multiprocessing as mp
import os
import subprocess
import sys
import tempfile
import time
import numpy as np


class AttrDict(dict):
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key)

    def __setattr__(self, key, value):
        self[key] = value


def _bench_fn(fn, warmup=2, repeats=5, sync_fn=None):
    for _ in range(warmup):
        out = fn()
        if sync_fn is not None:
            sync_fn(out)
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        out = fn()
        if sync_fn is not None:
            sync_fn(out)
        times.append((time.perf_counter() - t0) * 1000.0)  # ms
    return float(np.median(times)), float(np.min(times))


def _save_part(path, data):
    with open(path, "w") as f:
        json.dump(data, f)


def _load_part(path):
    with open(path, "r") as f:
        return json.load(f)


def _patch_curl_repo(curl_root):
    """Apply Python 3.12+/modern JAX/PyTorch compatibility fixes to a cloned jimouris/curl checkout."""
    for rel in ["curl/mpc/primitives/beaver.py", "curl/mpc/primitives/plaintext.py"]:
        p = os.path.join(curl_root, rel)
        if os.path.exists(p):
            txt = open(p).read()
            txt_new = txt.replace("from jax.lib import xla_bridge\n", "").replace(
                "xla_bridge.get_backend().platform\n", ""
            )
            if txt_new != txt:
                open(p, "w").write(txt_new)

    onnx_p = os.path.join(curl_root, "curl/nn/onnx_converter.py")
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

    llama_p = os.path.join(curl_root, "examples/llms/models/llama.py")
    if os.path.exists(llama_p):
        txt = open(llama_p).read()
        if "self.embed_dim, self.head_dim = 2048, 32" in txt:
            open(llama_p, "w").write(
                txt.replace(
                    "self.embed_dim, self.head_dim = 2048, 32",
                    "self.embed_dim, self.head_dim = embed_dim, head_dim",
                )
            )

    for cfg_rel in ["configs/default.yaml", "configs/fission.yaml"]:
        p = os.path.join(curl_root, cfg_rel)
        if os.path.exists(p):
            txt = open(p).read()
            if 'provider: "TTP"' in txt:
                open(p, "w").write(txt.replace('provider: "TTP"', 'provider: "TFP"'))

    init_p = os.path.join(curl_root, "curl/__init__.py")
    if os.path.exists(init_p):
        txt = open(init_p).read()
        target = '            generators[key][device] = torch.Generator(device=torch.device(device))'
        replacement = (
            '            gen = torch.Generator(device=torch.device(device))\n'
            '            generators[key][device] = gen\n'
            '            generators[key][torch.device("cuda")] = gen\n'
            '            generators[key][torch.device("cuda", 0)] = gen'
        )
        if target in txt and 'generators[key][torch.device("cuda", 0)]' not in txt:
            open(init_p, "w").write(txt.replace(target, replacement))


# ============================================================================
# PriX Benchmarks
# ============================================================================
def _prix_micro_worker(args_dict):
    import jax
    import jax.numpy as jnp
    import prix
    import prix.communicator as comm

    out_path = args_dict["out_path"]
    device = args_dict["device"]
    rank = comm.get().get_rank()
    sync = lambda x: x.share.block_until_ready() if hasattr(x, "share") else None

    _ = prix.LookupTables(device=device)

    np.random.seed(42)
    a_np = np.random.uniform(-2.0, 2.0, size=(128, 128)).astype(np.float32)
    b_np = np.random.uniform(-2.0, 2.0, size=(128, 128)).astype(np.float32)
    pos_np = np.random.uniform(0.5, 4.0, size=(32, 64)).astype(np.float32)
    act_np = np.random.uniform(-2.0, 2.0, size=(32, 64)).astype(np.float32)
    attn_np = np.random.uniform(-2.0, 2.0, size=(4, 32, 32)).astype(np.float32)
    ln_np = np.random.uniform(-2.0, 2.0, size=(4, 32, 64)).astype(np.float32)

    a = prix.cryptensor(a_np, device=device)
    b = prix.cryptensor(b_np, device=device)
    pos = prix.cryptensor(pos_np, device=device)
    act = prix.cryptensor(act_np, device=device)
    attn = prix.cryptensor(attn_np, device=device)
    ln = prix.cryptensor(ln_np, device=device)

    ops = [
        ("Private MatMul (128x128)", lambda: a @ b),
        ("Private Mul (128x128)", lambda: a * b),
        ("EGK TruncPr (128x128)", lambda: a.egk_trunc_pr(62, 16)),
        ("Comparison / ReLU (32x64)", lambda: act.relu()),
        ("Exp [Haar LUT] (32x64)", lambda: act.exp()),
        ("Reciprocal [Haar LUT] (32x64)", lambda: pos.reciprocal()),
        ("InvSqrt [Haar LUT] (32x64)", lambda: pos.inv_sqrt()),
        ("GeLU [Haar LUT] (32x64)", lambda: act.gelu()),
        ("SiLU [Haar LUT] (32x64)", lambda: act.silu()),
        ("Softmax [LUT] (4x32x32)", lambda: attn.softmax(dim=-1)),
        ("LayerNorm [LUT] (4x32x64)", lambda: ln.layernorm(1, 0)),
    ]

    out_dict = {}
    for name, fn in ops:
        comm.get().barrier()
        med_ms, _ = _bench_fn(fn, warmup=2, repeats=5, sync_fn=sync)
        if rank == 0:
            out_dict[name] = med_ms
    if rank == 0:
        _save_part(out_path, out_dict)


def _prix_llm_worker(args_dict):
    import prix
    import prix.communicator as comm
    from prix.config import cfg
    from examples.llms.models.bert import BertTiny, BertModel
    from examples.llms.models.gpt2 import GPT
    from examples.llms.models.llama import Llama
    from examples.llms.models.modern_bert import ModernBertModel

    config_path = args_dict["config_path"]
    out_path = args_dict["out_path"]
    device = args_dict["device"]
    if config_path is not None:
        cfg.load_config(config_path)

    rank = comm.get().get_rank()
    sync = lambda x: x.share.block_until_ready() if hasattr(x, "share") else None

    models_spec = [
        ("BertTiny (2-Block, seq=16)", lambda: BertTiny(seq_len=16, full=False), (1, 16, 128), False),
        (
            "BertBase (1-Block, seq=16)",
            lambda: BertModel(vocab_size=30522, emb_size=768, seq_len=16, n_heads=12, n_layers=1, full=False),
            (1, 16, 768),
            False,
        ),
        (
            "GPT2 (1-Block, seq=16)",
            lambda: GPT(embed_dim=768, num_heads=12, num_blocks=1, vocab_size=50257, seq_len=16, full=False),
            (1, 16, 768),
            False,
        ),
        (
            "Llama (1-Block, d=768, seq=16)",
            lambda: Llama(
                embed_dim=768,
                head_dim=12,
                n_layers=1,
                n_kv_heads=4,
                vocab_size=32000,
                multiple_of=256,
                ffn_dim_multiplier=1.0,
                seq_len=16,
                full=False,
                cache=False,
            ),
            (1, 16, 768),
            False,
        ),
        (
            "ModernBert (1-Block, seq=16)",
            lambda: ModernBertModel(
                vocab_size=50368,
                hidden_size=768,
                intermediate_size=1152,
                seq_len=16,
                n_heads=12,
                n_layers=1,
                full=False,
            ),
            (1, 16, 768),
            False,
        ),
    ]

    out_dict = {}
    for label, builder, shape, is_full in models_spec:
        m = builder().to(device).encrypt(src=0)
        m.eval()
        np.random.seed(123)
        if is_full:
            x_np = np.random.randint(0, 100, size=shape).astype(np.float32)
        else:
            x_np = np.random.uniform(-1.0, 1.0, size=shape).astype(np.float32)
        x = prix.cryptensor(x_np, device=device)
        comm.get().barrier()
        med_ms, _ = _bench_fn(lambda: m(x), warmup=1, repeats=2, sync_fn=sync)
        if rank == 0:
            print(f"[PriX ({device})] {label}: {med_ms:.2f} ms", flush=True)
            out_dict[label] = med_ms
    if rank == 0:
        _save_part(out_path, out_dict)


def run_prix_benchmarks(prix_root, device, tmp_dir):
    sys.path.insert(0, prix_root)
    from prix.launcher import MultiProcessLauncher

    default_cfg = os.path.join(prix_root, "configs/default.yaml")
    fission_cfg = os.path.join(prix_root, "configs/fission.yaml")

    p_micro = os.path.join(tmp_dir, "_prix_micro.json")
    launcher = MultiProcessLauncher(
        2, 0, _prix_micro_worker, AttrDict(out_path=p_micro, device=device), cfg_file=default_cfg
    )
    launcher.start()
    launcher.join()
    launcher.terminate()
    micro_res = _load_part(p_micro)

    p_def = os.path.join(tmp_dir, "_prix_def.json")
    launcher = MultiProcessLauncher(
        2,
        0,
        _prix_llm_worker,
        AttrDict(config_path=default_cfg, out_path=p_def, device=device),
        cfg_file=default_cfg,
    )
    launcher.start()
    launcher.join()
    launcher.terminate()
    llm_default_res = _load_part(p_def)

    p_fis = os.path.join(tmp_dir, "_prix_fis.json")
    launcher = MultiProcessLauncher(
        2,
        1,
        _prix_llm_worker,
        AttrDict(config_path=fission_cfg, out_path=p_fis, device=device),
        cfg_file=fission_cfg,
    )
    launcher.start()
    launcher.join()
    launcher.terminate()
    llm_fission_res = _load_part(p_fis)

    return micro_res, llm_default_res, llm_fission_res


# ============================================================================
# CuRL Benchmarks
# ============================================================================
def _curl_micro_worker(args_dict):
    import torch
    import curl
    import curl.communicator as comm

    out_path = args_dict["out_path"]
    device = args_dict["device"]
    rank = comm.get().get_rank()
    sync = (lambda _: torch.cuda.synchronize()) if str(device).startswith("cuda") else None

    _ = curl.common.functions.approximations.LookupTables(device=device)

    np.random.seed(42)
    a_t = torch.from_numpy(np.random.uniform(-2.0, 2.0, size=(128, 128)).astype(np.float32)).to(device)
    b_t = torch.from_numpy(np.random.uniform(-2.0, 2.0, size=(128, 128)).astype(np.float32)).to(device)
    pos_t = torch.from_numpy(np.random.uniform(0.5, 4.0, size=(32, 64)).astype(np.float32)).to(device)
    act_t = torch.from_numpy(np.random.uniform(-2.0, 2.0, size=(32, 64)).astype(np.float32)).to(device)
    attn_t = torch.from_numpy(np.random.uniform(-2.0, 2.0, size=(4, 32, 32)).astype(np.float32)).to(device)
    ln_t = torch.from_numpy(np.random.uniform(-2.0, 2.0, size=(4, 32, 64)).astype(np.float32)).to(device)

    a = curl.cryptensor(a_t, device=device)
    b = curl.cryptensor(b_t, device=device)
    pos = curl.cryptensor(pos_t, device=device)
    act = curl.cryptensor(act_t, device=device)
    attn = curl.cryptensor(attn_t, device=device)
    ln = curl.cryptensor(ln_t, device=device)

    ops = [
        ("Private MatMul (128x128)", lambda: a @ b),
        ("Private Mul (128x128)", lambda: a * b),
        ("EGK TruncPr (128x128)", lambda: a.egk_trunc_pr(62, 16)),
        ("Comparison / ReLU (32x64)", lambda: act.relu()),
        ("Exp [Haar LUT] (32x64)", lambda: act.exp()),
        ("Reciprocal [Haar LUT] (32x64)", lambda: pos.reciprocal()),
        ("InvSqrt [Haar LUT] (32x64)", lambda: pos.inv_sqrt()),
        ("GeLU [Haar LUT] (32x64)", lambda: act.gelu()),
        ("SiLU [Haar LUT] (32x64)", lambda: act.silu()),
        ("Softmax [LUT] (4x32x32)", lambda: attn.softmax(dim=-1)),
        ("LayerNorm [LUT] (4x32x64)", lambda: ln.layernorm(1, 0)),
    ]

    out_dict = {}
    with curl.no_grad():
        for name, fn in ops:
            comm.get().barrier()
            med_ms, _ = _bench_fn(fn, warmup=2, repeats=5, sync_fn=sync)
            if rank == 0:
                out_dict[name] = med_ms
    if rank == 0:
        _save_part(out_path, out_dict)


def _curl_llm_worker(args_dict):
    import torch
    import curl
    import curl.communicator as comm
    from curl.config import cfg
    from examples.llms.models.bert import BertTiny, BertModel
    from examples.llms.models.gpt2 import GPT
    from examples.llms.models.llama import Llama
    from examples.llms.models.modern_bert import ModernBertModel

    config_path = args_dict["config_path"]
    out_path = args_dict["out_path"]
    device = args_dict["device"]
    if config_path is not None:
        cfg.load_config(config_path)

    rank = comm.get().get_rank()
    sync = (lambda _: torch.cuda.synchronize()) if str(device).startswith("cuda") else None

    models_spec = [
        ("BertTiny (2-Block, seq=16)", lambda: BertTiny(seq_len=16, full=False), (1, 16, 128), False),
        (
            "BertBase (1-Block, seq=16)",
            lambda: BertModel(vocab_size=30522, emb_size=768, seq_len=16, n_heads=12, n_layers=1, full=False),
            (1, 16, 768),
            False,
        ),
        (
            "GPT2 (1-Block, seq=16)",
            lambda: GPT(embed_dim=768, num_heads=12, num_blocks=1, vocab_size=50257, seq_len=16, full=False),
            (1, 16, 768),
            False,
        ),
        (
            "Llama (1-Block, d=768, seq=16)",
            lambda: Llama(
                embed_dim=768,
                head_dim=12,
                n_layers=1,
                n_kv_heads=4,
                vocab_size=32000,
                multiple_of=256,
                ffn_dim_multiplier=1.0,
                seq_len=16,
                full=False,
                cache=False,
            ),
            (1, 16, 768),
            False,
        ),
        (
            "ModernBert (1-Block, seq=16)",
            lambda: ModernBertModel(
                vocab_size=50368,
                hidden_size=768,
                intermediate_size=1152,
                seq_len=16,
                n_heads=12,
                n_layers=1,
                full=False,
            ),
            (1, 16, 768),
            False,
        ),
    ]

    out_dict = {}
    with curl.no_grad():
        for label, builder, shape, is_full in models_spec:
            m = builder().to(device).encrypt(src=0)
            m.eval()
            np.random.seed(123)
            if is_full:
                x_t = torch.from_numpy(np.random.randint(0, 100, size=shape).astype(np.float32)).to(device)
            else:
                x_t = torch.from_numpy(np.random.uniform(-1.0, 1.0, size=shape).astype(np.float32)).to(device)
            x = curl.cryptensor(x_t, device=device)
            comm.get().barrier()
            med_ms, _ = _bench_fn(lambda: m(x), warmup=1, repeats=2, sync_fn=sync)
            if rank == 0:
                print(f"[CuRL ({device})] {label}: {med_ms:.2f} ms", flush=True)
                out_dict[label] = med_ms
    if rank == 0:
        _save_part(out_path, out_dict)


def run_curl_benchmarks(curl_root, device, tmp_dir):
    _patch_curl_repo(curl_root)
    sys.path.insert(0, curl_root)
    orig_set_start = mp.set_start_method
    mp.set_start_method = lambda method, force=True: orig_set_start(method, force=True)
    from examples.multiprocess_launcher import MultiProcessLauncher

    default_cfg = os.path.join(curl_root, "configs/default.yaml")
    fission_cfg = os.path.join(curl_root, "configs/fission.yaml")

    c_micro = os.path.join(tmp_dir, "_curl_micro.json")
    launcher = MultiProcessLauncher(
        2,
        0,
        _curl_micro_worker,
        AttrDict(out_path=c_micro, device=device, multi_gpu=False),
        cfg_file=default_cfg,
    )
    launcher.start()
    launcher.join()
    launcher.terminate()
    micro_res = _load_part(c_micro)

    c_def = os.path.join(tmp_dir, "_curl_def.json")
    launcher = MultiProcessLauncher(
        2,
        0,
        _curl_llm_worker,
        AttrDict(config_path=default_cfg, out_path=c_def, device=device, multi_gpu=False),
        cfg_file=default_cfg,
    )
    launcher.start()
    launcher.join()
    launcher.terminate()
    llm_default_res = _load_part(c_def)

    c_fis = os.path.join(tmp_dir, "_curl_fis.json")
    launcher = MultiProcessLauncher(
        2,
        1,
        _curl_llm_worker,
        AttrDict(config_path=fission_cfg, out_path=c_fis, device=device, multi_gpu=False),
        cfg_file=fission_cfg,
    )
    launcher.start()
    launcher.join()
    launcher.terminate()
    llm_fission_res = _load_part(c_fis)

    return micro_res, llm_default_res, llm_fission_res


def print_report(data):
    for section, title in [
        ("micro", "MPC Primitives & Wavelet LUT Microbenchmarks (ms)"),
        ("llm_default", "Private LLM Inference - Wavelet LUT Mode (ms)"),
        ("llm_fission", "Private LLM Inference - Fission Evaluator Mode (ms)"),
    ]:
        print(f"\n### {title}")
        print("| Benchmark | CuRL (PyTorch, ms) | PriX (JAX, ms) | Speedup (CuRL / PriX) |")
        print("|---|---:|---:|---:|")
        c_sec = data.get("curl", {}).get(section, {})
        p_sec = data.get("prix", {}).get(section, {})
        for k in p_sec:
            c_val = c_sec.get(k, float("nan"))
            p_val = p_sec.get(k, float("nan"))
            speedup = c_val / p_val if p_val > 0 else float("nan")
            print(f"| {k} | {c_val:.2f} | {p_val:.2f} | **{speedup:.2f}x** |")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark PriX (JAX) vs CuRL (PyTorch)")
    parser.add_argument("--framework", choices=["all", "prix", "curl", "report"], default="all")
    parser.add_argument("--device", default="cpu", help="Device to run on: 'cpu' or 'cuda'")
    parser.add_argument(
        "--prix-root",
        default=os.path.abspath(os.path.join(os.path.dirname(__file__), "..")),
    )
    parser.add_argument("--curl-root", default="/content/curl")
    parser.add_argument("--out", default=os.path.join(tempfile.gettempdir(), "prix_vs_curl_bench.json"))
    args = parser.parse_args()

    if args.framework == "all":
        cmd_base = [
            sys.executable,
            os.path.abspath(__file__),
            "--device",
            args.device,
            "--prix-root",
            args.prix_root,
            "--curl-root",
            args.curl_root,
            "--out",
            args.out,
        ]
        subprocess.run(cmd_base + ["--framework", "prix"], check=True)
        subprocess.run(cmd_base + ["--framework", "curl"], check=True)
        subprocess.run(cmd_base + ["--framework", "report"], check=True)
        sys.exit(0)

    tmp_dir = os.path.dirname(os.path.abspath(args.out))
    data = {}
    if os.path.exists(args.out):
        with open(args.out, "r") as f:
            data = json.load(f)

    if args.framework == "prix":
        micro, llm_def, llm_fis = run_prix_benchmarks(args.prix_root, args.device, tmp_dir)
        data["prix"] = {"micro": micro, "llm_default": llm_def, "llm_fission": llm_fis}
        with open(args.out, "w") as f:
            json.dump(data, f, indent=2)
    elif args.framework == "curl":
        micro, llm_def, llm_fis = run_curl_benchmarks(args.curl_root, args.device, tmp_dir)
        data["curl"] = {"micro": micro, "llm_default": llm_def, "llm_fission": llm_fis}
        with open(args.out, "w") as f:
            json.dump(data, f, indent=2)
    elif args.framework == "report":
        print_report(data)
