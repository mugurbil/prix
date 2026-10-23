#!/usr/bin/env python3
"""Run PriX across multiple terminals or multiple machines over peer-to-peer TCP.

Quick examples (run each line in a separate terminal):

  1. Two terminals on one machine (2-party MPC, SHAFT mode):
       python3 examples/multi_party_tcp.py --rank 0
       python3 examples/multi_party_tcp.py --rank 1

  2. Three terminals on one machine with a non-colluding TTP dealer:
       python3 examples/multi_party_tcp.py --rank 0 --provider TTP
       python3 examples/multi_party_tcp.py --rank 1 --provider TTP
       python3 examples/multi_party_tcp.py --rank 2 --provider TTP   # TTP dealer

  3. Three terminals on one machine in Fission hybrid-evaluator mode:
       python3 examples/multi_party_tcp.py --rank 0 --mode fission
       python3 examples/multi_party_tcp.py --rank 1 --mode fission
       python3 examples/multi_party_tcp.py --rank 2 --mode fission   # Evaluator

  4. Two separate machines over a LAN or WAN (with mutual TLS):
       # Generate a test CA and per-party certs once, then copy to each host:
       python3 examples/multi_party_tcp.py --generate-certs /tmp/prix_certs

       # On Host A (10.0.0.1):
       PRIX_BIND_HOST=0.0.0.0 python3 examples/multi_party_tcp.py \\
           --rank 0 --peers 10.0.0.1:29500,10.0.0.2:29500 --tls-dir /tmp/prix_certs

       # On Host B (10.0.0.2):
       PRIX_BIND_HOST=0.0.0.0 python3 examples/multi_party_tcp.py \\
           --rank 1 --peers 10.0.0.1:29500,10.0.0.2:29500 --tls-dir /tmp/prix_certs
"""

import argparse
import os
import subprocess
import tempfile
import time
import numpy as np

import prix
import prix.communicator as comm
import prix.nn as nn
from prix.config import cfg


def generate_certs(out_dir: str, num_ranks: int = 4) -> None:
    """Generate a self-signed CA and per-rank leaf certificates (RFC 5280 compliant)."""
    os.makedirs(out_dir, exist_ok=True)
    ca_key = os.path.join(out_dir, "ca.key")
    ca_pem = os.path.join(out_dir, "ca.pem")
    subprocess.run(
        [
            "openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes",
            "-keyout", ca_key, "-out", ca_pem, "-days", "365",
            "-subj", "/CN=prix-ca",
            "-addext", "basicConstraints=critical,CA:TRUE",
            "-addext", "keyUsage=critical,keyCertSign,cRLSign",
            "-addext", "subjectKeyIdentifier=hash",
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    ext_content = (
        "basicConstraints=critical,CA:FALSE\n"
        "keyUsage=critical,digitalSignature,keyEncipherment\n"
        "extendedKeyUsage=serverAuth,clientAuth\n"
        "subjectKeyIdentifier=hash\n"
        "authorityKeyIdentifier=keyid,issuer\n"
    )
    with tempfile.NamedTemporaryFile("w", suffix=".ext", delete=False) as ext_f:
        ext_f.write(ext_content)
        ext_path = ext_f.name
    try:
        for r in range(num_ranks):
            key_path = os.path.join(out_dir, f"party{r}.key")
            csr_path = os.path.join(out_dir, f"party{r}.csr")
            pem_path = os.path.join(out_dir, f"party{r}.pem")
            subprocess.run(
                [
                    "openssl", "req", "-newkey", "rsa:2048", "-nodes",
                    "-keyout", key_path, "-out", csr_path,
                    "-subj", f"/CN=prix-rank-{r}",
                ],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            subprocess.run(
                [
                    "openssl", "x509", "-req", "-in", csr_path,
                    "-CA", ca_pem, "-CAkey", ca_key, "-CAcreateserial",
                    "-out", pem_path, "-days", "365", "-extfile", ext_path,
                ],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            os.remove(csr_path)
    finally:
        if os.path.exists(ext_path):
            os.remove(ext_path)
    print(f"Generated CA ({ca_pem}) and party0..party{num_ranks - 1} certificates in {out_dir}")


class PrivateTransformerBlock(nn.Module):
    """Single pre-norm Transformer block held by Party 1 (model owner)."""

    def __init__(self, d_model: int = 64, n_heads: int = 4, d_ff: int = 256):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x):
        h = self.ln1(x)
        x = x + self.attn(h, h, h)
        return x + self.ffn(self.ln2(x))


def run_mlp_workload(rank: int) -> None:
    """Party 0 supplies secret inputs x; Party 1 supplies secret weights (W1, W2)."""
    rng = np.random.default_rng(42)
    x_ref = rng.standard_normal((4, 16), dtype=np.float32) * 0.5
    w1_ref = rng.standard_normal((16, 32), dtype=np.float32) * 0.25
    w2_ref = rng.standard_normal((32, 8), dtype=np.float32) * 0.25

    # Only Party 0 passes x; only Party 1 passes W1, W2. Other parties pass None.
    x = prix.array(x_ref if rank == 0 else None, src=0, broadcast_size=True)
    w1 = prix.array(w1_ref if rank == 1 else None, src=1, broadcast_size=True)
    w2 = prix.array(w2_ref if rank == 1 else None, src=1, broadcast_size=True)

    # Warmup (compiles XLA kernels and initializes LUTs on first run)
    _ = ((x @ w1).gelu() @ w2).softmax(dim=-1).reconstruct()

    comm.get().reset_communication_stats()
    comm.get().barrier()
    t0 = time.perf_counter()
    probs = ((x @ w1).gelu() @ w2).softmax(dim=-1).reconstruct()
    elapsed_ms = (time.perf_counter() - t0) * 1e3

    # Compare against plaintext reference
    h_ref = x_ref @ w1_ref
    gelu_ref = 0.5 * h_ref * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (h_ref + 0.044715 * h_ref**3)))
    logits_ref = gelu_ref @ w2_ref
    exp_ref = np.exp(logits_ref - logits_ref.max(axis=-1, keepdims=True))
    probs_ref = exp_ref / exp_ref.sum(axis=-1, keepdims=True)
    max_err = float(np.max(np.abs(np.asarray(probs) - probs_ref)))

    c = comm.get()
    print(
        f"[Rank {rank}] MLP (Linear -> GeLU -> Linear -> Softmax) | "
        f"time={elapsed_ms:.2f} ms | rounds={c.comm_rounds} | "
        f"bytes={c.comm_bytes / 1024:.1f} KiB | max_abs_err={max_err:.4e}"
    )


def run_transformer_workload(rank: int) -> None:
    """Party 0 supplies token embeddings; Party 1 supplies Transformer weights."""
    np.random.seed(0)
    block = PrivateTransformerBlock(d_model=64, n_heads=4, d_ff=256).encrypt(src=1)
    tokens_plain = np.random.randn(1, 16, 64).astype(np.float32) * 0.5 if rank == 0 else None
    x = prix.array(tokens_plain, src=0, broadcast_size=True)

    # Warmup
    _ = block(x).reconstruct()

    comm.get().reset_communication_stats()
    comm.get().barrier()
    t0 = time.perf_counter()
    out = block(x).reconstruct()
    elapsed_ms = (time.perf_counter() - t0) * 1e3

    c = comm.get()
    print(
        f"[Rank {rank}] TransformerBlock (d=64, H=4, L=16) | "
        f"output_shape={tuple(out.shape)} | time={elapsed_ms:.2f} ms | "
        f"rounds={c.comm_rounds} | bytes={c.comm_bytes / (1024 * 1024):.2f} MiB"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-terminal / multi-host PriX TCP example.")
    parser.add_argument("--rank", type=int, default=int(os.environ.get("RANK", -1)),
                        help="Rank of this process (0..world_size-1 for parties; >=world_size for TTP/evaluator).")
    parser.add_argument("--world-size", type=int, default=2, help="Number of MPC computing parties.")
    parser.add_argument("--mode", default="shaft", choices=["lut", "poly", "shaft", "crypten", "fission"],
                        help="MPC approximation mode.")
    parser.add_argument("--provider", default="TFP", choices=["TFP", "TTP"],
                        help="Correlated randomness provider (TFP or TTP).")
    parser.add_argument("--workload", default="mlp", choices=["mlp", "transformer"],
                        help="Workload to evaluate.")
    parser.add_argument("--peers", default=os.environ.get("PRIX_PEERS", ""),
                        help="Comma-separated host:port list in rank order.")
    parser.add_argument("--base-port", type=int, default=29500,
                        help="Base port on 127.0.0.1 when --peers is omitted.")
    parser.add_argument("--tls-dir", default="",
                        help="Directory with ca.pem, party<rank>.pem, and party<rank>.key.")
    parser.add_argument("--generate-certs", default="", metavar="DIR",
                        help="Generate a test CA and party certificates in DIR and exit.")
    args = parser.parse_args()

    if args.generate_certs:
        generate_certs(args.generate_certs)
        return

    if args.rank < 0:
        parser.error("Please specify --rank (or set RANK=...).")

    # Load mode config first, then apply --provider override and enable comm stats
    cfg.load_config(args.mode)
    cfg.mpc.provider = args.provider
    cfg.communicator.verbose = True
    evaluator_size = 1 if args.mode == "fission" else 0
    total_ranks = args.world_size + (1 if args.provider == "TTP" else 0) + evaluator_size

    if args.peers:
        peers = [p.strip() for p in args.peers.split(",") if p.strip()]
    else:
        peers = [f"127.0.0.1:{args.base_port + r}" for r in range(total_ranks)]

    if len(peers) != total_ranks:
        raise ValueError(
            f"Mode '{args.mode}' with provider='{args.provider}' requires {total_ranks} processes "
            f"(world_size={args.world_size}, TTP={int(args.provider == 'TTP')}, evaluators={evaluator_size}), "
            f"but got {len(peers)} peers: {peers}"
        )

    tls = None
    if args.tls_dir:
        tls = {
            "certfile": os.path.join(args.tls_dir, f"party{args.rank}.pem"),
            "keyfile": os.path.join(args.tls_dir, f"party{args.rank}.key"),
            "cafile": os.path.join(args.tls_dir, "ca.pem"),
        }

    print(f"[Rank {args.rank}/{total_ranks - 1}] Connecting to peers={peers} (mode={args.mode}, provider={args.provider}, tls={'on' if tls else 'off'})...")

    # Helper processes (TTP dealer or Fission evaluator) call prix.serve(...)
    if args.rank >= args.world_size:
        prix.serve(
            rank=args.rank,
            world_size=args.world_size,
            evaluator_size=evaluator_size,
            transport="tcp",
            peers=peers,
            tls=tls,
        )
        print(f"[Rank {args.rank}] Helper server exited cleanly.")
        return

    # Computing parties call prix.init(...)
    prix.init(
        rank=args.rank,
        world_size=args.world_size,
        evaluator_size=evaluator_size,
        transport="tcp",
        peers=peers,
        tls=tls,
    )
    try:
        if args.workload == "mlp":
            run_mlp_workload(args.rank)
        else:
            run_transformer_workload(args.rank)
    finally:
        prix.uninit()


if __name__ == "__main__":
    main()
