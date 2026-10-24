# Transports, Multi-Terminal, and Multi-Host Deployment

PriX parties exchange messages through a **transport**. Pick one based on what you are doing:

| Transport | Use it for | How |
|---|---|---|
| `"tcp"` (default for `prix.run`) | Everything: single-host runs and benchmarks, multi-terminal testing, and real multi-host LAN/WAN deployments | `prix.run(fn, world_size=2)` or `prix.init(..., transport="tcp", peers=[...])` |
| `"shm"` | Single-host simulation through shared memory | `prix.run(..., transport="shm")` |
| `"jax"` (legacy) | Nothing new; kept for compatibility | `prix.init(..., use_distributed=True)` |

> [!NOTE]
> `"shm"` passes openings through shared memory between processes on one machine. That is a
> simulation shortcut, not how MPC is deployed, so `"tcp"` is the default. On both our CPU and GPU
> hosts the two are within 4–5% of each other end to end (and TCP is faster for messages above
> 16 MiB, where shared memory falls back to pickled queues), so there is little reason to
> use `"shm"`.

---

## The Peer-to-Peer TCP Transport

- **Direct connections.** Every pair of processes (MPC parties, the TTP dealer if used, and
  Fission evaluators) shares one TCP connection. There is no coordinator in the data path.
- **No pickle.** Messages use a small binary format that only carries plain data (numbers,
  strings, lists, dicts, and numeric arrays), so a malicious peer cannot make the receiver run
  code.
- **No deadlocks on large messages.** Sockets are non-blocking, and every wait keeps flushing
  queued outgoing data, so two parties can send each other large tensors at the same time.
- **Optional mutual TLS.** Pass `tls={"certfile": ..., "keyfile": ..., "cafile": ...}` to
  encrypt and authenticate every connection. Each side must present a certificate signed by
  the shared CA.

---

## Step-by-Step Examples (`examples/multi_party_tcp.py`)

We include a ready-to-run script, [`examples/multi_party_tcp.py`](../examples/multi_party_tcp.py), that works across **multiple terminals on one computer** or **multiple computers over a LAN/WAN**.

### 1. Single-Process Launcher (Default `prix.run`)

When prototyping on one machine, `prix.run` spawns the processes and assigns free loopback TCP ports automatically:

```python
import prix
out = prix.run(my_fn, world_size=2, mode="shaft")  # transport="tcp" is the default
```

### 2. Two Terminals on One Machine (2-Party MPC)

You do **not** need two computers to test the multi-host TCP stack. Open two terminal tabs on the same machine; each process binds to its own port on `127.0.0.1` (`29500` and `29501` by default):

**Terminal 1 (Party 0, holds the secret input `x`):**
```bash
python3 examples/multi_party_tcp.py --rank 0 --mode shaft --workload mlp
```

**Terminal 2 (Party 1, holds the secret model weights `W`):**
```bash
python3 examples/multi_party_tcp.py --rank 1 --mode shaft --workload mlp
```

To run a full private Pre-LN Transformer block (`MultiheadAttention` + `LayerNorm` + `GELU` FFN) instead of the MLP:
```bash
# Terminal 1
python3 examples/multi_party_tcp.py --rank 0 --mode shaft --workload transformer
# Terminal 2
python3 examples/multi_party_tcp.py --rank 1 --mode shaft --workload transformer
```

### 3. Three Terminals on One Machine (`TTP` Dealer or `fission` Evaluator)

When you use a separate **Trusted Third Party (`TTP`) dealer** (`cfg.mpc.provider = "TTP"`) or a **Fission plaintext evaluator** (`mode="fission"`), the helper process has `rank >= world_size` and calls `prix.serve(...)`:

#### With a non-colluding `TTP` dealer (`--provider TTP`):
```bash
# Terminal 1 (Party 0)
python3 examples/multi_party_tcp.py --rank 0 --mode curl --provider TTP
# Terminal 2 (Party 1)
python3 examples/multi_party_tcp.py --rank 1 --mode curl --provider TTP
# Terminal 3 (TTP dealer, rank 2)
python3 examples/multi_party_tcp.py --rank 2 --mode curl --provider TTP
```

#### With a Fission hybrid evaluator (`--mode fission`):
```bash
# Terminal 1 (Party 0)
python3 examples/multi_party_tcp.py --rank 0 --mode fission
# Terminal 2 (Party 1)
python3 examples/multi_party_tcp.py --rank 1 --mode fission
# Terminal 3 (Fission evaluator, rank 2)
python3 examples/multi_party_tcp.py --rank 2 --mode fission
```

### 4. Two (or More) Separate Machines Over a LAN or WAN

To run across separate computers or cloud VMs (e.g., `10.0.0.1` and `10.0.0.2`), pass `--peers` (or set `PRIX_PEERS`) with each process's reachable `host:port` in rank order (**MPC parties first, then the TTP dealer if `provider == "TTP"`, then evaluators**):

1. **(Optional, recommended on untrusted networks) Generate a CA and per-rank certificates** on one machine and copy `ca.pem` plus each rank's `party<rank>.{pem,key}` to the corresponding machine:
   ```bash
   python3 examples/multi_party_tcp.py --generate-certs ./prix_certs
   ```
2. **On Machine A (`10.0.0.1`, Rank 0):**
   ```bash
   PRIX_BIND_HOST=0.0.0.0 python3 examples/multi_party_tcp.py \
       --rank 0 --peers 10.0.0.1:29500,10.0.0.2:29500 \
       --mode shaft --workload transformer --tls-dir ./prix_certs
   ```
3. **On Machine B (`10.0.0.2`, Rank 1):**
   ```bash
   PRIX_BIND_HOST=0.0.0.0 python3 examples/multi_party_tcp.py \
       --rank 1 --peers 10.0.0.1:29500,10.0.0.2:29500 \
       --mode shaft --workload transformer --tls-dir ./prix_certs
   ```

> [!TIP]
> **Cloud VM / NAT tip:** On cloud instances (AWS EC2, GCP GCE, Azure) where the public or VPC IP is not directly assigned to a local interface name, set `PRIX_BIND_HOST=0.0.0.0` so the listening socket binds to all local interfaces while peers connect to the address in `peers`. Make sure TCP port `29500` is open in your firewall / security group.

---

## Writing Your Own Multi-Host Script

In your own code, computing parties call `prix.init(...)` and helper processes (`TTP` or Fission evaluators) call `prix.serve(...)`:

```python
# Run on Host A (RANK=0) and Host B (RANK=1)
import os
import prix

RANK = int(os.environ["RANK"])
PEERS = ["10.0.0.1:29500", "10.0.0.2:29500"]
TLS = {"certfile": f"party{RANK}.pem", "keyfile": f"party{RANK}.key", "cafile": "ca.pem"}

prix.init(rank=RANK, world_size=2, mode="shaft", transport="tcp", peers=PEERS, tls=TLS)

# Each secret input comes from one party; the others pass None and learn only its shape.
x = prix.array(features if RANK == 0 else None, src=0, broadcast_size=True)
w = prix.array(weights if RANK == 1 else None, src=1, broadcast_size=True)
y = (x @ w).gelu().reconstruct()

prix.uninit()
```

A TTP dealer or Fission evaluator on another host runs:

```python
prix.serve(rank=2, world_size=2, evaluator_size=1, mode="fission",
           transport="tcp", peers=PEERS_WITH_EVALUATOR, tls=TLS)
```

Useful environment variables:
- `PRIX_PEERS`: comma-separated `host:port` list in rank order (used when `peers=` is omitted).
- `PRIX_SESSION`: any string shared by all processes of one run; connections from other runs are rejected.
- `PRIX_BIND_HOST`: local listen address (e.g. `0.0.0.0`) if a machine's advertised address is behind NAT.

---

## Emulating LAN and WAN on a Single Linux Machine (`tc netem`)

Following standard practice in MPC evaluations (CrypTen, Curl, SHAFT, PUMA), you can emulate realistic LAN or WAN round-trip latency and bandwidth limits between two terminals on a single Linux machine by shaping the loopback interface (`lo`) with `tc netem`:

```bash
# Example A: Emulate a fast datacenter LAN (0.2 ms RTT = 0.1 ms one-way delay, 10 Gbps)
sudo tc qdisc add dev lo root netem delay 0.1ms rate 10Gbit

# Example B: Emulate a cross-region WAN (40 ms RTT = 20 ms one-way delay, 200 Mbps)
sudo tc qdisc replace dev lo root netem delay 20ms rate 200Mbit

# Run your benchmark or two-terminal script while shaping is active:
python3 examples/multi_party_tcp.py --rank 0 --mode shaft --workload transformer &
python3 examples/multi_party_tcp.py --rank 1 --mode shaft --workload transformer &
wait

# IMPORTANT: Always remove the loopback rule when finished!
sudo tc qdisc del dev lo root
```

---

## Certificates & Mutual TLS Reference

### Do I need TLS?

- **Benchmarks on one machine or a trusted isolated lab network:** no. CrypTen's `gloo` transport has no TLS either, so comparisons stay fair.
- **Anything protecting real data or running over an untrusted network:** yes. Without TLS an eavesdropper can record the PRNG seeds exchanged at startup, the TTP dealer's messages, and output shares, which together reveal secrets; and nothing stops a machine in the middle from impersonating a party. PriX logs a warning when you use TCP between distinct hosts without TLS.

### Creating certificates manually with `openssl`

You can generate test certificates in one command with `python3 examples/multi_party_tcp.py --generate-certs ./certs`, or directly with `openssl` (including RFC 5280 extensions required by Python 3.13+):

```bash
# Certificate authority (keep ca.key offline)
openssl req -x509 -newkey rsa:2048 -nodes -keyout ca.key -out ca.pem -days 365 \
  -subj "/CN=prix-ca" -addext "basicConstraints=critical,CA:TRUE" \
  -addext "keyUsage=critical,keyCertSign,cRLSign" -addext "subjectKeyIdentifier=hash"

# One certificate per process
cat > leaf.ext <<'EOF'
basicConstraints=critical,CA:FALSE
keyUsage=critical,digitalSignature,keyEncipherment
extendedKeyUsage=serverAuth,clientAuth
subjectKeyIdentifier=hash
authorityKeyIdentifier=keyid,issuer
EOF
openssl req -newkey rsa:2048 -nodes -keyout party0.key -out party0.csr -subj "/CN=prix-party-0"
openssl x509 -req -in party0.csr -CA ca.pem -CAkey ca.key -CAcreateserial -out party0.pem \
  -days 365 -extfile leaf.ext
```

Any process holding a certificate from this CA is trusted, so use a CA dedicated to one deployment. Hostname checking is off by default because peers are often addressed by IP; pass `"check_hostname": True` in `tls` (and put the hostnames/SANs in the certificates) to enable it.

## Security Notes

The transport protects messages in flight. It does not change PriX's threat model (semi-honest parties, non-colluding dealer and evaluators) or its other limitations, such as the non-cryptographic PRNGs; see the README's security warning.
