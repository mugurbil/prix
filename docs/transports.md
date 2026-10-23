# Transports and Multi-Host Deployment

PriX parties exchange messages through a **transport**. Pick one based on what you are doing:

| Transport | Use it for | How |
|---|---|---|
| `"tcp"` (default for `prix.run`) | Everything: single-host runs and benchmarks, and real multi-host deployments | `prix.run(fn, world_size=2)` or `prix.init(..., transport="tcp", peers=[...])` |
| `"shm"` | Single-host simulation through shared memory | `prix.run(..., transport="shm")` |
| `"jax"` (legacy) | Nothing new; kept for compatibility | `prix.init(..., use_distributed=True)` |

> [!NOTE]
> `"shm"` passes openings through shared memory between processes on one machine. That is a
> simulation shortcut, not how MPC is deployed, so `"tcp"` is the default. On our CPU host the
> two are within a few percent of each other end to end (TCP is faster for messages above
> 16 MiB, where shared memory falls back to pickled queues), so there is little reason to
> use `"shm"`.

## The TCP transport

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

### Single host

```python
import prix
out = prix.run(my_fn, world_size=2, mode="shaft", transport="tcp")
```

This is the default, so `transport="tcp"` can be omitted. It runs the same code path as a
multi-host deployment, over loopback. `PRIX_TRANSPORT=shm` switches the default back.

### Multiple hosts

Each process gets a rank and the full list of addresses, in rank order:
**MPC parties first, then the TTP dealer (only if `cfg.mpc.provider == "TTP"`), then
evaluators (Fission mode).**

```python
# On host A (rank 0) and host B (rank 1); same code, different rank.
import prix

PEERS = ["10.0.0.1:29500", "10.0.0.2:29500"]
TLS = {"certfile": "party.pem", "keyfile": "party.key", "cafile": "ca.pem"}

prix.init(rank=RANK, world_size=2, mode="shaft", transport="tcp", peers=PEERS, tls=TLS)
# Each input comes from one party; the others pass None and learn only its shape.
x = prix.array(features if RANK == 0 else None, src=0, broadcast_size=True)
w = prix.array(weights if RANK == 1 else None, src=1, broadcast_size=True)
y = (x @ w).reconstruct()
prix.uninit()
```

A TTP dealer or a Fission evaluator on another host runs:

```python
prix.serve(rank=2, world_size=2, mode="fission", transport="tcp", peers=PEERS_WITH_EVALUATOR, tls=TLS)
```

Environment variables: `PRIX_PEERS` (comma-separated addresses, used when `peers` is not given),
`PRIX_SESSION` (any string shared by all processes of one run; connections from other runs
are rejected), and `PRIX_BIND_HOST` (listen address, e.g. `0.0.0.0`, if a process's public
address is not a local interface).

### Do I need TLS?

- **Benchmarks on one machine or a trusted network:** no. CrypTen's `gloo` transport has no
  TLS either, so comparisons stay fair.
- **Anything protecting real data:** yes. Without TLS an eavesdropper can record the PRNG seeds
  exchanged at startup, the TTP dealer's messages, and output shares, which together reveal
  secrets; and nothing stops a machine in the middle from impersonating a party. PriX logs a
  warning when you use TCP between hosts without TLS.

### Creating certificates

Python 3.13+ checks certificates strictly (RFC 5280), so include the extensions below.

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
openssl req -newkey rsa:2048 -nodes -keyout party.key -out party.csr -subj "/CN=prix-party-0"
openssl x509 -req -in party.csr -CA ca.pem -CAkey ca.key -CAcreateserial -out party.pem \
  -days 365 -extfile leaf.ext
```

Any process holding a certificate from this CA is trusted, so use a CA dedicated to one
deployment. Hostname checking is off by default because peers are often addressed by IP; pass
`"check_hostname": True` in `tls` (and put the hostnames in the certificates) to enable it.

## Security notes

The transport protects messages in flight. It does not change PriX's threat model
(semi-honest parties, non-colluding dealer and evaluators) or its other limitations, such as
the non-cryptographic PRNGs; see the README's security warning.
