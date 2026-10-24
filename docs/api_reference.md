# PriX API Reference

## 1. Top-Level Functions (`import prix`)

### Execution & Initialization
- **`@prix.mpc.run_multiprocess(world_size=2, mode=None, evaluator_size=0, device=None)`** *(also `@prix.run_multiprocess`)*
  Decorator that spawns `world_size` MPC processes, initializes PriX with the specified `mode` (`"curl"`, `"poly"`, `"shaft"`, `"crypten"`, `"fission"`), executes the decorated function, and returns Party 0's return value to the caller.
- **`prix.run(fn, *args, world_size=2, mode=None, evaluator_size=0, device=None, **kwargs)`**
  Functional equivalent of `@prix.mpc.run_multiprocess`: runs `fn(*args, **kwargs)` across `world_size` MPC parties and returns Party 0's result. Automatically sets `evaluator_size=1` when `mode="fission"`.
- **Process start method (`start_method=`)**: accepted by `prix.run`, `@prix.mpc.run_multiprocess` and `MultiProcessLauncher`; can also be set with the `PRIX_START_METHOD` environment variable.
  - `"auto"` (the default) **forks** the MPC parties. Forking is fast and works with functions defined in notebooks.
  - If JAX has already run computations in the calling process, forking can deadlock the children. In that case `"auto"` switches to **`spawn`**: the function and its arguments are serialized with `cloudpickle`, and the parent's `prix.cfg` settings (including `temp_override`) are copied to the children. This adds about 2 s of startup per run.
  - `"fork"`, `"spawn"` and `"forkserver"` force a specific method.
  - Plain scripts that use `spawn` must put their top-level code under `if __name__ == "__main__":`.
- **Transport (`transport=`, `tls=`)**: accepted by `prix.run`, `@prix.mpc.run_multiprocess` and `MultiProcessLauncher`; the default can be set with `PRIX_TRANSPORT`.
  - `"tcp"` (default): direct peer-to-peer TCP over loopback, the same code path as a multi-host deployment. `tls={"certfile", "keyfile", "cafile"}` enables mutual TLS. See [Transports](transports.md).
  - `"shm"`: shared memory between local processes, a single-host simulation shortcut.
- **`prix.set_mode(mode, device=None)`**
  Switches PriX's active configuration preset (`"default"` (Wave + SIGMA TR), `"wave"`, `"curl"`, `"sigma"`, `"poly"`, `"shaft"`, `"crypten"`, `"fission"`) and refreshes wavelet lookup tables if already initialized.
- **`prix.init(rank=0, world_size=1, evaluator_size=0, ..., device=None, mode=None, cfg_file=None, transport=None, peers=None, tls=None, session=None)`**
  Initializes the communicator, PRNG streams, wavelet lookup tables, and optional TTP/Evaluator clients for a single worker process. For multi-host runs pass `transport="tcp"` and `peers` (one `"host:port"` per process in rank order: parties, then the TTP if used, then evaluators).
- **`prix.serve(rank, world_size, evaluator_size=0, mode=None, cfg_file=None, **transport_kwargs)`**
  Runs a TTP dealer or Fission evaluator process in a multi-host deployment and returns when the parties shut down.
- **`prix.uninit()`**
  Tears down active communicator and client connections.

### Tensor Creation & Manipulation
- **`prix.array(array, src=0, precision=None, device=None, ptype=prix.mpc.arithmetic)`** *(CrypTen alias: `prix.cryptensor`)*
  Secret-shares `array` from party `src` across all parties.
- **`prix.is_encrypted_array(obj)`** *(CrypTen alias: `prix.is_encrypted_tensor`)*
  Returns `True` if `obj` is a secret-shared `prix.Array`.
- **`prix.zeros(*sizes, device=None, ptype=prix.mpc.arithmetic)`**
  Secret-shared tensor of zeros.
- **`prix.ones(*sizes, device=None, ptype=prix.mpc.arithmetic)`**
  Secret-shared tensor of ones.
- **`prix.full(size, fill_value, device=None, ptype=prix.mpc.arithmetic)`**
  Secret-shared tensor filled with `fill_value`.
- **`prix.arange(*args, device=None, ptype=prix.mpc.arithmetic)`**
  Secret-shared 1-D range tensor.
- **`prix.eye(n, m=None, device=None, ptype=prix.mpc.arithmetic)`**
  Secret-shared 2-D identity matrix.
- **`prix.rand(*sizes, device=None, ptype=prix.mpc.arithmetic)`**
  Secret-shared uniform random tensor in $[0, 1)$.
- **`prix.randn(*sizes, device=None)`**
  Secret-shared standard normal $\mathcal{N}(0, 1)$ random tensor (Box–Muller).
- **`prix.bernoulli(tensor)`**
  Secret-shared Bernoulli draws with success probabilities `tensor`.
- **`prix.cat(tensors, dim=0)`** / **`prix.stack(tensors, dim=0)`**
  Concatenates or stacks a list of `prix.Array` tensors along `dim`.
- **`prix.where(condition, input_tensor, other)`**
  Private multiplexer selecting `input_tensor` where `condition == 1` and `other` where `condition == 0`.

---

## 2. Secret-Shared Tensor (`prix.Array`)

### Decryption & Conversion
- **`tensor.reconstruct(dst=None)`** *(CrypTen alias: `tensor.get_plain_text(dst=None)`)*: Reconstructs and decodes the fixed-point secret-shared tensor to a float32 JAX array.
- **`tensor.reveal(dst=None)`**: Reveals the raw 64-bit integer ring element without fixed-point downscaling.
- **`tensor.to(ptype, ...)`**: Converts between `prix.mpc.arithmetic` ($\mathbb{Z}_{2^{64}}$) and `prix.mpc.binary` ($\mathbb{Z}_2^{64}$) secret shares (`A2B` / `B2A`).

### Arithmetic & Matrix Operations
- **`+`, `-`, `*`, `@` (`matmul`), `/`, `**` (`pow`), `square()`**: Private and mixed public-private arithmetic and matrix multiplication.
- **`sum(dim=None, keepdim=False)`**, **`mean(dim=None, keepdim=False)`**, **`var(dim=None, unbiased=True, keepdim=False)`**: Reductions along any axis.
- **`cumsum(dim=0)`**, **`trace()`**, **`dot(y)`**, **`ger(y)`**: Linear algebra utilities.

### Nonlinear Activations & Math Functions
- **Elementwise Math**: `exp()`, `log()`, `reciprocal()`, `sqrt()`, `inv_sqrt()`, `abs()`, `sign()`, `cos()`, `sin()`, `cossin()`, `erf()`, `polynomial(coeffs)`
- **Activations**: `relu()`, `gelu()`, `silu()`, `sigmoid()`, `tanh()`, `hardtanh(min_value=-1, max_value=1)`, `softmax(dim=-1)`, `log_softmax(dim=-1)`
- **Normalizations**: `layernorm(weight=None, bias=None, eps=1e-5)` (for RMSNorm use `nn.RMSNorm`)
- **Comparisons & Extremes**: `>`, `>=`, `<`, `<=`, `==`, `!=`, `max(dim=None, keepdim=False)`, `min(dim=None, keepdim=False)`, `argmax(dim=None, keepdim=False)`, `argmin(dim=None, keepdim=False)`

### Shape & Indexing Operations
- `reshape(*shape)`, `view(*shape)`, `transpose(dim0, dim1)`, `permute(*dims)`, `T`, `squeeze(dim=None)`, `unsqueeze(dim)`, `flatten(start_dim=0, end_dim=-1)`, `expand(*sizes)`, `repeat(*sizes)`, `roll(shifts, dims=None)`, `flip(dims)`, `pad(pad, mode="constant", value=0)`, `split(split_size, dim=0)`, `chunk(chunks, dim=0)`, `gather(dim, index)`, `take(index, dimension=None)`, `narrow(dim, start, length)`

---

## 3. Neural Network Modules (`import prix.nn as nn`)

All modules inherit from `prix.nn.Module` and support `.encrypt(src=0)`, `.decrypt()`, `.state_dict()`, `.load_state_dict(state_dict)`, `.parameters()`, and `__call__(*args, **kwargs)`.

- **Containers**: `nn.Module`, `nn.Sequential`, `nn.ModuleList`, `nn.ModuleDict`
- **Linear & Embeddings**: `nn.Linear(in_features, out_features, bias=True)`, `nn.Embedding(num_embeddings, embedding_dim)`
- **Convolutions & Pooling**: `nn.Conv1d`, `nn.Conv2d`, `nn.AvgPool2d`, `nn.MaxPool2d`, `nn.AdaptiveAvgPool2d`, `nn.AdaptiveMaxPool2d`
- **Normalizations**: `nn.LayerNorm(normalized_shape, eps=1e-5, bias=True)`, `nn.RMSNorm(dim, eps=1e-5)`
- **Attention**: `nn.MultiheadAttention(embed_dim, num_heads, bias=True)`: `forward(query, key=None, value=None, attn_mask=None)` takes `(batch, seq, embed_dim)` inputs and returns only the attention output; `attn_mask` is a public additive mask. Inference only (no attention dropout).
- **Activations**: `nn.ReLU`, `nn.ReLU6`, `nn.GELU`, `nn.SiLU`, `nn.Sigmoid`, `nn.Tanh`, `nn.Hardtanh`, `nn.Softmax(dim=-1)`, `nn.LogSoftmax(dim=-1)`
- **Other**: `nn.Identity`, `nn.Dropout`, `nn.MatMul`, `nn.Parameter`

Layers also accept plaintext NumPy/JAX arrays (computed exactly), which gives a plaintext reference for checking MPC accuracy with the same weights. Losses, `Flatten` and `BatchNorm` are not provided.
