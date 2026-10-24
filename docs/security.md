# Security, and what PriX is not

> [!WARNING]
> **Research prototype: not suitable for protecting real secrets.** Security is semi-honest. The default `TFP` provider is a *simulated* dealer (both parties expand the same seeds, so no key material is ever transferred and party 0 could learn every secret, as in CrypTen); use `TTP` for a non-colluding dealer. FSS PRG trees are expanded with `SplitMix64` from 62-bit seeds (`cfg.mpc.prng="siphash24"` for a keyed PRF), which is *not* a cryptographic PRG at 128-bit security.

- **Threat model.** Semi-honest, dishonest-majority 2PC (and $N$-party additive sharing for the CrypTen-style modes) in the preprocessing model. `cfg.mpc.active_security` is a Beaver-consistency check, not malicious security.
- **Dealer.** `TFP` simulates the dealer: correlated randomness and FSS keys are expanded locally from shared seeds, so nothing is transferred and the measured latencies are *online-phase* numbers. `prix.provider.get_preproc_stats()` reports the bytes a real dealer would have to send; Section 6.2 of the [paper](../paper/prix_arxiv.pdf) tabulates them. SIGMA's published GPU latencies *include* key transfer; do not compare device-local PriX timings against them.
- **PRG.** FSS trees use `SplitMix64` with 62-bit seeds for speed; this is below 128-bit security and not a standard PRG assumption. A production build needs AES/ChaCha in `prix/fss.py`'s PRG.
- **Input ranges.** LUT modes are exact only on their configured domains; the `default` preset sizes the softmax reciprocal from the sequence length and uses a range-extended `inv_sqrt` (variances $[2^{-8}, 2^{16})$), but every other function's range is documented in [Modes & Configuration](modes_and_configuration.md#5-accuracy-and-valid-input-ranges) and is **not** checked at run time.
- **Transport.** Parties talk over peer-to-peer TCP with a pickle-free wire format; enable mutual TLS for anything beyond experiments ([Transports](transports.md)).
