#!/usr/bin/env bash
# Re-runs every CPU benchmark behind the paper / README / docs tables, sequentially, with provenance.
# GPU numbers come from the Colab notebooks (examples/prix_vs_curl_gpu_colab.ipynb, examples/prix_fss_wave_sigma_gpu_colab.ipynb).
#
# Usage:  CRYPTEN_ROOT=/path/to/jimouris/curl bash benchmarks/run_all.sh [results_tag]
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-python}
TAG=${1:-cpu_$(hostname -s)}
CRYPTEN_ROOT=${CRYPTEN_ROOT:?set CRYPTEN_ROOT to a jimouris/curl checkout}
OUT=benchmarks/results/${TAG}.json
export JAX_PLATFORMS=${JAX_PLATFORMS:-cpu}
log() { echo "[$(date +%H:%M:%S) load=$(cut -d' ' -f1 /proc/loadavg)] $*"; }

log "R1  PriX vs CrypTen, 5 modes, both frameworks, 5 repeats, comm stats -> $OUT"
$PY benchmarks/benchmark_prix_vs_crypten.py --device cpu --crypten-root "$CRYPTEN_ROOT" \
    --repeats 5 --llm-repeats 5 --comm-stats --out "$OUT" --overwrite
log "R3  transport ablation: PriX over gloo and shared memory -> $OUT"
$PY benchmarks/benchmark_prix_vs_crypten.py --framework prix --device cpu --crypten-root "$CRYPTEN_ROOT" \
    --repeats 5 --llm-repeats 5 --comm-stats --prix-transport gloo --out "$OUT"
$PY benchmarks/benchmark_prix_vs_crypten.py --framework prix --device cpu --crypten-root "$CRYPTEN_ROOT" \
    --repeats 5 --llm-repeats 5 --comm-stats --prix-transport shm --out "$OUT"
$PY benchmarks/transport_ablation.py "$OUT" --fig docs/figures/transport_ablation
log "R4  accuracy sweep, all 8 modes"
$PY benchmarks/accuracy_sweep.py
log "R5  FSS modes (default, wave x3, sigma, sigma_ext, curl, shaft), 5 repeats"
$PY benchmarks/benchmark_fss_modes.py --repeats 5 --models bert_base,bert_large,gpt2_64,gpt2_128
log "R6  truncation x indexing ablation of 2-round Wave"
$PY benchmarks/benchmark_fss_modes.py --ablation --repeats 5 --models bert_base,bert_large,gpt2_64
log "R8  figures"
$PY benchmarks/plot_results.py --cpu "$OUT"
log "done"
