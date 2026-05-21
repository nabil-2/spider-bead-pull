#!/usr/bin/env bash
#SBATCH --job-name=spider-et-full
#SBATCH --partition=maxcpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --output=data/spider_et_variation_fit_cache/logs/full_%j.out
#SBATCH --error=data/spider_et_variation_fit_cache/logs/full_%j.err

set -euo pipefail

cd "${SPIDER_ET_REPO:?SPIDER_ET_REPO is required}"

export JULIA_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

force_arg=()
if [[ -n "${SPIDER_ET_FORCE:-}" ]]; then
  force_arg+=(--force)
fi

"${JULIA_EXE:-/software/julia/julia-1.12.6/bin/julia}" \
  scripts/run_spider_et_variation_fit.jl full \
  --cache-dir "${SPIDER_ET_CACHE_DIR:-data/spider_et_variation_fit_cache}" \
  --data-file "${SPIDER_ET_DATA_FILE:-data/ET_results_variations.npz}" \
  --n-mc "${SPIDER_ET_N_MC:-100}" \
  --seed "${SPIDER_ET_SEED:-904230}" \
  "${force_arg[@]}"
