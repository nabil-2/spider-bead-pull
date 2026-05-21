#!/usr/bin/env bash
#SBATCH --job-name=spider-et-cfg
#SBATCH --partition=maxcpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --array=0-47%8
#SBATCH --output=data/spider_et_variation_fit_cache/logs/config_%A_%a.out
#SBATCH --error=data/spider_et_variation_fit_cache/logs/config_%A_%a.err

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
  scripts/run_spider_et_variation_fit.jl config \
  --config-index "${SLURM_ARRAY_TASK_ID}" \
  --cache-dir "${SPIDER_ET_CACHE_DIR:-data/spider_et_variation_fit_cache}" \
  --data-file "${SPIDER_ET_DATA_FILE:-data/ET_results_variations.npz}" \
  --n-mc "${SPIDER_ET_N_MC:-100}" \
  --seed "${SPIDER_ET_SEED:-904230}" \
  "${force_arg[@]}"
