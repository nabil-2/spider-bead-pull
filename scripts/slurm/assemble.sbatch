#!/usr/bin/env bash
#SBATCH --job-name=spider-et-asm
#SBATCH --partition=maxcpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=data/spider_et_variation_fit_cache/logs/assemble_%j.out
#SBATCH --error=data/spider_et_variation_fit_cache/logs/assemble_%j.err

set -euo pipefail

cd "${SPIDER_ET_REPO:?SPIDER_ET_REPO is required}"

export JULIA_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

"${JULIA_EXE:-/software/julia/julia-1.12.6/bin/julia}" \
  scripts/run_spider_et_variation_fit.jl assemble \
  --cache-dir "${SPIDER_ET_CACHE_DIR:-data/spider_et_variation_fit_cache}" \
  --data-file "${SPIDER_ET_DATA_FILE:-data/ET_results_variations.npz}" \
  --results-file "${SPIDER_ET_RESULTS_FILE:-data/5_result_bf_determination_spider_ET_variations.jld2}" \
  --n-mc "${SPIDER_ET_N_MC:-100}"
